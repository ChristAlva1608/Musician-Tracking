#!/usr/bin/env python3
"""MotionBERT (MB_ft_h36m) benchmark: spinal flexion of P1, Camera-2, t=600..660.

- Reads COCO17 keypoints from people_id_360-Camera-2.json (P1 only).
- Converts COCO17 -> H36M17 (official coco2h36m from repo's dataset_action.py).
- Normalizes with crop_scale (same as WildDetDataset scale_range=[1,1] path).
- Feeds the 27-frame sequence as one clip, flip TTA like infer_wild.py.
- Outputs: motionbert_angles.csv + motionbert_pose3d.npz
Angle = angle between pelvis->thorax (j0->j8) and vertical (0,-1,0) in camera space (y down).
Curvature = angle between (spine-pelvis) and (thorax-spine), j7 = belly/spine mid.
"""
import os, sys, json, time, copy
import numpy as np
import torch

BENCH = os.path.dirname(os.path.abspath(__file__))
MB = os.path.join(BENCH, 'MotionBERT')
sys.path.insert(0, MB)
os.chdir(MB)  # get_config uses relative ROOT paths sometimes

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import flip_data, crop_scale

PEOPLE_JSON = os.path.join(BENCH, '..', 'person_id_christina', 'people_id_360-Camera-2.json')
T0, T1, PID = 600, 660, 1
CONF_ZERO = 0.15   # joints below this conf are zeroed (ignored by crop_scale, conf=0 to model)

# ---------- load 2D ----------
d = json.load(open(PEOPLE_JSON))
frames = []
for e in d['timeline']:
    if not (T0 <= e['t'] <= T1):
        continue
    pp = [p for p in e['people'] if p['id'] == PID]
    if not pp:
        continue
    kp = np.array(pp[0]['keypoints'], dtype=np.float64)  # (17,3) x,y,conf; x may exceed 4000 (seam wrap)
    frames.append((e['t'], kp))
frames.sort()
ts = [t for t, _ in frames]
kpts = np.stack([k for _, k in frames])  # (T,17,3)
print(f'frames: {len(ts)} ({ts[0]}..{ts[-1]})')

# x continuity: this window's coords are already continuous (3050..4040), no %4000 needed for the model
# zero out unreliable joints
low = kpts[:, :, 2] < CONF_ZERO
kpts[low] = 0.0
print('zeroed joints per frame (mean):', low.sum(1).mean())

# ---------- coco2h36m (copied from lib/data/dataset_action.py, M x T x V x C) ----------
def coco2h36m(x):
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y

# midpoint of a zeroed joint pair would be wrong -> handle conf-aware midpoints
def safe_mid(a, b):
    """midpoint of two (T,3) joints, conf-aware: if one is zeroed use the other."""
    out = (a + b) * 0.5
    only_a = (a[:, 2] > 0) & (b[:, 2] == 0)
    only_b = (b[:, 2] > 0) & (a[:, 2] == 0)
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]
    return out

h36m = coco2h36m(kpts[None])[0]  # (T,17,3)
h36m[:, 0] = safe_mid(kpts[:, 11], kpts[:, 12])          # pelvis
h36m[:, 8] = safe_mid(kpts[:, 5],  kpts[:, 6])           # thorax/neck
h36m[:, 7] = (h36m[:, 0] + h36m[:, 8]) * 0.5             # belly
h36m[:, 10] = safe_mid(kpts[:, 1], kpts[:, 2])           # head from eyes
# joints that ended up with 0 conf keep (0,0,0)
h36m[h36m[:, :, 2] == 0] = 0.0

norm2d = crop_scale(h36m.copy(), scale_range=[1, 1])     # (T,17,3) in [-1,1]

# ---------- model ----------
t_setup = time.time()
args = get_config('configs/pose3d/MB_ft_h36m.yaml')
model = load_backbone(args)
ckpt = torch.load('checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin',
                  map_location='cpu', weights_only=False)
sd = {k.replace('module.', '', 1): v for k, v in ckpt['model_pos'].items()}
model.load_state_dict(sd, strict=True)
model.eval()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
try:
    model = model.to(device)
    _ = model(torch.zeros(1, 2, 17, 3, device=device))
except Exception as ex:
    print('MPS failed, falling back to CPU:', ex)
    device = 'cpu'
    model = model.to(device)
print('device:', device, '| load time %.1fs' % (time.time() - t_setup))

x = torch.from_numpy(norm2d[None]).float().to(device)    # (1,T,17,3)
t_inf = time.time()
with torch.no_grad():
    p1 = model(x)
    p2 = flip_data(model(flip_data(x)))
    pred = ((p1 + p2) / 2)[0].cpu().numpy()              # (T,17,3), rootrel
dt = time.time() - t_inf
print('inference: %.2fs total, %.3fs/frame' % (dt, dt / len(ts)))

# ---------- angles ----------
UP = np.array([0, -1, 0.0])  # y is down in image/camera space
def ang(v, ref):
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.dot(v, ref), -1, 1))))

rows = []
for i, t in enumerate(ts):
    P = pred[i]
    pelvis, spine, thorax = P[0], P[7], P[8]
    a = ang(thorax - pelvis, UP)
    v1 = (spine - pelvis); v2 = (thorax - spine)
    curv = ang(v2, v1 / (np.linalg.norm(v1) + 1e-9))
    rows.append((t, a, curv))
    print(f't={t}: flexion={a:.1f} deg, spine_curve={curv:.1f} deg')

import csv
with open(os.path.join(BENCH, 'motionbert_angles.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t', 'angle_deg', 'spine_curve_deg', 'model'])
    for t, a, c in rows:
        w.writerow([t, f'{a:.2f}', f'{c:.2f}', 'motionbert'])

np.savez(os.path.join(BENCH, 'motionbert_pose3d.npz'),
         ts=np.array(ts), pred3d=pred, h36m2d=h36m, norm2d=norm2d)
arr = np.array([r[1] for r in rows])
print(f'MEAN flexion {arr.mean():.1f} +- {arr.std():.1f} deg')
