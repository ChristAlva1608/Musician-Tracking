#!/usr/bin/env python3
"""4D-Humans / HMR2.0 benchmark: spinal flexion of P1, Camera-2, t=600..660.

Uses existing bboxes (no detectron2). Crops via ViTDetDataset, runs HMR2,
computes SMPL native joints (J_regressor @ vertices) in camera space (y down).
flexion = angle(pelvis(0)->neck(12), up=(0,-1,0))
curve   = angle between (spine2(6)-pelvis) and (neck-spine2)
Outputs hmr2_angles.csv + hmr2_results.npz
"""
import os, sys, json, time
os.environ['PYOPENGL_PLATFORM'] = 'darwin'  # hmr2 forces 'egl' otherwise -> crash on macOS
import numpy as np
import cv2
import torch

BENCH = os.path.dirname(os.path.abspath(__file__))
FR = "/Volumes/MAML 8TB 1/Phan Dissertation Data/Christina - MultiCam Data - Piano - 2025-09-19/pose_cache/eq_360-Camera-2_%d_4000x2000.jpg"
ROLL = 1000
T0, T1, PID = 600, 660, 1

DEVICE_PREF = os.environ.get('HMR2_DEVICE', 'mps')

d = json.load(open(os.path.join(BENCH, '..', 'person_id_christina', 'people_id_360-Camera-2.json')))
frames = []
for e in d['timeline']:
    if not (T0 <= e['t'] <= T1):
        continue
    pp = [p for p in e['people'] if p['id'] == PID]
    if pp:
        frames.append((e['t'], np.array(pp[0]['bbox'], dtype=np.float32)))
frames.sort()
print('frames:', len(frames))

t_setup = time.time()
# torch>=2.6 defaults weights_only=True; official ckpt contains omegaconf objects -> force False (trusted source)
_orig_load = torch.load
def _load(*a, **k):
    k['weights_only'] = False
    return _orig_load(*a, **k)
torch.load = _load

import hmr2.models.hmr2 as hmr2_mod
class _NoRenderer:  # OffscreenRenderer unsupported on macOS; we project joints ourselves
    def __init__(self, *a, **k): pass
hmr2_mod.MeshRenderer = _NoRenderer
hmr2_mod.SkeletonRenderer = _NoRenderer
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset
model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
model.eval()
device = DEVICE_PREF if (DEVICE_PREF == 'cpu' or torch.backends.mps.is_available()) else 'cpu'
try:
    model = model.to(device)
except Exception as ex:
    print('to(device) failed:', ex); device = 'cpu'; model = model.to(device)
print('model loaded in %.1fs, device=%s' % (time.time() - t_setup, device))

J_reg = model.smpl.J_regressor.cpu().numpy()  # (24,6890)
UP = np.array([0, -1, 0.0])
def ang(v, ref):
    v = v / (np.linalg.norm(v) + 1e-9)
    ref = ref / (np.linalg.norm(ref) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.dot(v, ref), -1, 1))))

rows, verts_all, joints_all, cams, centers, sizes = [], [], [], [], [], []
kp2d_all, camt_all, bc_all, bs_all = [], [], [], []
t_all = time.time()
for t, bb in frames:
    img = cv2.imread(FR % t)
    img = np.ascontiguousarray(np.roll(img, ROLL, axis=1))
    b = bb.copy(); b[0] = (b[0] + ROLL) % 4000; b[2] = (b[2] + ROLL) % 4000
    ds = ViTDetDataset(model_cfg, img, b[None])
    batch = ds[0]
    def _to_t(v):
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64: v = v.astype(np.float32)
            return torch.from_numpy(v[None]).to(device)
        tt = torch.tensor([v])
        if tt.dtype == torch.float64: tt = tt.float()
        return tt.to(device)
    batch = {k: _to_t(v) for k, v in batch.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model(batch)
    dt = time.time() - t0
    verts = out['pred_vertices'][0].cpu().numpy()          # (6890,3) camera space
    J = J_reg @ verts                                       # (24,3) native SMPL joints
    pelvis, sp1, sp2, sp3, neck, head = J[0], J[3], J[6], J[9], J[12], J[15]
    flex = ang(neck - pelvis, UP)
    curve = ang(neck - sp2, sp2 - pelvis)
    rows.append((float(t), flex, curve, dt))
    verts_all.append(verts); joints_all.append(J)
    cams.append(out['pred_cam'][0].cpu().numpy())
    centers.append(ds.center[0]); sizes.append(ds.scale[0] * 200)
    kp2d_all.append(out['pred_keypoints_2d'][0].cpu().numpy())   # (44,2) crop-normalized [-0.5,0.5]
    camt_all.append(out['pred_cam_t'][0].cpu().numpy())          # model's crop-space translation
    bc_all.append(batch['box_center'][0].cpu().numpy().astype(np.float64))
    bs_all.append(float(batch['box_size'][0].cpu().numpy()))     # exact square crop side used
    print(f't={int(t)}: flexion={flex:.1f} curve={curve:.1f}  ({dt:.2f}s)')

tot = time.time() - t_all
print('total %.1fs, %.2fs/frame (incl. IO)' % (tot, tot / len(frames)))

import csv
with open(os.path.join(BENCH, 'hmr2_angles.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t', 'angle_deg', 'spine_curve_deg', 'model'])
    for t, a, c, _ in rows:
        w.writerow([int(t), f'{a:.2f}', f'{c:.2f}', 'hmr2'])

np.savez(os.path.join(BENCH, 'hmr2_results.npz'),
         ts=np.array([r[0] for r in rows]), verts=np.stack(verts_all),
         joints=np.stack(joints_all), pred_cam=np.stack(cams),
         box_center=np.stack(centers), box_size=np.stack(sizes),
         kp2d=np.stack(kp2d_all), cam_t=np.stack(camt_all),
         batch_box_center=np.stack(bc_all), batch_box_size=np.array(bs_all),
         faces=model.smpl.faces.astype(np.int64),
         focal=float(model_cfg.EXTRA.FOCAL_LENGTH), img_size_model=int(model_cfg.MODEL.IMAGE_SIZE))
arr = np.array([r[1] for r in rows])
print(f'MEAN flexion {arr.mean():.1f} +- {arr.std():.1f} deg | device={device}')
