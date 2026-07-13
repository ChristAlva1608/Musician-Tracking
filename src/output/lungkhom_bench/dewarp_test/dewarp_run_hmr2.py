#!/usr/bin/env python3
"""Decisive test: gnomonic (perspective) dewarp of equirect before HMR2.

Jobs:
  A) cam3_p2 : Camera-3, P2 (student on bench, clean full-body side view), t=882..950
  B) cam2_p1 : Camera-2, P1 (teacher at piano, occluded hips), t=600..660  [bonus rerun]

Pipeline per job:
  - fixed view center (union bbox center -> lon/lat) + fixed fov (union bbox + margin)
  - equirect -> perspective via ray grid + cv2.remap (rolled by 1000 to avoid seam)
  - person bbox in persp image from forward-projected COCO keypoints
  - HMR2 on persp image with that bbox
  - flexion vs TRUE gravity up in the persp camera frame: up_cam = (0,-cos(phi0),-sin(phi0))
    (also report naive up (0,-1,0) for bias comparison)
Outputs: dewarp_test/frames_<job>/*.jpg, dewarp_<job>.csv, dewarp_<job>.npz
"""
import os, json, time, csv
os.environ['PYOPENGL_PLATFORM'] = 'darwin'
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
POSE = "/Volumes/MAML 8TB 1/Phan Dissertation Data/Christina - MultiCam Data - Piano - 2025-09-19/pose_cache/eq_360-Camera-%d_%d_4000x2000.jpg"
PID_JSON = os.path.join(BENCH, '..', 'person_id_christina', 'people_id_360-Camera-%d.json')
W_EQ, H_EQ, ROLL = 4000, 2000, 1000
OUT_W, OUT_H = 768, 1024

JOBS = {
    'cam3_p2': dict(cam=3, pid=2, t0=882, t1=950),
    'cam2_p1': dict(cam=2, pid=1, t0=600, t1=660),
}

def theta_phi(x, y):
    """equirect px -> (lon, lat_down). x may exceed W_EQ (wrap)."""
    return (x / W_EQ) * 2 * np.pi - np.pi, (y / H_EQ) * np.pi - np.pi / 2

def rot(theta0, phi0):
    c, s = np.cos(phi0), np.sin(phi0)
    Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    c, s = np.cos(theta0), np.sin(theta0)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Ry @ Rx  # d_world = R @ d_cam ; R @ (0,0,1) = view center

def build_maps(theta0, phi0, f_p):
    R = rot(theta0, phi0)
    u, v = np.meshgrid(np.arange(OUT_W), np.arange(OUT_H))
    d = np.stack([(u - OUT_W / 2) / f_p, (v - OUT_H / 2) / f_p, np.ones_like(u, float)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    dw = d @ R.T
    th = np.arctan2(dw[..., 0], dw[..., 2])
    ph = np.arcsin(np.clip(dw[..., 1], -1, 1))
    map_x = (((th + np.pi) / (2 * np.pi)) * W_EQ + ROLL) % W_EQ   # rolled-image coords
    map_y = (ph + np.pi / 2) / np.pi * H_EQ
    return map_x.astype(np.float32), map_y.astype(np.float32), R

def eq_to_persp(pts_xy, R, f_p):
    """equirect px (unrolled, may exceed W_EQ) -> persp px"""
    th, ph = theta_phi(pts_xy[:, 0], pts_xy[:, 1])
    dw = np.stack([np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th)], -1)
    dc = dw @ R  # R^T @ dw
    uv = np.stack([f_p * dc[:, 0] / dc[:, 2] + OUT_W / 2, f_p * dc[:, 1] / dc[:, 2] + OUT_H / 2], -1)
    return uv

# ---------- prepare jobs (dewarp frames + bboxes) ----------
prepared = {}
for name, jb in JOBS.items():
    d = json.load(open(PID_JSON % jb['cam']))
    frames = []
    for e in d['timeline']:
        if not (jb['t0'] <= e['t'] <= jb['t1']):
            continue
        pp = [p for p in e['people'] if p['id'] == jb['pid']]
        if pp and os.path.exists(POSE % (jb['cam'], e['t'])):
            frames.append((e['t'], np.array(pp[0]['bbox']), np.array(pp[0]['keypoints'])))
    frames.sort()
    allb = np.array([f[1] for f in frames])
    ux1, uy1, ux2, uy2 = allb[:, 0].min(), allb[:, 1].min(), allb[:, 2].max(), allb[:, 3].max()
    th0, ph0 = theta_phi((ux1 + ux2) / 2, (uy1 + uy2) / 2)
    dphi = (uy2 - uy1) / H_EQ * np.pi
    fov_y = min(dphi * 1.35, np.radians(125))
    f_p = (OUT_H / 2) / np.tan(fov_y / 2)
    mx, my, R = build_maps(th0, ph0, f_p)
    os.makedirs(os.path.join(HERE, f'frames_{name}'), exist_ok=True)
    jobs_f = []
    for t, bb, kp in frames:
        img = cv2.imread(POSE % (jb['cam'], t))
        img = np.ascontiguousarray(np.roll(img, ROLL, axis=1))
        persp = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        good = kp[:, 2] > 0.2
        uv = eq_to_persp(kp[good, :2], R, f_p)
        x1, y1 = uv.min(0); x2, y2 = uv.max(0)
        mgx, mgy = 0.12 * (x2 - x1), 0.08 * (y2 - y1)
        pb = np.array([max(0, x1 - mgx), max(0, y1 - mgy), min(OUT_W, x2 + mgx), min(OUT_H, y2 + mgy)], dtype=np.float32)
        fp_out = os.path.join(HERE, f'frames_{name}', f'persp_{t}.jpg')
        cv2.imwrite(fp_out, persp, [cv2.IMWRITE_JPEG_QUALITY, 95])
        jobs_f.append((t, fp_out, pb))
    prepared[name] = dict(frames=jobs_f, phi0=ph0, th0=th0, f_p=f_p, fov_deg=np.degrees(fov_y))
    print(f'{name}: {len(jobs_f)} frames, center lon={np.degrees(th0):.1f} lat_down={np.degrees(ph0):.1f} deg, fov_y={np.degrees(fov_y):.1f} deg, f_p={f_p:.0f}px')

# ---------- HMR2 ----------
_orig_load = torch.load
def _load(*a, **k):
    k['weights_only'] = False
    return _orig_load(*a, **k)
torch.load = _load
import hmr2.models.hmr2 as hmr2_mod
class _NoRenderer:
    def __init__(self, *a, **k): pass
hmr2_mod.MeshRenderer = _NoRenderer
hmr2_mod.SkeletonRenderer = _NoRenderer
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset

model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
model.eval()
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device)
J_reg = model.smpl.J_regressor.cpu().numpy()

def ang(v, ref):
    v = v / (np.linalg.norm(v) + 1e-9); ref = ref / (np.linalg.norm(ref) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.dot(v, ref), -1, 1))))

for name, pr in prepared.items():
    phi0 = pr['phi0']
    UP_TRUE = np.array([0, -np.cos(phi0), -np.sin(phi0)])
    UP_NAIVE = np.array([0, -1, 0.0])
    rows, store = [], dict(ts=[], joints=[], kp2d=[], cam_t=[], bbc=[], bbs=[])
    for t, fp, pb in pr['frames']:
        img = cv2.imread(fp)
        ds = ViTDetDataset(model_cfg, img, pb[None])
        batch = ds[0]
        def _to_t(v):
            if isinstance(v, np.ndarray):
                if v.dtype == np.float64: v = v.astype(np.float32)
                return torch.from_numpy(v[None]).to(device)
            tt = torch.tensor([v])
            if tt.dtype == torch.float64: tt = tt.float()
            return tt.to(device)
        batch = {k: _to_t(v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        verts = out['pred_vertices'][0].cpu().numpy()
        J = J_reg @ verts
        pelvis, sp2, neck = J[0], J[6], J[12]
        flex_true = ang(neck - pelvis, UP_TRUE)
        flex_naive = ang(neck - pelvis, UP_NAIVE)
        curve = ang(neck - sp2, sp2 - pelvis)
        rows.append((t, flex_true, flex_naive, curve))
        store['ts'].append(t); store['joints'].append(J)
        store['kp2d'].append(out['pred_keypoints_2d'][0].cpu().numpy())
        store['cam_t'].append(out['pred_cam_t'][0].cpu().numpy())
        store['bbc'].append(batch['box_center'][0].cpu().numpy()); store['bbs'].append(float(batch['box_size'][0].cpu().numpy()))
        print(f'{name} t={t}: flex_true={flex_true:.1f} flex_naive={flex_naive:.1f} curve={curve:.1f}')
    with open(os.path.join(HERE, f'dewarp_{name}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['t', 'angle_deg', 'angle_naiveup_deg', 'spine_curve_deg', 'model'])
        for t, a, an, c in rows:
            w.writerow([t, f'{a:.2f}', f'{an:.2f}', f'{c:.2f}', 'hmr2_dewarp'])
    np.savez(os.path.join(HERE, f'dewarp_{name}.npz'),
             ts=np.array(store['ts']), joints=np.stack(store['joints']),
             kp2d=np.stack(store['kp2d']), cam_t=np.stack(store['cam_t']),
             bbc=np.stack(store['bbc']), bbs=np.array(store['bbs']),
             phi0=phi0, focal=float(model_cfg.EXTRA.FOCAL_LENGTH), S=int(model_cfg.MODEL.IMAGE_SIZE))
    a = np.array([r[1] for r in rows]); an = np.array([r[2] for r in rows])
    print(f'== {name}: flex_TRUEup {a.mean():.1f} +- {a.std():.1f} | flex_naive {an.mean():.1f} +- {an.std():.1f} | phi0={np.degrees(phi0):.1f} deg')
