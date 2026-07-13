#!/usr/bin/env python3
"""Evidence montage for HMR2 (v2 — verified projection).

Projection uses the model's own quantities saved in hmr2_results.npz:
  full_xy = batch_box_center + (FOCAL/IMAGE_SIZE) * (P + pred_cam_t)[:2]/(z) * batch_box_size
This was verified to match out['pred_keypoints_2d'] (the model's own reprojection).

Overlay per tile:
  green = model's reprojected OpenPose keypoints (context; proves mesh-image alignment)
  red   = SMPL torso chain pelvis->spine1->spine2->spine3->neck (+ thin neck->head)
  blue dashed = vertical reference through pelvis
"""
import os, json, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BENCH = os.path.dirname(os.path.abspath(__file__))
FR = "/Volumes/MAML 8TB 1/Phan Dissertation Data/Christina - MultiCam Data - Piano - 2025-09-19/pose_cache/eq_360-Camera-2_%d_4000x2000.jpg"
ROLL = 1000
W, H_IMG = 4000, 2000

z = np.load(os.path.join(BENCH, 'hmr2_results.npz'))
ts = z['ts'].astype(int); joints = z['joints']; kp2d = z['kp2d']
cam_t = z['cam_t']; bbc = z['batch_box_center']; bbs = z['batch_box_size']
FOCAL = float(z['focal']); S = int(z['img_size_model'])

angles = {}
with open(os.path.join(BENCH, 'hmr2_angles.csv')) as f:
    for r in csv.DictReader(f):
        angles[int(r['t'])] = (float(r['angle_deg']), float(r['spine_curve_deg']))

d = json.load(open(os.path.join(BENCH, '..', 'person_id_christina', 'people_id_360-Camera-2.json')))
bbox = {e['t']: p['bbox'] for e in d['timeline'] for p in e['people'] if p['id'] == 1 and 600 <= e['t'] <= 660}

# OpenPose-25 bones for context (indices in pred_keypoints_2d)
OP_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (8, 12)]
PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
TORSO_CHAIN = [0, 3, 6, 9, 12]  # pelvis, spine1..3, neck (head drawn thin separately)

def dashed_line(img, p0, p1, color, thickness=2, dash=14):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    n = max(int(np.linalg.norm(p1 - p0) / dash), 1)
    for k in range(0, n, 2):
        a = p0 + (p1 - p0) * k / n
        b = p0 + (p1 - p0) * min(k + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness)

PICK = [601, 608, 617, 622, 630, 644, 651, 659]
tiles = []
for t in PICK:
    i = int(np.where(ts == t)[0][0])
    img = cv2.imread(FR % t)
    img = np.ascontiguousarray(np.roll(img, ROLL, axis=1))

    # verified projection (matches model's pred_keypoints_2d)
    P = joints[i] + cam_t[i]
    j2d = bbc[i] + (FOCAL / S) * (P[:, :2] / P[:, 2:3]) * bbs[i]
    kp = bbc[i] + kp2d[i] * bbs[i]

    x1, y1, x2, y2 = bbox[t]
    x1 = max(0, int((x1 + ROLL) % 4000) - 30); x2 = int((x2 + ROLL) % 4000) + 30
    y1 = max(0, int(y1) - 30); y2 = min(H_IMG, int(y2) + 30)
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    off = np.array([x1, y1], float)

    for a, b in OP_BONES:
        pa, pb = (kp[a] - off).astype(int), (kp[b] - off).astype(int)
        cv2.line(crop, tuple(pa), tuple(pb), (80, 220, 80), 2)
        cv2.circle(crop, tuple(pa), 4, (0, 255, 255), -1)
    # vertical reference through pelvis (blue dashed), length = pelvis->neck extent
    pel, nk, hd = j2d[0] - off, j2d[12] - off, j2d[15] - off
    L = np.linalg.norm(nk - pel)
    dashed_line(crop, pel, pel - np.array([0, L]), (255, 120, 0), 3)
    for a, b in zip(TORSO_CHAIN[:-1], TORSO_CHAIN[1:]):
        cv2.line(crop, tuple((j2d[a] - off).astype(int)), tuple((j2d[b] - off).astype(int)), (0, 0, 255), 6)
    cv2.line(crop, tuple(nk.astype(int)), tuple(hd.astype(int)), (0, 0, 255), 2)
    for a in TORSO_CHAIN:
        cv2.circle(crop, tuple((j2d[a] - off).astype(int)), 6, (255, 255, 255), -1)
        cv2.circle(crop, tuple((j2d[a] - off).astype(int)), 6, (0, 0, 200), 2)

    HT = 640
    crop = cv2.resize(crop, (int(crop.shape[1] * HT / crop.shape[0]), HT))

    Pj = joints[i]
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 3.2), dpi=100)
    for ax, (h_ax, name) in zip(axes, [(0, 'front (x-y)'), (2, 'side (z-y)')]):
        for c, p in enumerate(PARENTS):
            if p < 0: continue
            tors = (c in TORSO_CHAIN + [15] and p in TORSO_CHAIN + [15])
            col, lw = ('r', 3) if tors else ('g', 1)
            ax.plot([Pj[c, h_ax], Pj[p, h_ax]], [-Pj[c, 1], -Pj[p, 1]], col, lw=lw)
        ax.plot([Pj[0, h_ax], Pj[0, h_ax]], [-Pj[0, 1], -Pj[0, 1] + 0.6], 'b--', lw=1)
        ax.set_title(name, fontsize=8); ax.set_aspect('equal'); ax.axis('off')
    a_deg, c_deg = angles[t]
    fig.suptitle(f't={t}  flexion {a_deg:.0f}°  curve {c_deg:.0f}°', fontsize=11)
    fig.tight_layout(); fig.canvas.draw()
    plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1]
    plt.close(fig)
    plot = cv2.resize(plot, (int(plot.shape[1] * (HT // 2) / plot.shape[0]), HT // 2))
    pw = plot.shape[1]
    right = np.full((HT, pw, 3), 255, np.uint8)
    right[:HT // 2] = plot
    cv2.putText(right, f't={t}', (10, HT // 2 + 40), 0, 1.0, (0, 0, 0), 2)
    cv2.putText(right, f'flexion {a_deg:.1f} deg', (10, HT // 2 + 90), 0, 0.9, (0, 0, 255), 2)
    cv2.putText(right, f'curve   {c_deg:.1f} deg', (10, HT // 2 + 135), 0, 0.9, (0, 128, 0), 2)
    cv2.putText(right, 'red: SMPL spine (proj)', (10, HT // 2 + 185), 0, 0.55, (0, 0, 255), 1)
    cv2.putText(right, 'green: HMR2 reproj kpts', (10, HT // 2 + 215), 0, 0.55, (0, 128, 0), 1)
    cv2.putText(right, 'blue dash: vertical ref', (10, HT // 2 + 245), 0, 0.55, (200, 100, 0), 1)
    tiles.append(np.hstack([crop, right]))

maxw = max(tl.shape[1] for tl in tiles)
tiles = [np.hstack([tl, np.full((tl.shape[0], maxw - tl.shape[1], 3), 255, np.uint8)]) for tl in tiles]
row1 = np.hstack(tiles[:4]); row2 = np.hstack(tiles[4:])
if row2.shape[1] < row1.shape[1]:
    row2 = np.hstack([row2, np.full((row2.shape[0], row1.shape[1] - row2.shape[1], 3), 255, np.uint8)])
mont = np.vstack([row1, row2])
out = os.path.join(BENCH, 'montage_hmr2_P1.jpg')
cv2.imwrite(out, mont, [cv2.IMWRITE_JPEG_QUALITY, 90])
print('saved', out, mont.shape)
