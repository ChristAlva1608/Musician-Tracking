#!/usr/bin/env python3
"""Montage for dewarped HMR2 runs. Same style as montage_hmr2.py:
green = model's reprojected keypoints, red = SMPL spine chain, blue dashed = TRUE gravity
(world-up line through pelvis, pitch-corrected: up_cam = (0,-cos(phi0),-sin(phi0))).
Usage: montage_dewarp.py <job>   (cam3_p2 | cam2_p1)
"""
import os, sys, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

JOB = sys.argv[1] if len(sys.argv) > 1 else 'cam3_p2'
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, f'dewarp_{JOB}.npz'))
ts = z['ts'].astype(int); joints = z['joints']; kp2d = z['kp2d']
cam_t = z['cam_t']; bbc = z['bbc']; bbs = z['bbs']
FOCAL = float(z['focal']); S = int(z['S']); phi0 = float(z['phi0'])
UP = np.array([0, -np.cos(phi0), -np.sin(phi0)])

angles = {}
with open(os.path.join(HERE, f'dewarp_{JOB}.csv')) as f:
    for r in csv.DictReader(f):
        angles[int(r['t'])] = (float(r['angle_deg']), float(r['spine_curve_deg']))

OP_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
CHAIN = [0, 3, 6, 9, 12]

def dashed(img, p0, p1, color, th=3, dash=14):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    n = max(int(np.linalg.norm(p1 - p0) / dash), 1)
    for k in range(0, n, 2):
        a = p0 + (p1 - p0) * k / n; b = p0 + (p1 - p0) * min(k + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, th)

idx = np.linspace(0, len(ts) - 1, 8).round().astype(int)
tiles = []
for i in idx:
    t = int(ts[i])
    img = cv2.imread(os.path.join(HERE, f'frames_{JOB}', f'persp_{t}.jpg'))
    P = joints[i] + cam_t[i]
    proj = lambda Q: bbc[i] + (FOCAL / S) * (Q[..., :2] / Q[..., 2:3]) * bbs[i]
    j2d = proj(P)
    kp = bbc[i] + kp2d[i] * bbs[i]
    crop = np.ascontiguousarray(img)
    for a, b in OP_BONES:
        cv2.line(crop, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (80, 220, 80), 2)
        cv2.circle(crop, tuple(kp[a].astype(int)), 3, (0, 255, 255), -1)
    # true gravity line through pelvis (projected 3D vertical segment)
    L3 = np.linalg.norm(joints[i][12] - joints[i][0])
    g0, g1 = proj(P[0]), proj(P[0] + L3 * UP)
    dashed(crop, g0, g1, (255, 120, 0), 3)
    for a, b in zip(CHAIN[:-1], CHAIN[1:]):
        cv2.line(crop, tuple(j2d[a].astype(int)), tuple(j2d[b].astype(int)), (0, 0, 255), 5)
    cv2.line(crop, tuple(j2d[12].astype(int)), tuple(j2d[15].astype(int)), (0, 0, 255), 2)
    for a in CHAIN:
        cv2.circle(crop, tuple(j2d[a].astype(int)), 5, (255, 255, 255), -1)
        cv2.circle(crop, tuple(j2d[a].astype(int)), 5, (0, 0, 200), 2)
    HT = 640
    crop = cv2.resize(crop, (int(crop.shape[1] * HT / crop.shape[0]), HT))

    Pj = joints[i]
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 3.0), dpi=100)
    for ax, (h_ax, nm) in zip(axes, [(0, 'front'), (2, 'side (z-y)')]):
        for c, p in enumerate(PARENTS):
            if p < 0: continue
            tors = (c in CHAIN + [15] and p in CHAIN + [15])
            col, lw = ('r', 3) if tors else ('g', 1)
            ax.plot([Pj[c, h_ax], Pj[p, h_ax]], [-Pj[c, 1], -Pj[p, 1]], col, lw=lw)
        up2 = {0: (UP[0], -UP[1]), 2: (UP[2], -UP[1])}[h_ax]
        ax.plot([Pj[0, h_ax], Pj[0, h_ax] + 0.6 * up2[0]], [-Pj[0, 1], -Pj[0, 1] + 0.6 * up2[1]], 'b--', lw=1)
        ax.set_title(nm, fontsize=8); ax.set_aspect('equal'); ax.axis('off')
    a_deg, c_deg = angles[t]
    fig.suptitle(f't={t}', fontsize=10)
    fig.tight_layout(); fig.canvas.draw()
    plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1]
    plt.close(fig)
    pw = int(plot.shape[1] * (HT // 2) / plot.shape[0])
    plot = cv2.resize(plot, (pw, HT // 2))
    right = np.full((HT, pw, 3), 255, np.uint8)
    right[:HT // 2] = plot
    cv2.putText(right, f't={t}', (10, HT // 2 + 40), 0, 1.0, (0, 0, 0), 2)
    cv2.putText(right, f'flexion {a_deg:.1f} deg', (10, HT // 2 + 90), 0, 0.85, (0, 0, 255), 2)
    cv2.putText(right, f'curve   {c_deg:.1f} deg', (10, HT // 2 + 135), 0, 0.85, (0, 128, 0), 2)
    cv2.putText(right, '(gravity-corrected up)', (10, HT // 2 + 175), 0, 0.5, (0, 0, 0), 1)
    cv2.putText(right, 'red: SMPL spine', (10, HT // 2 + 215), 0, 0.55, (0, 0, 255), 1)
    cv2.putText(right, 'green: reproj kpts', (10, HT // 2 + 245), 0, 0.55, (0, 128, 0), 1)
    cv2.putText(right, 'blue: true vertical', (10, HT // 2 + 275), 0, 0.55, (200, 100, 0), 1)
    tiles.append(np.hstack([crop, right]))

maxw = max(tl.shape[1] for tl in tiles)
tiles = [np.hstack([tl, np.full((tl.shape[0], maxw - tl.shape[1], 3), 255, np.uint8)]) for tl in tiles]
row1 = np.hstack(tiles[:4]); row2 = np.hstack(tiles[4:])
if row2.shape[1] < row1.shape[1]:
    row2 = np.hstack([row2, np.full((row2.shape[0], row1.shape[1] - row2.shape[1], 3), 255, np.uint8)])
mont = np.vstack([row1, row2])
out = os.path.join(HERE, f'montage_dewarp_{JOB}.jpg')
cv2.imwrite(out, mont, [cv2.IMWRITE_JPEG_QUALITY, 90])
print('saved', out, mont.shape)
