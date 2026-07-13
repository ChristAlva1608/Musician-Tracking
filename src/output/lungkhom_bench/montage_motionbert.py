#!/usr/bin/env python3
"""Evidence montage for MotionBERT: crop + 2D h36m overlay + 3D side/front views + angle."""
import os, json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BENCH = os.path.dirname(os.path.abspath(__file__))
FR = "/Volumes/MAML 8TB 1/Phan Dissertation Data/Christina - MultiCam Data - Piano - 2025-09-19/pose_cache/eq_360-Camera-2_%d_4000x2000.jpg"
ROLL = 1000

z = np.load(os.path.join(BENCH, 'motionbert_pose3d.npz'))
ts, pred, h36m2d = z['ts'], z['pred3d'], z['h36m2d']
import csv
angles = {}
with open(os.path.join(BENCH, 'motionbert_angles.csv')) as f:
    for r in csv.DictReader(f):
        angles[int(r['t'])] = (float(r['angle_deg']), float(r['spine_curve_deg']))

d = json.load(open(os.path.join(BENCH, '..', 'person_id_christina', 'people_id_360-Camera-2.json')))
bbox = {}
for e in d['timeline']:
    for p in e['people']:
        if p['id'] == 1 and 600 <= e['t'] <= 660:
            bbox[e['t']] = p['bbox']

BONES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
         (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
TORSO = {(0,7),(7,8)}

PICK = [601, 611, 622, 630, 641, 650, 659]
tiles = []
for t in PICK:
    i = int(np.where(ts == t)[0][0])
    img = cv2.imread(FR % t)
    img = np.ascontiguousarray(np.roll(img, ROLL, axis=1))
    x1, y1, x2, y2 = bbox[t]
    x1 = (x1 + ROLL) % 4000; x2 = (x2 + ROLL) % 4000
    x1, y1, x2, y2 = int(x1) - 30, int(y1) - 30, int(x2) + 30, int(y2) + 30
    x1, y1 = max(0, x1), max(0, y1); y2 = min(2000, y2)
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    k2 = h36m2d[i].copy()
    k2[:, 0] = (k2[:, 0] + ROLL) % 4000
    for a, b in BONES:
        if k2[a, 2] > 0 and k2[b, 2] > 0:
            col = (0, 0, 255) if (a, b) in TORSO else (0, 255, 0)
            w = 6 if (a, b) in TORSO else 2
            cv2.line(crop, (int(k2[a,0]-x1), int(k2[a,1]-y1)), (int(k2[b,0]-x1), int(k2[b,1]-y1)), col, w)
    H = 640
    crop = cv2.resize(crop, (int(crop.shape[1] * H / crop.shape[0]), H))

    # 3D side view (z-y plane) — flexion is most visible from the side
    P = pred[i]
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 3.2), dpi=100)
    for ax, (h_ax, name) in zip(axes, [(0, 'front (x-y)'), (2, 'side (z-y)')]):
        for a, b in BONES:
            col = 'r' if (a, b) in TORSO else 'g'
            lw = 3 if (a, b) in TORSO else 1
            ax.plot([P[a, h_ax], P[b, h_ax]], [-P[a, 1], -P[b, 1]], col, lw=lw)
        # vertical reference through pelvis
        ax.plot([P[0, h_ax], P[0, h_ax]], [-P[0, 1], -P[0, 1] + 0.6], 'b--', lw=1)
        ax.set_title(name, fontsize=8); ax.set_aspect('equal'); ax.axis('off')
    a_deg, c_deg = angles[t]
    fig.suptitle(f't={t}  flexion {a_deg:.0f}°  curve {c_deg:.0f}°', fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1]
    plt.close(fig)
    plot = cv2.resize(plot, (int(plot.shape[1] * (H // 2) / plot.shape[0]), H // 2))
    # stack: crop on left, plot on right (pad to H)
    pw = plot.shape[1]
    right = np.full((H, pw, 3), 255, np.uint8)
    right[:H // 2] = plot
    cv2.putText(right, f't={t}', (10, H // 2 + 40), 0, 1.0, (0, 0, 0), 2)
    cv2.putText(right, f'flexion {a_deg:.1f} deg', (10, H // 2 + 90), 0, 0.9, (0, 0, 255), 2)
    cv2.putText(right, f'curve   {c_deg:.1f} deg', (10, H // 2 + 135), 0, 0.9, (0, 128, 0), 2)
    tiles.append(np.hstack([crop, right]))

maxw = max(tl.shape[1] for tl in tiles)
tiles = [np.hstack([tl, np.full((tl.shape[0], maxw - tl.shape[1], 3), 255, np.uint8)]) for tl in tiles]
row1 = np.hstack(tiles[:4]); row2 = np.hstack(tiles[4:])
row2 = np.hstack([row2, np.full((row2.shape[0], row1.shape[1] - row2.shape[1], 3), 255, np.uint8)])
mont = np.vstack([row1, row2])
out = os.path.join(BENCH, 'montage_motionbert_P1.jpg')
cv2.imwrite(out, mont, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('saved', out, mont.shape)
