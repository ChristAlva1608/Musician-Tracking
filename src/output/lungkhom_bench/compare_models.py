#!/usr/bin/env python3
"""Merge both CSVs, compute agreement stats, write combined CSV + comparison chart."""
import os, csv
import numpy as np

BENCH = os.path.dirname(os.path.abspath(__file__))

def load(name):
    out = {}
    with open(os.path.join(BENCH, name)) as f:
        for r in csv.DictReader(f):
            out[int(r['t'])] = (float(r['angle_deg']), float(r['spine_curve_deg']))
    return out

mb, hm = load('motionbert_angles.csv'), load('hmr2_angles.csv')
ts = sorted(set(mb) & set(hm))
a_mb = np.array([mb[t][0] for t in ts]); c_mb = np.array([mb[t][1] for t in ts])
a_hm = np.array([hm[t][0] for t in ts]); c_hm = np.array([hm[t][1] for t in ts])

with open(os.path.join(BENCH, 'lungkhom_angles.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t', 'angle_deg', 'spine_curve_deg', 'model'])
    for t in ts:
        w.writerow([t, f'{mb[t][0]:.2f}', f'{mb[t][1]:.2f}', 'motionbert'])
    for t in ts:
        w.writerow([t, f'{hm[t][0]:.2f}', f'{hm[t][1]:.2f}', 'hmr2'])

r = np.corrcoef(a_mb, a_hm)[0, 1]
rc = np.corrcoef(c_mb, c_hm)[0, 1]
diff = a_mb - a_hm
print(f'n={len(ts)}')
print(f'MotionBERT flexion: {a_mb.mean():.1f} +- {a_mb.std():.1f} (min {a_mb.min():.1f}, max {a_mb.max():.1f})')
print(f'HMR2       flexion: {a_hm.mean():.1f} +- {a_hm.std():.1f} (min {a_hm.min():.1f}, max {a_hm.max():.1f})')
print(f'MotionBERT curve:   {c_mb.mean():.1f} +- {c_mb.std():.1f}')
print(f'HMR2       curve:   {c_hm.mean():.1f} +- {c_hm.std():.1f}')
print(f'Pearson r (flexion): {r:.3f} | (curve): {rc:.3f}')
print(f'mean |diff| flexion: {np.abs(diff).mean():.1f} deg, bias (MB-HMR2): {diff.mean():+.1f} deg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), dpi=130, sharex=True)
C_MB, C_HM = '#4059ad', '#e07a2f'
for ax, (ymb, yhm, ttl) in zip(axes, [(a_mb, a_hm, 'Flexion: pelvis-to-thorax/neck vs vertical (deg)'),
                                      (c_mb, c_hm, 'Spine curvature (deg)')]):
    ax.plot(ts, ymb, 'o-', color=C_MB, ms=4, lw=1.4, label='MotionBERT (H36M lift)')
    ax.plot(ts, yhm, 's-', color=C_HM, ms=4, lw=1.4, label='HMR2 / 4D-Humans (SMPL)')
    ax.set_title(ttl, fontsize=10, loc='left')
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(frameon=False, fontsize=9)
axes[1].set_xlabel('t (s), Camera-2, P1')
fig.suptitle(f'Spinal flexion P1 (piano), t=600-660 — Pearson r = {r:.2f}', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(BENCH, 'compare_flexion.png'))
print('saved compare_flexion.png')
