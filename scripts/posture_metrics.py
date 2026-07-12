#!/usr/bin/env python3
"""
Compute per-person posture metrics from people_id_*.json timelines
(produced by person_id_from_cache.py).

Metrics per frame, per person (2D screen-space; the subject is mostly
stationary within a session so trends over time are meaningful even in
equirectangular projection):
- trunk_lean_deg   : mid_hip -> mid_shoulder line vs vertical (0 = upright,
                     positive = leaning toward image-right)
- head_forward     : horizontal ear-to-shoulder offset / torso length
                     (turtle-neck proxy; higher = head further forward)
- shoulder_tilt_deg: shoulder line vs horizontal (asymmetry)

Output: one CSV per input JSON + a combined summary printed to stdout.

Usage:
    python3 scripts/posture_metrics.py src/output/person_id_christina/*.json
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
MIN_CONF = 0.3


def midpoint(kp, a, b):
    if kp[a][2] >= MIN_CONF and kp[b][2] >= MIN_CONF:
        return (np.asarray(kp[a][:2]) + np.asarray(kp[b][:2])) / 2
    for i in (a, b):
        if kp[i][2] >= MIN_CONF:
            return np.asarray(kp[i][:2])
    return None


def metrics(kp):
    sh = midpoint(kp, L_SH, R_SH)
    hip = midpoint(kp, L_HIP, R_HIP)
    ear = midpoint(kp, L_EAR, R_EAR)
    out = {}
    if sh is not None and hip is not None:
        v = sh - hip                       # points upward on screen (y down)
        torso = np.linalg.norm(v)
        if torso > 10:
            out['trunk_lean_deg'] = round(
                float(np.degrees(np.arctan2(v[0], -v[1]))), 2)
            if ear is not None:
                out['head_forward'] = round(float((ear[0] - sh[0]) / torso), 3)
    if kp[L_SH][2] >= MIN_CONF and kp[R_SH][2] >= MIN_CONF:
        d = np.asarray(kp[R_SH][:2]) - np.asarray(kp[L_SH][:2])
        torso = np.linalg.norm(sh - hip) if (sh is not None and hip is not None) else 0
        # profile views collapse the shoulder line in 2D: tilt is meaningless
        if torso > 10 and abs(d[0]) > 0.3 * torso:
            deg = float(np.degrees(np.arctan2(d[1], d[0])))
            out['shoulder_tilt_deg'] = round(((deg + 90) % 180) - 90, 2)
    return out


def main():
    files = [Path(p) for p in sys.argv[1:] if p.endswith('.json')]
    if not files:
        print(__doc__); sys.exit(1)

    for f in files:
        data = json.loads(f.read_text())
        rows = []
        for entry in data['timeline']:
            for person in entry['people']:
                m = metrics(person['keypoints'])
                if m:
                    rows.append({'t': entry['t'], 'person': person['id'],
                                 'conf': person['conf'], **m})
        out_csv = f.with_suffix('.posture.csv')
        cols = ['t', 'person', 'conf', 'trunk_lean_deg', 'head_forward',
                'shoulder_tilt_deg']
        with open(out_csv, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)

        print(f"\n{f.name} -> {out_csv.name} ({len(rows)} rows)")
        for pid in (1, 2):
            sub = [r for r in rows if r['person'] == pid]
            if not sub:
                continue
            for m in ('trunk_lean_deg', 'head_forward', 'shoulder_tilt_deg'):
                vals = np.array([r[m] for r in sub if m in r])
                if len(vals):
                    print(f"  P{pid} {m:18s} n={len(vals):4d} "
                          f"median={np.median(vals):7.2f} "
                          f"p10={np.percentile(vals, 10):7.2f} "
                          f"p90={np.percentile(vals, 90):7.2f}")


if __name__ == '__main__':
    main()
