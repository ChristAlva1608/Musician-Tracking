#!/usr/bin/env python3
"""
Global ID-relabel QC pass over a people_id_<camera>.json produced by
person_id_from_cache.py.

The online assigner sees each frame once; a single early mistake (two people
close together, similar colors) can swap P1/P2 PERSISTENTLY. Offline we know
the whole session, so identity is decided globally:

1. Extract a clothing feature for every assigned detection (from cache frames).
2. Build anchor models for P1/P2 (median feature over all their frames),
   iterating so anchors are computed from *corrected* labels.
3. Per frame, score KEEP vs FLIP against the anchors; smooth the binary
   decision over time with a switching penalty (a real swap is persistent,
   per-frame noise is not) — simple 2-state Viterbi.
4. Rewrite the JSON in place (backup .pre_relabel.json) + report flips.

Usage:
    python3 scripts/relabel_ids_global.py CACHE_DIR --camera 360-Camera-2 \
        [--data-dir src/output/person_id_christina] [--switch-cost 1.5]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.person_identity import ClothingExtractor, _feat_dist  # noqa: E402


def collect_frames(cache_dir: Path, camera: str):
    pat = re.compile(rf'^eq_{re.escape(camera)}_(-?\d+)_')
    out = {}
    for f in cache_dir.iterdir():
        if f.is_file() and not f.name.startswith('._'):
            m = pat.match(f.name)
            if m:
                out[int(m.group(1))] = f
    return out


def median_feat(feats):
    parts = {}
    for part in ('torso', 'thigh'):
        vs = [f[part] for f in feats if f and f.get(part) is not None]
        parts[part] = np.median(np.stack(vs), axis=0) if vs else None
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache_dir')
    ap.add_argument('--camera', required=True)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--switch-cost', type=float, default=1.5,
                    help='Viterbi penalty for changing the keep/flip state')
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else root / 'src/output/person_id_christina'
    jpath = data_dir / f'people_id_{args.camera}.json'
    doc = json.loads(jpath.read_text())
    frames = collect_frames(Path(args.cache_dir), args.camera)

    ex = ClothingExtractor()
    canon_w = None

    # 1. features per assigned detection
    rows = []  # (entry_idx, t, {id: feat})
    for idx, e in enumerate(doc['timeline']):
        path = frames.get(e['t'])
        if path is None or not e['people']:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        if canon_w is None:
            canon_w = max(img.shape[1], 4000 if img.shape[1] <= 4000 else img.shape[1])
        if img.shape[1] != canon_w:
            img = cv2.resize(img, (canon_w, canon_w // 2))
        feats = {}
        for p in e['people']:
            f = ex.extract(img, p)
            if f is not None:
                feats[p['id']] = f
        if feats:
            rows.append((idx, e['t'], feats))
        if len(rows) % 200 == 0:
            print(f'  features {len(rows)} rows (t={e["t"]})', flush=True)

    if not rows:
        print('nothing to relabel'); return

    # 2-3. iterate: anchors from current labels -> Viterbi keep/flip
    flip = np.zeros(len(rows), dtype=bool)
    for it in range(3):
        lab_feats = {1: [], 2: []}
        for (idx, t, feats), fl in zip(rows, flip):
            for pid, f in feats.items():
                out_id = (3 - pid) if fl else pid
                lab_feats[out_id].append(f)
        anchors = {pid: median_feat(fs) for pid, fs in lab_feats.items()}

        # per-row cost of keep vs flip (lower = better)
        def cost(feats, flipped):
            tot, n = 0.0, 0
            for pid, f in feats.items():
                out_id = (3 - pid) if flipped else pid
                d = _feat_dist(f, anchors[out_id])
                if d is not None:
                    tot += d; n += 1
            return tot / n if n else 0.0

        keep_c = np.array([cost(f, False) for _, _, f in rows])
        flip_c = np.array([cost(f, True) for _, _, f in rows])

        # 2-state Viterbi with switching penalty scaled by typical cost
        scale = float(np.median(np.abs(keep_c - flip_c))) or 0.01
        sw = args.switch_cost * scale * 10
        n = len(rows)
        dp = np.zeros((n, 2)); bp = np.zeros((n, 2), dtype=int)
        dp[0] = [keep_c[0], flip_c[0]]
        for i in range(1, n):
            for s, c in enumerate((keep_c[i], flip_c[i])):
                stay = dp[i-1][s]
                move = dp[i-1][1-s] + sw
                bp[i][s] = s if stay <= move else 1 - s
                dp[i][s] = c + min(stay, move)
        s = int(np.argmin(dp[-1]))
        new_flip = np.zeros(n, dtype=bool)
        for i in range(n - 1, -1, -1):
            new_flip[i] = bool(s)
            s = bp[i][s]
        changed = int((new_flip != flip).sum())
        flip = new_flip
        print(f'iter {it}: flipped rows={int(flip.sum())}/{n} (changed {changed})')
        if changed == 0 and it > 0:
            break

    # boundary refinement: Viterbi finds the right number of swap segments
    # but can place a cut a frame or two off (creates a visible 1-frame
    # P1/P2 flicker at the boundary) — slide each cut to the local exact
    # minimum-cost position, judged by per-row costs alone
    idxs = np.flatnonzero(np.diff(flip.astype(int))) + 1
    for i0 in idxs:
        lo, hi = max(1, i0 - 8), min(len(rows) - 1, i0 + 8)
        left_state, right_state = flip[lo - 1], flip[hi]
        if left_state == right_state:
            continue
        best_c, best_cut = None, i0
        for cut in range(lo, hi + 1):
            c = sum((flip_c[j] if (left_state if j < cut else right_state) else keep_c[j])
                    for j in range(lo, hi + 1))
            if best_c is None or c < best_c:
                best_c, best_cut = c, cut
        if best_cut != i0:
            print(f'  boundary moved: row {i0} (t={rows[i0][1]}) -> '
                  f'row {best_cut} (t={rows[best_cut][1]})')
        flip[lo:hi + 1] = np.where(np.arange(lo, hi + 1) < best_cut,
                                   left_state, right_state)

    # 4. rewrite — every timeline entry takes the flip state of the nearest
    # feature row in time (entries whose feature extraction failed still
    # carry ids and must flip with their segment)
    row_ts = np.array([t for _, t, _ in rows], dtype=float)
    n_flip = int(flip.sum())
    if n_flip:
        backup = jpath.with_suffix('.pre_relabel.json')
        backup.write_text(json.dumps(doc, indent=1))
        segs, prev_fl, flipped_entries = [], False, 0
        for e in doc['timeline']:
            if not e['people']:
                continue
            i = int(np.clip(np.searchsorted(row_ts, e['t']), 0, len(rows) - 1))
            if i > 0 and abs(row_ts[i - 1] - e['t']) < abs(row_ts[i] - e['t']):
                i -= 1
            fl = bool(flip[i])
            if fl:
                for p in e['people']:
                    p['id'] = 3 - p['id']
                flipped_entries += 1
                if segs and prev_fl:
                    segs[-1][1] = e['t']
                else:
                    segs.append([e['t'], e['t']])
            prev_fl = fl
        doc['stats']['relabel_flipped_frames'] = flipped_entries
        doc['stats']['relabel_segments'] = [[int(a), int(b)] for a, b in segs]
        jpath.write_text(json.dumps(doc, indent=1))
        print(f'REWROTE {jpath.name}: {flipped_entries} entries flipped, segments: {segs[:10]}')
    else:
        print('no flips needed — labels globally consistent')

    sep = _feat_dist(anchors[1], anchors[2])
    print(f'anchor separation: {sep:.3f}')


if __name__ == '__main__':
    main()
