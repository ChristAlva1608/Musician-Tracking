#!/usr/bin/env python3
"""
Run tiled pose detection + clothing-color person ID over a pose_cache
directory of equirectangular frames, and produce:
- people_id_timeline.json (per-frame ids, coverage stats, id separation)
- montage JPGs showing P1/P2 in stable colors across the session

Usage:
    python3 scripts/person_id_from_cache.py CACHE_DIR --camera 360-Camera-1 \
        [--step 5] [--out OUT_DIR] [--montage-every 40]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.pose.yolo_tiled import YOLOTiledPoseDetector  # noqa: E402
from src.models.person_identity import PersonIdentifier  # noqa: E402

ID_COLORS = {1: (0, 255, 0), 2: (0, 165, 255), None: (128, 128, 128)}
ID_NAMES = {1: 'P1', 2: 'P2', None: '?'}


def collect(cache_dir: Path, camera: str, step: int):
    pat = re.compile(rf'^eq_{re.escape(camera)}_(-?\d+)_')
    frames = []
    for f in cache_dir.iterdir():
        if f.name.startswith('._'):
            continue
        m = pat.match(f.name)
        if m:
            frames.append((int(m.group(1)), f))
    frames.sort()
    return frames[::step]


def crop_person(frame, person, size=380):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person['bbox']
    cx, cy = int((x1 + x2) / 2) % w, int((y1 + y2) / 2)
    half = max(int(max(x2 - x1, y2 - y1) * 0.7), 160)
    if person.get('crosses_seam'):
        frame = np.roll(frame, w // 2, axis=1)
        cx = (cx + w // 2) % w
    a, b = max(0, cy - half), min(h, cy + half)
    c, d = max(0, cx - half), min(w, cx + half)
    crop = frame[a:b, c:d]
    return cv2.resize(crop, (size, size)) if crop.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache_dir')
    ap.add_argument('--camera', default='360-Camera-1')
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--montage-every', type=int, default=40,
                    help='put every Nth processed frame into the montage')
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    frames = collect(cache, args.camera, args.step)
    if not frames:
        print('No frames matched'); sys.exit(1)

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else root / 'src/output/person_id'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(frames)} frames ({args.camera}, step={args.step}) -> {out_dir}")

    detector = YOLOTiledPoseDetector(max_people=4)  # keep extras; gate rejects them
    first = cv2.imread(str(frames[0][1]))
    identifier = PersonIdentifier(k=2, frame_width=first.shape[1])

    timeline, tiles = [], []
    n_both = n_one = n_zero = 0
    t_start = time.time()

    for n, (t, path) in enumerate(frames):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        people = detector.detect(frame) or []
        ids = identifier.assign(float(t), frame, people)

        entry = {'t': t, 'people': []}
        got = set()
        for person, pid in zip(people, ids):
            if pid is None:
                continue
            got.add(pid)
            entry['people'].append({
                'id': pid, 'conf': round(person['conf'], 3),
                'bbox': [round(v, 1) for v in person['bbox']],
                'crosses_seam': person['crosses_seam'],
                'keypoints': [[round(x, 1), round(y, 1), round(c, 3)]
                              for x, y, c in person['keypoints']],
            })
        timeline.append(entry)
        n_both += len(got) == 2; n_one += len(got) == 1; n_zero += len(got) == 0

        if n % args.montage_every == 0:
            for person, pid in zip(people, ids):
                if pid is None:
                    continue
                crop = crop_person(frame.copy(), person)
                if crop is None:
                    continue
                color = ID_COLORS[pid]
                cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), color, 6)
                cv2.rectangle(crop, (0, 0), (170, 44), (0, 0, 0), -1)
                cv2.putText(crop, f"{ID_NAMES[pid]} t={t}s", (8, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                tiles.append((pid, t, crop))

        if n % 50 == 0:
            print(f"  {n}/{len(frames)} t={t}s both={n_both} one={n_one} zero={n_zero}")

    sep = identifier.separation()
    stats = {
        'camera': args.camera,
        'frames_processed': len(timeline),
        'both_people': n_both, 'one_person': n_one, 'no_person': n_zero,
        'both_pct': round(100 * n_both / max(1, len(timeline)), 1),
        'id_separation': round(sep, 3) if sep else None,
        'gate': identifier.gate,
        'sec_per_frame': round((time.time() - t_start) / max(1, len(timeline)), 2),
        'model': detector.model_path,
    }
    (out_dir / f'people_id_{args.camera}.json').write_text(
        json.dumps({'stats': stats, 'timeline': timeline}, indent=1))

    # montage: one row per identity, time flowing left->right
    for pid in (1, 2):
        row = [c for p, _, c in tiles if p == pid]
        if row:
            cv2.imwrite(str(out_dir / f'montage_{args.camera}_P{pid}.jpg'),
                        cv2.hconcat(row[:12]), [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
