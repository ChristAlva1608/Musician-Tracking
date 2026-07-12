#!/usr/bin/env python3
"""
Compare pose detection: old pipeline (yolo11n, full frame, first person only)
vs new pipeline (largest model, hybrid tiles + 360 seam handling, all people).

Usage:
    python3 scripts/compare_tiled_vs_baseline.py IMAGE_OR_DIR [more...] \
        [--out OUT_DIR] [--limit N]

Writes side-by-side annotated JPGs and a summary JSON to OUT_DIR
(default: src/output/tiled_comparison/).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.pose.yolo_tiled import YOLOTiledPoseDetector  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from src.models.tiled_inference import pick_device  # noqa: E402


def collect_images(paths, limit):
    exts = {'.jpg', '.jpeg', '.png'}
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir()
                                if f.suffix.lower() in exts
                                and not f.name.startswith('._')))
        elif p.suffix.lower() in exts:
            files.append(p)
    if limit and len(files) > limit:
        # spread picks across the folder instead of taking the first N
        idx = np.linspace(0, len(files) - 1, limit).astype(int)
        files = [files[i] for i in sorted(set(idx))]
    return files


def draw_baseline(frame, results, color=(0, 0, 255)):
    """Draw ALL people the baseline found (old code only kept the first)."""
    n = 0
    if results and results[0].keypoints is not None:
        kpts = results[0].keypoints.xy.cpu().numpy()
        n = len(kpts)
        for person in kpts:
            for x, y in person:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    return n


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(img, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=6)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else root / 'src/output/tiled_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.inputs, args.limit)
    if not images:
        print("No images found"); sys.exit(1)
    print(f"Comparing on {len(images)} frames -> {out_dir}")

    device = pick_device()
    baseline = YOLO('yolo11n-pose.pt')            # old pipeline's model
    tiled = YOLOTiledPoseDetector(max_people=0)   # new: largest + tiles + seam

    summary = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        base_res = baseline(frame, conf=0.5, device=device, verbose=False)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        people = tiled.detect(frame) or []
        t_tiled = time.perf_counter() - t0

        vis_base = frame.copy()
        n_base = draw_baseline(vis_base, base_res)
        vis_tiled = tiled.draw_landmarks(frame.copy(), people)

        joint_confs = [c for p in people for (_, _, c) in p['keypoints'] if c > 0]
        row = {
            'image': img_path.name,
            'size': [w, h],
            'baseline_people': n_base,
            'baseline_sec': round(t_base, 3),
            'tiled_people': len(people),
            'tiled_sec': round(t_tiled, 3),
            'tiled_mean_joint_conf': round(float(np.mean(joint_confs)), 3) if joint_confs else 0,
            'seam_crossers': sum(1 for p in people if p.get('crosses_seam')),
        }
        summary.append(row)
        print(f"  {img_path.name}: baseline {n_base} people ({t_base:.2f}s) | "
              f"tiled {len(people)} people ({t_tiled:.2f}s)"
              f"{' [SEAM]' if row['seam_crossers'] else ''}")

        label(vis_base, f"OLD: yolo11n full-frame  ({n_base} people, {t_base:.2f}s)")
        label(vis_tiled, f"NEW: {Path(tiled.model_path).stem} tiled+seam  "
                         f"({len(people)} people, {t_tiled:.2f}s)")
        combo = np.vstack([vis_base, vis_tiled])
        if combo.shape[1] > 2400:
            s = 2400 / combo.shape[1]
            combo = cv2.resize(combo, None, fx=s, fy=s)
        cv2.imwrite(str(out_dir / f"cmp_{img_path.stem}.jpg"), combo,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    (out_dir / 'summary.json').write_text(json.dumps({
        'device': device,
        'tiled_model': tiled.model_path,
        'frames': summary,
    }, indent=2))
    print(f"\nSaved {len(summary)} comparisons + summary.json in {out_dir}")


if __name__ == '__main__':
    main()
