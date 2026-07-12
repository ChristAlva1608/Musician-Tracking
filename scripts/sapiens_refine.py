#!/usr/bin/env python3
"""
Sapiens2 refinement pass ("kinh lup"): re-run pose on selected time windows
with Meta's Sapiens2 (308 keypoints: body + 274 face + hands + feet),
using OUR person boxes from people_id_*.json — no DETR needed.

Run with the sapiens2 venv python:
    ~/Downloads/sapiens2/.venv/bin/python scripts/sapiens_refine.py \
        src/output/person_id_christina/people_id_360-Camera-1.json \
        "/Volumes/.../pose_cache" \
        --window 1300 1500 --model 5b --device mps

Requires the sapiens package importable (pip install -e ~/Downloads/sapiens2
or PYTHONPATH=~/Downloads/sapiens2) and checkpoints in ~/sapiens2_host/pose/.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SAPIENS_ROOT = Path(os.environ.get('SAPIENS_ROOT',
                                   Path.home() / 'Downloads/sapiens2'))
CKPT_ROOT = Path(os.environ.get('SAPIENS_CHECKPOINT_ROOT',
                                Path.home() / 'sapiens2_host'))
sys.path.insert(0, str(SAPIENS_ROOT))

import torch  # noqa: E402


def load_model(size: str, device: str):
    from sapiens.pose.models import init_model
    from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
    cfg = (SAPIENS_ROOT / 'sapiens/pose/configs/keypoints308/'
           f'shutterstock_goliath_3po/sapiens2_{size}_keypoints308_'
           'shutterstock_goliath_3po-1024x768.py')
    ckpt = CKPT_ROOT / f'pose/sapiens2_{size}_pose.safetensors'
    if not ckpt.exists():
        sys.exit(f"Checkpoint missing: {ckpt}")
    cwd = os.getcwd()
    os.chdir(SAPIENS_ROOT / 'sapiens/pose')  # configs use relative from_file
    try:
        model = init_model(str(cfg), str(ckpt), device=device)
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file='configs/_base_/keypoints308.py'))
        codec_cfg = dict(model.cfg.codec)
        codec_cfg.pop('type', None)
        model.codec = UDPHeatmap(**codec_cfg)
    finally:
        os.chdir(cwd)
    return model


def pose_on_boxes(model, image_bgr: np.ndarray, boxes: np.ndarray):
    """Top-down 308-kp pose for given person boxes (x1,y1,x2,y2)."""
    inputs_list, samples = [], []
    for bbox in boxes:
        data_info = dict(img=image_bgr, bbox=np.asarray(bbox, np.float32)[None],
                         bbox_score=np.ones(1, np.float32))
        data = model.pipeline(data_info)
        data = model.data_preprocessor(data)
        inputs_list.append(data['inputs'])
        samples.append(data['data_samples'])
    inputs = torch.cat(inputs_list, dim=0)
    with torch.no_grad():
        pred = model(inputs).cpu().numpy()
    out = []
    for i, ds in enumerate(samples):
        kps, scores = model.codec.decode(pred[i])
        size = ds['meta']['input_size']
        center = ds['meta']['bbox_center']
        scale = ds['meta']['bbox_scale']
        kps = kps / size * scale + center - 0.5 * scale
        out.append((kps[0], scores[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('timeline_json')
    ap.add_argument('cache_dir')
    ap.add_argument('--window', nargs=2, type=float, default=None,
                    metavar=('T0', 'T1'), help='seconds; default = whole file')
    ap.add_argument('--model', default='5b',
                    choices=['0.4b', '0.8b', '1b', '5b'])
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default=None)
    ap.add_argument('--vis-every', type=int, default=20)
    args = ap.parse_args()

    data = json.loads(Path(args.timeline_json).read_text())
    camera = data['stats']['camera']
    entries = [e for e in data['timeline'] if e['people']]
    if args.window:
        t0, t1 = args.window
        entries = [e for e in entries if t0 <= e['t'] <= t1]
    if not entries:
        sys.exit('No frames in window')

    root = Path(__file__).resolve().parents[1]
    out_dir = (Path(args.out) if args.out
               else root / 'src/output/sapiens_refine')
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache_dir)
    frame_of = {}
    pat = re.compile(rf'^eq_{re.escape(camera)}_(-?\d+)_')
    for f in cache.iterdir():
        m = pat.match(f.name)
        if m and not f.name.startswith('._'):
            frame_of[int(m.group(1))] = f

    print(f"Sapiens2-{args.model} on {len(entries)} frames "
          f"({camera}, device={args.device})")
    model = load_model(args.model, args.device)

    results = []
    t_start = time.time()
    for n, e in enumerate(entries):
        path = frame_of.get(e['t'])
        if path is None:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        w = frame.shape[1]
        boxes, metas = [], []
        for p in e['people']:
            x1, y1, x2, y2 = p['bbox']
            shift = 0
            if p.get('crosses_seam'):
                shift = w // 2
                x1, x2 = (x1 + shift) % w, (x2 + shift) % w
            boxes.append([x1, y1, x2, y2])
            metas.append((p['id'], shift))
        img = np.roll(frame, metas[0][1], axis=1) if metas[0][1] else frame

        preds = pose_on_boxes(model, img, np.asarray(boxes))
        rec = {'t': e['t'], 'people': []}
        for (pid, shift), (kps, scores) in zip(metas, preds):
            kps = kps.copy()
            if shift:
                kps[:, 0] = (kps[:, 0] - shift) % w
            rec['people'].append({
                'id': pid,
                'keypoints308': np.round(kps, 1).tolist(),
                'scores': np.round(scores, 3).tolist(),
            })
        results.append(rec)

        if n % args.vis_every == 0:
            vis = img.copy()
            for (pid, _), (kps, scores) in zip(metas, preds):
                col = (0, 255, 0) if pid == 1 else (0, 165, 255)
                for (x, y), s in zip(kps, scores):
                    if s > 0.3:
                        cv2.circle(vis, (int(x), int(y)), 2, col, -1)
            cv2.imwrite(str(out_dir / f'vis_{camera}_{e["t"]}.jpg'), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        if n % 20 == 0:
            rate = (time.time() - t_start) / (n + 1)
            print(f"  {n}/{len(entries)} ({rate:.1f}s/frame)")

    out_json = out_dir / f'sapiens_{args.model}_{camera}.json'
    out_json.write_text(json.dumps({
        'model': f'sapiens2-{args.model}', 'camera': camera,
        'window': args.window, 'frames': len(results),
        'timeline': results}, indent=1))
    print(f"Saved {len(results)} frames -> {out_json}")


if __name__ == '__main__':
    main()
