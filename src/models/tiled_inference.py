"""
Tiled (sliced) YOLO inference engine — hybrid full-frame + overlapping tiles.

Why hybrid: the full-frame pass guarantees results are never worse than
non-tiled inference; tiles only ADD detections of small/distant subjects.
Overlapping tiles (default 25%) guarantee any person shorter than the
overlap band appears un-clipped in at least one tile.

Merging: greedy NMS by adjusted confidence. Detections clipped by an
interior tile edge are penalized so the un-clipped copy of the same person
wins. For matched pairs, keypoints are fused per-joint (higher confidence
joint wins) so partial skeletons from two tiles combine into a complete one.

360/equirectangular (wrap_360=True): the left/right frame edges are the
same physical direction (the stitch seam). A person standing on the seam is
split into two half-bodies at x≈0 and x≈W and lost by normal inference.
We add wrapped seam tiles built from hstack(right strip, left strip) so the
person appears whole; their coordinates may exceed W and IoU matching is
done modulo frame width. Output keypoints are folded back into [0, W).
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class TiledYOLO:
    """Generic tiled inference wrapper around an ultralytics YOLO model.

    Works for pose models (returns keypoints) and plain detect models
    (keypoints empty). One instance per model.
    """

    EDGE_TOL = 4          # px: bbox within this of an interior tile edge counts as clipped
    CLIP_PENALTY = 0.8    # confidence multiplier for tile-edge-clipped detections

    def __init__(self,
                 model_path: str,
                 confidence: float = 0.4,
                 tile_target: int = 1200,
                 overlap: float = 0.25,
                 iou_merge: float = 0.55,
                 imgsz: int = 640,
                 device: Optional[str] = None,
                 include_full_frame: bool = True,
                 wrap_360: bool = False):
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.confidence = confidence
        self.tile_target = tile_target
        self.overlap = overlap
        self.iou_merge = iou_merge
        self.imgsz = imgsz
        self.device = device or pick_device()
        self.include_full_frame = include_full_frame
        self.wrap_360 = wrap_360

    # ---------------- tiling ----------------

    def tile_layout(self, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Compute overlapping tile boxes (x1, y1, x2, y2) covering the frame."""
        def axis_tiles(size: int) -> List[Tuple[int, int]]:
            n = max(1, round(size / self.tile_target))
            if n == 1:
                return [(0, size)]
            tile = int(size / (n - (n - 1) * self.overlap))
            step = int(tile * (1 - self.overlap))
            spans = []
            for i in range(n):
                start = size - tile if i == n - 1 else i * step
                spans.append((start, start + tile))
            return spans

        return [(x1, y1, x2, y2)
                for (y1, y2) in axis_tiles(h)
                for (x1, x2) in axis_tiles(w)]

    # ---------------- inference ----------------

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run hybrid full-frame + tiled inference on a BGR frame.

        Returns merged detections sorted by bbox area (largest first):
        [{'bbox': [x1,y1,x2,y2], 'conf': float,
          'keypoints': [[x,y,conf], ...17] or [],
          'source': 'full' | 'tile'}]
        """
        h, w = frame.shape[:2]
        tiles = self.tile_layout(w, h)
        crops, origins = [], []
        if self.include_full_frame:
            crops.append(frame)
            origins.append((0, 0, w, h, 'full'))
        for (x1, y1, x2, y2) in tiles:
            crops.append(frame[y1:y2, x1:x2])
            origins.append((x1, y1, x2, y2, 'tile'))

        if self.wrap_360:
            # seam tiles: right strip + left strip glued, so a person split
            # across the equirectangular seam appears whole in one crop
            tile_w = tiles[0][2] - tiles[0][0]
            half = max(tile_w // 2, min(w // 4, self.tile_target // 2))
            y_spans = sorted({(y1, y2) for (_, y1, _, y2) in tiles})
            for (y1, y2) in y_spans:
                seam = np.hstack([frame[y1:y2, w - half:], frame[y1:y2, :half]])
                crops.append(seam)
                # x-origin = w - half; global x may exceed w, folded later
                origins.append((w - half, y1, w + half, y2, 'seam'))

        results = self.model(crops, conf=self.confidence, imgsz=self.imgsz,
                             device=self.device, verbose=False)

        detections = []
        for res, (ox1, oy1, ox2, oy2, src) in zip(results, origins):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            kpts = None
            if res.keypoints is not None and res.keypoints.xy is not None:
                kxy = res.keypoints.xy.cpu().numpy()
                kconf = (res.keypoints.conf.cpu().numpy()
                         if res.keypoints.conf is not None
                         else np.ones(kxy.shape[:2], dtype=np.float32))
                if kxy.ndim == 3 and kxy.shape[0] == len(res.boxes):
                    kpts = np.concatenate([kxy, kconf[..., None]], axis=2)
            for i in range(len(boxes)):
                bx = boxes[i].copy()
                clipped = False
                if src in ('tile', 'seam'):
                    tw, th = ox2 - ox1, oy2 - oy1
                    # seam-crop x edges are always interior cuts of the panorama
                    clipped = (
                        (bx[0] < self.EDGE_TOL and (ox1 > 0 or src == 'seam')) or
                        (bx[1] < self.EDGE_TOL and oy1 > 0) or
                        (bx[2] > tw - self.EDGE_TOL and (ox2 < w or src == 'seam')) or
                        (bx[3] > th - self.EDGE_TOL and oy2 < h)
                    )
                    bx[0] += ox1; bx[2] += ox1
                    bx[1] += oy1; bx[3] += oy1
                kp = None
                if kpts is not None:
                    kp = kpts[i].copy()
                    if src in ('tile', 'seam'):
                        valid = kp[:, 2] > 0
                        kp[valid, 0] += ox1
                        kp[valid, 1] += oy1
                detections.append({
                    'bbox': bx,
                    'conf': float(confs[i]),
                    'adj_conf': float(confs[i]) * (self.CLIP_PENALTY if clipped else 1.0),
                    'keypoints': kp,
                    'source': src,
                })

        merged = self._merge(detections, w if self.wrap_360 else None)
        merged.sort(key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]),
                    reverse=True)
        for d in merged:
            bx, kp = d['bbox'], d['keypoints']
            crosses_seam = False
            if self.wrap_360:
                if bx[0] >= w:                      # fully past the seam: plain left-edge det
                    bx[0] -= w; bx[2] -= w
                    if kp is not None:
                        kp[kp[:, 2] > 0, 0] -= w
                elif bx[2] > w:                     # genuinely straddles the seam
                    crosses_seam = True
                    if kp is not None:
                        valid = kp[:, 2] > 0
                        kp[valid, 0] = np.mod(kp[valid, 0], w)
            d['crosses_seam'] = crosses_seam
            # for seam-crossers bbox x2 stays > w (unwrapped) so width is meaningful
            d['bbox'] = [float(v) for v in bx]
            d['keypoints'] = ([[float(x), float(y), float(c)] for x, y, c in kp]
                              if kp is not None else [])
            d.pop('adj_conf', None)
        return merged

    # ---------------- merging ----------------

    def _merge(self, detections: List[Dict], wrap_w: Optional[int] = None) -> List[Dict]:
        shifts = (0.0,) if wrap_w is None else (0.0, -float(wrap_w), float(wrap_w))
        detections = sorted(detections, key=lambda d: d['adj_conf'], reverse=True)
        kept: List[Dict] = []
        for det in detections:
            match, match_shift = None, 0.0
            box = np.asarray(det['bbox'])
            for k in kept:
                kbox = np.asarray(k['bbox'])
                for s in shifts:  # compare modulo panorama width
                    if _iou(box + np.array([s, 0, s, 0]), kbox) >= self.iou_merge:
                        match, match_shift = k, s
                        break
                if match is not None:
                    break
            if match is None:
                kept.append(det)
            elif match['keypoints'] is not None and det['keypoints'] is not None:
                # per-joint fusion: fill in / upgrade joints the kept copy saw poorly
                mk = match['keypoints']
                dk = det['keypoints'].copy()
                dk[dk[:, 2] > 0, 0] += match_shift  # align to kept copy's x convention
                better = dk[:, 2] > mk[:, 2]
                mk[better] = dk[better]
        return kept
