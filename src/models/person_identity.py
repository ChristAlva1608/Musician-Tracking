"""
Person identity by clothing color (shirt / pants) + temporal consistency.

Separates the k=2 persistent people in a closed-room 360 recording:
- torso color (between shoulders and hips)  -> shirt
- thigh color (hips to knees)               -> pants / shorts
Regions come from pose keypoints, not the whole bbox, so background and
skin contribute little. Colors are compared in a cone-mapped HSV space
(hue is circular, gray clothes have no meaningful hue).

People never leave the room, so identities are persistent: a frame where
one person is missing is a detection miss, not an exit. Reflections
(piano lid, screens) are rejected by the appearance gate + k=2 cap.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional

# COCO keypoint ids
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14


def _hsv_to_vec(h: float, s: float, v: float) -> np.ndarray:
    """Map OpenCV HSV (h:0-179, s,v:0-255) to a 3D cone where euclidean
    distance behaves: hue circular, hue irrelevant when saturation is low."""
    ang = h / 180.0 * 2 * np.pi
    sn, vn = s / 255.0, v / 255.0
    return np.array([np.cos(ang) * sn * vn, np.sin(ang) * sn * vn, vn])


def _region_color(frame_hsv: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    px = frame_hsv[mask > 0]
    if len(px) < 50:
        return None
    # circular median for hue via saturation-weighted vector mean
    ang = px[:, 0].astype(np.float32) / 180.0 * 2 * np.pi
    w = px[:, 1].astype(np.float32) + 1.0
    h = (np.arctan2((np.sin(ang) * w).sum(), (np.cos(ang) * w).sum())
         % (2 * np.pi)) / (2 * np.pi) * 180.0
    s = float(np.median(px[:, 1]))
    v = float(np.median(px[:, 2]))
    return _hsv_to_vec(h, s, v)


def _shrink(poly: np.ndarray, factor: float = 0.7) -> np.ndarray:
    c = poly.mean(axis=0)
    return (c + (poly - c) * factor).astype(np.int32)


class ClothingExtractor:
    """Extracts a 6D clothing feature (torso color ⊕ thigh color)."""

    def __init__(self, min_joint_conf: float = 0.3):
        self.min_joint_conf = min_joint_conf

    def _ok(self, kp, idx) -> bool:
        return kp[idx][2] >= self.min_joint_conf

    def extract(self, frame_bgr: np.ndarray, person: Dict[str, Any]) -> Optional[Dict]:
        kp = person.get('keypoints')
        if not kp:
            return None
        kp = np.asarray(kp, dtype=np.float32)
        h, w = frame_bgr.shape[:2]

        if person.get('crosses_seam'):
            # unwrap: shift so the person is contiguous, roll frame to match
            shift = w // 2
            frame_bgr = np.roll(frame_bgr, shift, axis=1)
            kp = kp.copy()
            kp[kp[:, 2] > 0, 0] = (kp[kp[:, 2] > 0, 0] + shift) % w

        hsv = None
        mask = np.zeros((h, w), np.uint8)
        torso = thigh = None

        if all(self._ok(kp, i) for i in (L_SH, R_SH, L_HIP, R_HIP)):
            poly = _shrink(kp[[L_SH, R_SH, R_HIP, L_HIP], :2])
            cv2.fillPoly(mask, [poly], 255)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            torso = _region_color(hsv, mask)

        mask[:] = 0
        thick = 0
        for hip, knee in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
            if self._ok(kp, hip) and self._ok(kp, knee):
                p1, p2 = kp[hip][:2], kp[knee][:2]
                thick = max(4, int(np.linalg.norm(p2 - p1) * 0.35))
                # sample the middle 60% of the thigh: skips waistband & knee
                a = p1 + (p2 - p1) * 0.2
                b = p1 + (p2 - p1) * 0.8
                cv2.line(mask, tuple(a.astype(int)), tuple(b.astype(int)), 255, thick)
        if thick:
            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            thigh = _region_color(hsv, mask)

        if torso is None and thigh is None:
            return None
        return {'torso': torso, 'thigh': thigh}


def _feat_dist(a: Dict, b: Dict) -> Optional[float]:
    """Distance between two clothing features; compares only shared parts."""
    dists, weights = [], []
    for part, wgt in (('torso', 1.0), ('thigh', 0.8)):
        if a.get(part) is not None and b.get(part) is not None:
            dists.append(float(np.linalg.norm(a[part] - b[part])))
            weights.append(wgt)
    if not dists:
        return None
    return float(np.average(dists, weights=weights))


class PersonIdentifier:
    """Assigns stable ids (1..k) to detections across frames."""

    def __init__(self, k: int = 2, frame_width: int = 4000,
                 gate: float = 0.45, ema: float = 0.08,
                 w_spatial: float = 0.15, max_speed_frac: float = 0.10,
                 dup_iou: float = 0.45):
        self.k = k
        self.w = frame_width
        self.gate = gate          # max clothing distance to accept a match
        self.ema = ema
        self.w_spatial = w_spatial
        # velocity gate: a real person can't cross more than this fraction of
        # the panorama width per second (reflections/TV people "teleport")
        self.max_speed = max_speed_frac * frame_width
        self.dup_iou = dup_iou    # two dets overlapping this much = same person
        self.identities: List[Dict] = []
        self.extractor = ClothingExtractor()

    def _spatial(self, cx_a, cx_b, cy_a, cy_b) -> float:
        dx = abs(cx_a - cx_b)
        dx = min(dx, self.w - dx) / (self.w / 4)   # seam-aware, normalized
        dy = abs(cy_a - cy_b) / (self.w / 8)
        return min(1.0, np.hypot(dx, dy))

    def _wrap_dx(self, xa, xb):
        dx = abs(xa - xb)
        return min(dx, self.w - dx)

    def _dedup(self, people):
        """Suppress duplicate detections of the same physical person that
        survived tile merging (partial box + full box). Keeps higher conf."""
        order = sorted(range(len(people)), key=lambda d: people[d]['conf'],
                       reverse=True)
        keep, dropped = [], set()
        for d in order:
            bd = np.asarray(people[d]['bbox'])
            dup = False
            for k in keep:
                bk = np.asarray(people[k]['bbox'])
                # IoU or near-total containment of the smaller box
                x1, y1 = np.maximum(bd[:2], bk[:2])
                x2, y2 = np.minimum(bd[2:], bk[2:])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_d = (bd[2]-bd[0])*(bd[3]-bd[1])
                area_k = (bk[2]-bk[0])*(bk[3]-bk[1])
                union = area_d + area_k - inter
                if union <= 0:
                    continue
                if inter/union >= self.dup_iou or inter >= 0.75*min(area_d, area_k):
                    dup = True
                    break
            if dup:
                dropped.add(d)
            else:
                keep.append(d)
        return dropped

    def assign(self, t: float, frame_bgr: np.ndarray,
               people: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Returns a person id (or None) per detection, and updates models."""
        dropped = self._dedup(people)
        feats = [None if d in dropped else self.extractor.extract(frame_bgr, p)
                 for d, p in enumerate(people)]
        centers = [(0.5 * (p['bbox'][0] + p['bbox'][2]) % self.w,
                    0.5 * (p['bbox'][1] + p['bbox'][3])) for p in people]

        pairs = []  # (cost, det_idx, ident_idx)
        for d, feat in enumerate(feats):
            if feat is None:
                continue
            for i, ident in enumerate(self.identities):
                c = _feat_dist(feat, ident['feat'])
                if c is None:
                    continue
                gap = t - ident['last_t']
                if 0 < gap <= 30:
                    # velocity gate: reject teleports (reflections, TV, swaps)
                    if self._wrap_dx(centers[d][0], ident['cx']) > self.max_speed * gap:
                        continue
                if gap < 90:  # position only helps over short gaps
                    c += self.w_spatial * self._spatial(
                        centers[d][0], ident['cx'], centers[d][1], ident['cy'])
                pairs.append((c, d, i))

        pairs.sort()
        ids: List[Optional[int]] = [None] * len(people)
        used_d, used_i = set(), set()
        for c, d, i in pairs:
            if d in used_d or i in used_i or c > self.gate:
                continue
            # never give the second identity to a box overlapping an
            # already-assigned one (one physical person, two ids)
            clash = False
            for d2 in used_d:
                b1, b2 = people[d]['bbox'], people[d2]['bbox']
                x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                if inter > 0.5 * min((b1[2]-b1[0])*(b1[3]-b1[1]),
                                     (b2[2]-b2[0])*(b2[3]-b2[1])):
                    clash = True
                    break
            if clash:
                continue
            ids[d] = self.identities[i]['id']
            used_d.add(d); used_i.add(i)
            self._update(self.identities[i], feats[d], centers[d], t)

        # new identities from confident unmatched detections (up to k);
        # size floor keeps TV/screen people from founding an identity
        h_img = frame_bgr.shape[0]
        order = sorted(range(len(people)),
                       key=lambda d: people[d]['conf'], reverse=True)
        for d in order:
            if (ids[d] is None and feats[d] is not None
                    and len(self.identities) < self.k
                    and people[d]['conf'] >= 0.5
                    and (people[d]['bbox'][3] - people[d]['bbox'][1]) >= 0.06 * h_img):
                ident = {'id': len(self.identities) + 1, 'feat': feats[d],
                         'cx': centers[d][0], 'cy': centers[d][1],
                         'last_t': t, 'n': 1}
                self.identities.append(ident)
                ids[d] = ident['id']
        return ids

    def _update(self, ident, feat, center, t):
        for part in ('torso', 'thigh'):
            if feat.get(part) is not None:
                if ident['feat'].get(part) is None:
                    ident['feat'][part] = feat[part]
                else:
                    ident['feat'][part] = ((1 - self.ema) * ident['feat'][part]
                                           + self.ema * feat[part])
        ident['cx'], ident['cy'] = center
        ident['last_t'] = t
        ident['n'] += 1

    def separation(self) -> Optional[float]:
        """Clothing distance between the identity models (higher = easier)."""
        if len(self.identities) < 2:
            return None
        return _feat_dist(self.identities[0]['feat'], self.identities[1]['feat'])
