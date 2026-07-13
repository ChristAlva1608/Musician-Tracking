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

import itertools

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

    @staticmethod
    def _ring_v(frame_bgr: np.ndarray, bbox, w: int) -> Optional[float]:
        """Median brightness (0..1) of a band AROUND the bbox, wrap-aware.
        A reflection sits embedded in dark glass (piano lid, TV) so its
        surroundings are near-black; a real person — even in black clothes —
        stands against floor/wall. Measured: reflection ring 0.20, real
        people 0.55-0.77."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            return None
        mx, my = int(bw * 0.15) + 1, int(bh * 0.15) + 1
        h = frame_bgr.shape[0]
        oy1, oy2 = max(0, y1 - my), min(h, y2 + my)
        xs = np.arange(x1 - mx, x2 + mx) % w
        patch = frame_bgr[oy1:oy2][:, xs]
        inner = np.zeros(patch.shape[:2], bool)
        ix1, iy1 = x1 - (x1 - mx), y1 - oy1
        inner[max(0, iy1):iy1 + bh, ix1:ix1 + bw] = True
        ring = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[~inner]
        if len(ring) < 50:
            return None
        return float(np.median(ring[:, 2])) / 255.0

    def extract(self, frame_bgr: np.ndarray, person: Dict[str, Any]) -> Optional[Dict]:
        kp = person.get('keypoints')
        if not kp:
            return None
        kp = np.asarray(kp, dtype=np.float32)
        h, w = frame_bgr.shape[:2]
        ring = self._ring_v(frame_bgr, person['bbox'], w)

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
        return {'torso': torso, 'thigh': thigh, 'ring': ring}


def _brightness(feat: Dict) -> Optional[float]:
    """Clothing brightness (v of the HSV cone, 0..1), torso preferred."""
    for part in ('torso', 'thigh'):
        if feat.get(part) is not None:
            return float(feat[part][2])
    return None


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

    def _same_person(self, pa, pb) -> bool:
        """Same physical person = their SKELETONS coincide. Box overlap alone
        is NOT enough — an occluded person's box can sit almost entirely
        inside the other's (teacher behind student) while the joints differ."""
        ka, kb = pa.get('keypoints'), pb.get('keypoints')
        if ka and kb:
            ka, kb = np.asarray(ka), np.asarray(kb)
            shared = (ka[:, 2] > 0.3) & (kb[:, 2] > 0.3)
            if shared.sum() >= 4:
                dx = np.abs(ka[shared, 0] - kb[shared, 0])
                dx = np.minimum(dx, self.w - dx)      # seam wrap
                dy = ka[shared, 1] - kb[shared, 1]
                d = np.hypot(dx, dy)
                ref = max(pa['bbox'][3] - pa['bbox'][1],
                          pb['bbox'][3] - pb['bbox'][1], 1.0)
                return float(np.median(d)) < 0.10 * ref
        # no comparable skeletons: only near-identical boxes count
        bd, bk = np.asarray(pa['bbox']), np.asarray(pb['bbox'])
        x1, y1 = np.maximum(bd[:2], bk[:2])
        x2, y2 = np.minimum(bd[2:], bk[2:])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = ((bd[2] - bd[0]) * (bd[3] - bd[1])
                 + (bk[2] - bk[0]) * (bk[3] - bk[1]) - inter)
        return union > 0 and inter / union >= 0.85

    def _dedup(self, people):
        """Suppress duplicate detections of the same physical person that
        survived tile merging. Keeps the most COMPLETE skeleton (a truncated
        partial box often has higher conf than the full-body one — keeping it
        would lose the hips/thighs the clothing feature needs)."""
        def richness(d):
            p = people[d]
            return (sum(1 for k in p['keypoints'] if k[2] > 0.3), p['conf'])
        order = sorted(range(len(people)), key=richness, reverse=True)
        keep, dropped = [], set()
        for d in order:
            if any(self._same_person(people[d], people[k]) for k in keep):
                dropped.add(d)
            else:
                keep.append(d)
        return dropped

    def assign(self, t: float, frame_bgr: np.ndarray,
               people: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Returns a person id (or None) per detection, and updates models."""
        feats = [self.extractor.extract(frame_bgr, p) for p in people]
        # reflection filter: dark torso alone is NOT enough (a real person in
        # a black top measures the same V=13 as a reflection) — but only a
        # reflection is EMBEDDED in the dark glass that makes it, so its
        # bbox surroundings are near-black too
        for d, f in enumerate(feats):
            if (f is not None and f.get('ring') is not None
                    and (_brightness(f) or 1.0) < 0.25 and f['ring'] < 0.35):
                feats[d] = None
        centers = [(0.5 * (p['bbox'][0] + p['bbox'][2]) % self.w,
                    0.5 * (p['bbox'][1] + p['bbox'][3])) for p in people]
        heights = [max(1.0, p['bbox'][3] - p['bbox'][1]) for p in people]

        pairs = []  # (cost, det_idx, ident_idx)
        for d, feat in enumerate(feats):
            if feat is None:
                continue
            for i, ident in enumerate(self.identities):
                c = _feat_dist(feat, ident['feat'])
                if c is None:
                    continue
                gap = t - ident['last_t']
                if gap < 90:  # position only helps over short gaps
                    c += self.w_spatial * self._spatial(
                        centers[d][0], ident['cx'], centers[d][1], ident['cy'])
                if gap < 30 and ident.get('h'):
                    # apparent height can't jump 2x+ in seconds — breaks the
                    # tie when a seam-straddler's wrapped center lands next
                    # to the other person and clothing colors are similar
                    c += 0.12 * min(2.5, abs(np.log2(heights[d] / ident['h'])))
                pairs.append((c, d, i))

        # jointly optimal assignment over all identities (k is tiny, so
        # brute force). Greedy per-pair matching swapped P1/P2 for single
        # frames when the two people's clothing is similar — the SUM of
        # costs is far more stable than the single lowest pair.
        cand: List[Dict[int, float]] = [{} for _ in self.identities]
        for c, d, i in pairs:
            if c <= self.gate and (d not in cand[i] or c < cand[i][d]):
                cand[i][d] = c
        ids: List[Optional[int]] = [None] * len(people)
        best, best_cost = None, None
        options = [list(ci.items()) + [(None, self.gate)] for ci in cand]
        for combo in itertools.product(*options):
            ds = [d for d, _ in combo if d is not None]
            if len(set(ds)) != len(ds):
                continue
            # never give two identities to the SAME skeleton — but heavy box
            # overlap alone is fine (occluded person behind the other)
            if any(self._same_person(people[a], people[b])
                   for x, a in enumerate(ds) for b in ds[x + 1:]):
                continue
            tot = sum(c for _, c in combo)
            if best_cost is None or tot < best_cost:
                best, best_cost = combo, tot
        if best:
            for i, (d, _) in enumerate(best):
                if d is not None:
                    ids[d] = self.identities[i]['id']
                    self._update(self.identities[i], feats[d], centers[d],
                                 heights[d], t)

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
                         'h': heights[d], 'last_t': t, 'n': 1}
                self.identities.append(ident)
                ids[d] = ident['id']
        return ids

    def _update(self, ident, feat, center, height, t):
        for part in ('torso', 'thigh'):
            if feat.get(part) is not None:
                if ident['feat'].get(part) is None:
                    ident['feat'][part] = feat[part]
                else:
                    ident['feat'][part] = ((1 - self.ema) * ident['feat'][part]
                                           + self.ema * feat[part])
        ident['cx'], ident['cy'] = center
        ident['h'] = height
        ident['last_t'] = t
        ident['n'] += 1

    def separation(self) -> Optional[float]:
        """Clothing distance between the identity models (higher = easier)."""
        if len(self.identities) < 2:
            return None
        return _feat_dist(self.identities[0]['feat'], self.identities[1]['feat'])
