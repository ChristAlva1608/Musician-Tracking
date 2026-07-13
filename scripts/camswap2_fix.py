#!/usr/bin/env python3
"""camswap2: phát hiện & sửa hoán đổi ID người theo màu áo (Christina pilot).

Bối cảnh: P1 = cô áo sơ mi hoa sẫm (đeo kính), P2 = cô áo tank đen + quần nâu đỏ.
Cam-1 từng bị ngược quy ước gần cả buổi (P1=tank) + vài đoạn swap nội bộ;
Cam-2/Cam-3 có đoạn swap ngắn. Đã sửa 2026-07-13, bằng chứng ảnh trong
src/output/person_id_christina/camswap2_evidence/ (mọi biên đoạn đều được
xác nhận bằng mắt trên montage trước khi flip).

Classifier per-frame (chỉ dùng để GỢI Ý vùng nghi vấn — kết luận cuối cùng
luôn bằng mắt, vì che khuất/mờ chuyển động gây nhiễu):
- Crop thân trong từ keypoints vai(5,6)-hông(11,12), co ~27% ngang,
  22%/12% dọc; cột ảnh lấy theo x % 4000 để xử lý bbox vắt mép seam.
- Thêm 2 dải nhỏ dọc đoạn vai->khuỷu (tank = da trần, hoa = tay áo).
- Pixel HSV phân lớp: da / vải đen / hoạ tiết màu + entropy hue.
- tank_score = 1.3*skin + 0.7*black - 1.6*print - 0.25*hue_ent.
- So sánh TƯƠNG ĐỐI giữa 2 người trong frame (diff = s1 - s2) mới đáng tin;
  điểm tuyệt đối chỉ đủ tốt để flag frame 1 người có mâu thuẫn rõ.

Các đoạn đã sửa (giây, theo t trong timeline):
- Cam-1: flip nội bộ [1092], [1097-1101], [1311-1315], [1321], [1343], [2712]
         (lưu ý 1089/1095 KHÔNG swap — 1089 là detection chimera, 1095 đúng),
         sau đó FLIP TOÀN CAMERA 1<->2 để P1=áo hoa khớp Cam-2/3.
- Cam-2: flip [2305-2306].
- Cam-3: flip [1092-1119] (sự kiện chỉnh camera; 1097 là chimera, flip theo
         tính liên tục của đoạn).
Các cụm nghi khác của Cam-1 (653-867, 1055-1068, 1206-1309, 1368-1410,
1507-1522, 2001-2060, 2120-2132 + các frame lẻ) đã kiểm tra ảnh: KHÔNG swap,
diff âm do che khuất / mờ chuyển động / da tay trần của cô áo hoa.

Usage (chạy lại từ backup nếu cần):
    python3 scripts/camswap2_fix.py src/output/person_id_christina \
        "/Volumes/MAML 8TB 1/Phan Dissertation Data/Christina - MultiCam Data - Piano - 2025-09-19"
Script tự backup people_id_*.json -> *.json.pre_camswap2.json (không ghi đè
backup cũ) rồi áp các đoạn flip ở trên và ghi stats['camswap2'].
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

W = 4000

# Các đoạn đã xác minh bằng mắt (xem docstring)
FIXES = {
    "1": {"segments_fixed": [[1092, 1092], [1097, 1101], [1311, 1315],
                              [1321, 1321], [1343, 1343], [2712, 2712]],
          "whole_flip": True},
    "2": {"segments_fixed": [[2305, 2306]], "whole_flip": False},
    "3": {"segments_fixed": [[1092, 1119]], "whole_flip": False},
}


# ---------- classifier (để quét lại / dùng cho buổi khác) ----------

def torso_crop(img, person):
    kp = np.array(person["keypoints"])
    pts = kp[[5, 6, 11, 12]]
    good = pts[:, 2] > 0.35
    if good.sum() < 3 or (pts[0, 2] < 0.35 and pts[1, 2] < 0.35) \
            or (pts[2, 2] < 0.35 and pts[3, 2] < 0.35):
        return None
    xs, ys = pts[good, 0], pts[good, 1]
    x1, x2 = xs.min(), xs.max()
    sh = [p[1] for p, g in zip(pts[:2], good[:2]) if g]
    hp = [p[1] for p, g in zip(pts[2:], good[2:]) if g]
    y1 = min(np.mean(sh) if sh else ys.min(), np.mean(hp) if hp else ys.max())
    y2 = max(np.mean(sh) if sh else ys.min(), np.mean(hp) if hp else ys.max())
    w, h = x2 - x1, y2 - y1
    if w < 20 or h < 20:
        return None
    cy1 = int(max(0, min(img.shape[0] - 2, y1 + 0.22 * h)))
    cy2 = int(max(cy1 + 5, min(img.shape[0] - 1, y2 - 0.12 * h)))
    cols = (np.arange(int(x1 + 0.27 * w), int(x2 - 0.27 * w)) % W).astype(int)
    crop = img[cy1:cy2][:, cols]
    return crop if crop.size >= 500 else None


def arm_strips(img, person):
    kp = np.array(person["keypoints"])
    H = img.shape[0]
    out = []
    for sh, el in ((5, 7), (6, 8)):
        if kp[sh, 2] < 0.4 or kp[el, 2] < 0.4:
            continue
        seg = kp[el, :2] - kp[sh, :2]
        L = float(np.hypot(*seg))
        if L < 25:
            continue
        r = max(6, int(0.14 * L))
        for frac in (0.35, 0.65):
            cx, cy = kp[sh, :2] + frac * seg
            y1, y2 = int(max(0, cy - r)), int(min(H - 1, cy + r))
            if y2 - y1 < 4:
                continue
            cols = (np.arange(int(cx - r), int(cx + r)) % W).astype(int)
            out.append(img[y1:y2][:, cols])
    return out


def tank_score(pieces):
    """>0 nghiêng tank đen; so diff giữa 2 người trong frame mới đáng tin."""
    pix = np.concatenate([c.reshape(-1, 3) for c in pieces if c is not None and c.size])
    hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    Hh, Ss, Vv = (hsv[:, i].astype(float) for i in range(3))
    skin = ((Hh <= 25) | (Hh >= 172)) & (Ss >= 40) & (Ss <= 190) & (Vv >= 95)
    black = Vv < 70
    print_c = (~skin) & (Ss > 60) & (Vv > 60)
    sat_h = Hh[(Ss > 60) & (Vv > 60) & ~skin]
    hue_ent = 0.0
    if len(sat_h) > 50:
        p, _ = np.histogram(sat_h, bins=18, range=(0, 180))
        p = p[p > 0] / p.sum()
        hue_ent = float(-(p * np.log(p)).sum())
    return (1.3 * skin.mean() + 0.7 * black.mean()
            - 1.6 * print_c.mean() - 0.25 * hue_ent)


def classify_person(img, person):
    pieces = [c for c in [torso_crop(img, person)] if c is not None]
    pieces += arm_strips(img, person)
    return tank_score(pieces) if pieces else None


# ---------- áp fix ----------

def apply_fixes(indir):
    for cam, fx in FIXES.items():
        path = os.path.join(indir, f"people_id_360-Camera-{cam}.json")
        bak = path + ".pre_camswap2.json"
        d = json.load(open(path))
        if d.get("stats", {}).get("camswap2"):
            print(f"Cam-{cam}: đã có stats.camswap2, bỏ qua (tránh flip 2 lần)")
            continue
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        segs = fx["segments_fixed"]
        n = 0
        for e in d["timeline"]:
            if any(a <= e["t"] <= b for a, b in segs) ^ fx["whole_flip"]:
                for p in e["people"]:
                    p["id"] = 2 if p["id"] == 1 else 1
                n += 1
        d["stats"]["camswap2"] = fx
        json.dump(d, open(path, "w"))
        print(f"Cam-{cam}: flip {n} frame | segs={segs} whole_flip={fx['whole_flip']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    apply_fixes(sys.argv[1])
    print("Xong. Chạy tiếp export_people2d_v3.py và posture_metrics.py.")
