#!/usr/bin/env python3
"""Bảng chấm điểm detector state iPad (v3) — NGƯỜI chấm, không phải máy tự chấm.

v3 in nhãn HAI CHIỀU dưới mỗi ô (v2 chỉ in 1 nhãn hình học — thiết kế sai từ gốc):
    "LƯỚI 4 · 3 MÀN"           <- hình học lưới  ·  SỐ Ô CÓ HÌNH THẬT
    + dải cờ: menu_open / system_overlay / no_signal / non_app / transition

Sinh tấm ghép để user duyệt bằng mắt. Mỗi ô: frame thật + SỐ THỨ TỰ + nhãn detector.
User chỉ cần đọc số ô nào sai.

  gt40  — các mốc đã dùng trong đợt audit (đề thi cũ)
  blind — N giây NGẪU NHIÊN chưa từng thấy (kiểm mù, đúng chuẩn nghiên cứu)

Usage:
  python3 scripts/ipad_grade_sheet.py --mode blind --n 30 --seed 20260713
  python3 scripts/ipad_grade_sheet.py --mode gt40
"""
import argparse, glob, json, os, random, re, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipad_state_detect as D

S = "/Volumes/MAML 8TB 1/Phan Dissertation Data"
SESS = {
    'christina': f"{S}/Christina - MultiCam Data - Piano - 2025-09-19",
    'jennifer':  f"{S}/Jennifer - MultiCam Data - Violin and Piano - 2025-07-09",
    'bryan':     f"{S}/Bryan - MultiCam Data - Piano and Dan Tranh and Voice - 2025-07-16",
}
# mau theo SO O CO HINH (thong diep chinh cua bang)
NCOL = {0: (60, 60, 200), 1: (168, 120, 76), 2: (58, 165, 242),
        3: (108, 165, 74), 4: (194, 87, 126)}
NONAPP = (120, 120, 120)
OUT = 'src/output/ipad_cut_preview/grade'


def ipad_video(sdir):
    fs = []
    for e in ('*.mov', '*.MOV', '*.mp4', '*.MP4'):
        fs += glob.glob(os.path.join(sdir, 'videos-original', 'iPad-Screen', e))
    return max(fs, key=os.path.getsize) if fs else None


def predict(path, t):
    g = D.ffmpeg_frame_at(path, t, gray=True)
    if g is None:
        return None, None
    grid, n_live, tiles, flags, _ = D.classify(g)
    bgr = D.ffmpeg_frame_at(path, t, gray=False)
    return {'grid': grid, 'n_live': n_live, 'live_tiles': tiles, 'flags': flags}, bgr


def tile(bgr, n, t, rec, who):
    im = cv2.resize(bgr, (420, 315))
    im = np.vstack([im, np.zeros((88, 420, 3), np.uint8)])
    cv2.putText(im, f'#{n}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(im, f'{who} t={t:.0f}s', (52, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1)
    col = NONAPP if rec['grid'] is None else NCOL[rec['n_live']]
    cv2.rectangle(im, (0, 318), (419, 366), col, -1)
    main = 'NON-APP' if rec['grid'] is None else \
           f"LUOI {rec['grid']} - {rec['n_live']} MAN"
    cv2.putText(im, main, (10, 353), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    if rec['grid'] is not None and rec['live_tiles']:
        cv2.putText(im, ','.join(rec['live_tiles']), (250, 352),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    fl = [f for f in rec['flags'] if f != 'non_app']
    if fl:
        cv2.putText(im, '! ' + ' '.join(fl), (10, 390), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (60, 220, 255), 1)
    return cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(30, 30, 30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['gt40', 'blind'], default='blind')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20260713)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    jobs = []
    if a.mode == 'gt40':
        for f in sorted(glob.glob('src/output/ipad_cut_preview/audit_2025_detections/*.jpg')):
            m = re.match(r'(?:bnd_)?(\w+?)_t(\d+(?:\.\d+)?)', os.path.basename(f))
            if m and m.group(1) in SESS and (m.group(1), float(m.group(2))) not in jobs:
                jobs.append((m.group(1), float(m.group(2))))
    else:
        rnd = random.Random(a.seed)
        for who, sdir in SESS.items():
            f = os.path.join(sdir, 'ipad_state_v3.json')
            dur = None
            if os.path.exists(f):
                d = json.load(open(f))
                if d.get('timeline'):
                    dur = max(x['t'] for x in d['timeline'])
            if not dur:
                v = ipad_video(sdir)
                dur = D.probe_meta(v)['duration'] if v else 0
            for _ in range(a.n // 3):
                jobs.append((who, round(rnd.uniform(5, max(6, dur - 5)), 1)))

    tiles, rows = [], []
    for i, (who, t) in enumerate(jobs, 1):
        rec, bgr = predict(ipad_video(SESS[who]), t)
        if bgr is None:
            continue
        tiles.append(tile(bgr, i, t, rec, who.capitalize()))
        rows.append({'n': i, 'who': who, 't': t, **rec})

    per = 6
    grid = [np.hstack(tiles[i:i + per] +
                      [np.zeros_like(tiles[0])] * (per - len(tiles[i:i + per])))
            for i in range(0, len(tiles), per)]
    tag = a.tag or (f'_seed{a.seed}' if a.mode == 'blind' else '')
    p = f'{OUT}/grade_{a.mode}_v3{tag}.jpg'
    cv2.imwrite(p, np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(rows, open(f'{OUT}/grade_{a.mode}_v3{tag}.json', 'w'), indent=1)
    print(f'{len(rows)} ô -> {p}')
    from collections import Counter
    print('grid/n_live:', dict(Counter(
        ('non_app' if r['grid'] is None else f"luoi{r['grid']}/{r['n_live']}man")
        for r in rows)))
    print('flags:', dict(Counter(f for r in rows for f in r['flags'])))


if __name__ == '__main__':
    main()
