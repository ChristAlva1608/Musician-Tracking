#!/usr/bin/env python3
"""TEST: YOLO26x-pose có detect được người trong các ô màn hình iPad không?
Đặc biệt các ô BIRD'S EYE (camera treo trên đầu nhìn xuống phím đàn) — góc hiếm trong COCO.

- Lấy trạng thái từ <session>/ipad_state_v3.json (grid + ô nào có hình).
- Cắt ô theo src/output/ipad_cut_preview/ipad_cut_scheme.json (toạ độ normalized).
- Chạy YOLO26x-pose lên TỪNG Ô, vẽ box + skeleton, ghi conf.
- Montage để mắt người chấm.

Usage:
  python3 scripts/ipad_person_test.py --sessions Christina,Bryan,Richard --n 12
"""
import argparse, glob, json, os, subprocess, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipad_state_detect as D

S = "/Volumes/MAML 8TB 1/Phan Dissertation Data"
SCHEME = 'src/output/ipad_cut_preview/ipad_cut_scheme.json'
OUT = 'src/output/ipad_person_test'
EDGES = [[5,7],[7,9],[6,8],[8,10],[11,13],[13,15],[12,14],[14,16],[5,6],[11,12],[5,11],[6,12],
         [0,1],[0,2],[1,3],[2,4],[0,5],[0,6]]


def sess_dir(name):
    hit = [d for d in glob.glob(f'{S}/*') if os.path.basename(d).split(' - ')[0].lower() == name.lower()]
    return hit[0] if hit else None


def ipad_video(sdir):
    fs = []
    for e in ('*.mov', '*.MOV', '*.mp4', '*.MP4'):
        fs += glob.glob(os.path.join(sdir, 'videos-original', 'iPad-Screen', e))
    return max(fs, key=os.path.getsize) if fs else None


def tiles_norm(scheme, grid):
    """trả {tên ô: (x0,y0,x1,y1) normalized} cho lưới grid"""
    key = {1: '1_view', 2: '2_view', 4: '4_view'}[grid]
    return {name: tuple(t['norm']) for name, t in scheme['layouts'][key]['tiles'].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', default='Christina,Bryan,Richard')
    ap.add_argument('--n', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    scheme = json.load(open(SCHEME))
    from ultralytics import YOLO
    model = YOLO('yolo26x-pose.pt')

    cards = []
    for name in a.sessions.split(','):
        sdir = sess_dir(name.strip())
        if not sdir: print('bỏ qua', name); continue
        st_f = os.path.join(sdir, 'ipad_state_v3.json')
        if not os.path.exists(st_f): print('chưa có state:', name); continue
        st = json.load(open(st_f))
        v = ipad_video(sdir)
        # chọn các giây có nhiều ô hình nhất (nhiều góc để test)
        vids = st.get('videos') or [st]      # 3 buổi đầu: schema phẳng; cohort: bọc trong videos[]
        cand = []
        for vid in vids:
            for e in vid.get('timeline', []):
                if e.get('grid') and e.get('n_live', 0) >= 1 and not (e.get('flags') or []):
                    cand.append(e)
        if not cand: continue
        per_sess = max(1, a.n // len(a.sessions.split(',')))
        cand.sort(key=lambda e: -e['n_live'])
        top = cand[:max(per_sess * 8, 40)]          # ưu tiên frame nhiều ô, rồi rải đều theo thời gian
        top.sort(key=lambda e: e['t'])
        picks = top[::max(1, len(top) // per_sess)][:per_sess]
        for e in picks:
            frame = D.ffmpeg_frame_at(v, e['t'], gray=False)
            if frame is None: continue
            H, W = frame.shape[:2]
            tn = tiles_norm(scheme, e['grid'])
            for tname in e['live_tiles']:
                if tname not in tn: continue
                x0, y0, x1, y1 = tn[tname]
                crop = frame[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]
                if crop.size == 0: continue
                r = model(crop, verbose=False, conf=0.25)[0]
                img = crop.copy()
                n = 0
                for b, kp in zip(r.boxes, r.keypoints):
                    n += 1
                    x1b, y1b, x2b, y2b = [int(v_) for v_ in b.xyxy[0]]
                    cv2.rectangle(img, (x1b, y1b), (x2b, y2b), (60, 220, 60), 2)
                    cv2.putText(img, f'{float(b.conf[0]):.2f}', (x1b, max(14, y1b-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 2)
                    k = kp.data[0].cpu().numpy()
                    for i, j in EDGES:
                        if k[i][2] > .3 and k[j][2] > .3:
                            cv2.line(img, tuple(k[i][:2].astype(int)), tuple(k[j][:2].astype(int)), (255, 160, 40), 2)
                    for p in k:
                        if p[2] > .3: cv2.circle(img, tuple(p[:2].astype(int)), 3, (255, 160, 40), -1)
                img = cv2.resize(img, (400, 300))
                bar = np.zeros((44, 400, 3), np.uint8)
                col = (60, 220, 60) if n else (60, 60, 220)
                cv2.rectangle(bar, (0, 0), (399, 43), col, -1)
                cv2.putText(bar, f'{name[:9]} t={e["t"]:.0f} {tname}: {n} nguoi', (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cards.append(cv2.copyMakeBorder(np.vstack([img, bar]), 2, 2, 2, 2,
                                                cv2.BORDER_CONSTANT, value=(30, 30, 30)))
    per = 6
    rows = [np.hstack(cards[i:i+per] + [np.zeros_like(cards[0])]*(per-len(cards[i:i+per])))
            for i in range(0, len(cards), per)]
    p = f'{OUT}/yolo_in_ipad_tiles.jpg'
    cv2.imwrite(p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'{len(cards)} ô -> {p}')


if __name__ == '__main__':
    main()
