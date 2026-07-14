#!/usr/bin/env python3
"""Track NGƯỜI TRONG TỪNG Ô MÀN HÌNH iPad, cả cohort.

Vì sao: iPad là màn feedback real-time (Live Multicam) — participant TỰ THẤY mình qua đó.
Biết trong mỗi ô có ai / to nhỏ ra sao = biết họ đang nhìn phần nào của cơ thể mình.
(Không dùng đo tư thế: góc bird's-eye chỉ thấy tay/vai — tư thế lấy từ 360.)

Đầu vào : <session>/ipad_state_v3.json  (lưới + ô nào CÓ HÌNH, đã qua kiểm của user)
          src/output/ipad_cut_preview/ipad_cut_scheme.json  (toạ độ ô, normalized)
Đầu ra  : <session>/ipad_people_v1.json
          {video, timeline:[{t, grid, tiles:{Q1:[{conf,bbox,kp,area_frac}], ...}}], stats}
          bbox/kp = toạ độ TRONG Ô (px), kèm area_frac = diện tích người / diện tích ô
          -> "người choán bao nhiêu phần khung mà participant đang nhìn".

Chạy 1fps, decode qua ffmpeg PTS thật (iPad là VFR!). GPU (MPS) — chia sẻ với YOLO26 360.

Usage: python3 scripts/run_ipad_people.py [--only Tên1,Tên2] [--conf 0.35] [--force]
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipad_state_detect as D

BASE8 = Path('/Volumes/MAML 8TB 1/Phan Dissertation Data')
LOGD = Path('/Volumes/MAML 4TB 1/TrackingResults/ipad_people_v1')
SCHEME = Path(__file__).resolve().parent.parent / 'src/output/ipad_cut_preview/ipad_cut_scheme.json'


def log(m):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"
    print(line, flush=True)
    with open(LOGD / 'cohort_ipad_people.log', 'a') as f:
        f.write(line + '\n')


def bgr_stream(path, fps=1.0):
    """Sinh (giây, ảnh BGR full-res) — MỘT lần decode tuần tự, PTS thật (VFR-safe)."""
    import subprocess
    m = D.probe_meta(path)
    w, h = int(m['width']), int(m['height'])
    if abs(int(m.get('rotation') or 0)) in (90, 270):      # autorotate đổi chiều khung
        w, h = h, w
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-hwaccel', 'videotoolbox', '-i', path,
           '-vf', f'fps={fps}', '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=w * h * 3 * 4)
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            yield idx, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            idx += 1
    finally:
        proc.stdout.close(); proc.wait()


def ipad_videos(sdir):
    fs = []
    for e in ('*.mov', '*.MOV', '*.mp4', '*.MP4'):
        fs += glob.glob(str(sdir / 'videos-original' / 'iPad-Screen' / e))
    return sorted(fs, key=os.path.getsize, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only'); ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    LOGD.mkdir(parents=True, exist_ok=True)
    scheme = json.load(open(SCHEME))
    TN = {g: {n: t['norm'] for n, t in scheme['layouts'][k]['tiles'].items()}
          for g, k in ((1, '1_view'), (2, '2_view'), (4, '4_view'))}

    from ultralytics import YOLO
    model = YOLO('yolo26x-pose.pt')

    sess = [d for d in sorted(BASE8.iterdir())
            if d.is_dir() and ' - MultiCam Data - ' in d.name
            and (d / 'ipad_state_v3.json').exists()]
    if a.only:
        want = {w.strip().lower() for w in a.only.split(',')}
        sess = [d for d in sess if d.name.split(' - ')[0].lower() in want]
    log(f'=== iPad people: {len(sess)} buổi ===')

    for sdir in sess:
        p = sdir.name.split(' - ')[0]
        out = sdir / 'ipad_people_v1.json'
        if out.exists() and not a.force:
            log(f'{p}: đã có — bỏ qua'); continue
        st = json.load(open(sdir / 'ipad_state_v3.json'))
        vids_state = st.get('videos') or [st]     # 3 buổi đầu schema phẳng
        files = ipad_videos(sdir)
        res = {'session': sdir.name, 'videos': []}
        t0 = time.time()
        for vs in vids_state:
            vname = vs.get('video')
            vpath = next((f for f in files if os.path.basename(f) == vname), None) \
                    or (files[0] if len(files) == 1 else None)
            if not vpath:
                log(f'{p}: không khớp file cho {vname} — bỏ'); continue
            byt = {e['t']: e for e in vs.get('timeline', [])
                   if e.get('grid') and e.get('n_live', 0) >= 1}
            tl, n_det = [], 0
            # MỘT luồng BGR tuần tự 1fps (trước đây decode 2 lần: stream gray + seek lại
            # từng giây -> chậm gấp ~10 lần; Colette từng mất 4.1h/buổi vì lỗi này)
            for idx, bgr in bgr_stream(vpath):
                e = byt.get(float(idx))
                if not e: continue
                H, W = bgr.shape[:2]
                tiles = {}
                for tn in e['live_tiles']:
                    box = TN.get(e['grid'], {}).get(tn)
                    if not box: continue
                    x0, y0, x1, y1 = box
                    crop = bgr[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]
                    if crop.size == 0: continue
                    ch, cw = crop.shape[:2]
                    r = model(crop, verbose=False, conf=a.conf)[0]
                    people = []
                    for b, kp in zip(r.boxes, r.keypoints):
                        bb = [round(float(v), 1) for v in b.xyxy[0].tolist()]
                        k = kp.data[0].cpu().numpy()
                        people.append({
                            'conf': round(float(b.conf[0]), 2),
                            'bbox': bb,
                            'area_frac': round(((bb[2]-bb[0])*(bb[3]-bb[1]))/(cw*ch), 3),
                            'kp': [[round(float(x), 1), round(float(y), 1), round(float(c), 2)]
                                   for x, y, c in k],
                        })
                    if people:
                        people.sort(key=lambda q: -q['area_frac'])
                        tiles[tn] = people; n_det += len(people)
                if tiles:
                    tl.append({'t': float(idx), 'grid': e['grid'], 'tiles': tiles})
            res['videos'].append({'video': os.path.basename(vpath), 'timeline': tl,
                                  'n_frames': len(tl), 'n_people': n_det})
            log(f'  {p}/{os.path.basename(vpath)}: {len(tl)} giây, {n_det} người')
        if res['videos']:
            json.dump(res, open(out, 'w'))
            tot = sum(v['n_frames'] for v in res['videos'])
            log(f'{p}: XONG — {tot} giây có người ({time.time()-t0:.0f}s)')
    log('=== HẾT ===')


if __name__ == '__main__':
    main()
