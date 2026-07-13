#!/usr/bin/env python3
"""Đo độ lệch kinh tuyến (yaw) giữa pose_cache jpg và mp4 teleport đang phát.

Hai bản render khác nhau: mp4 export có thể chống rung (yaw TRÔI theo thời gian),
cache stitch thô (gắn với thân máy). Nên offset là HÀM của t chứ không phải hằng số
→ đo bằng cross-correlation vòng theo trục x mỗi STEP giây, ghi loạt mẫu {t, deg}
vào people_2d_v3.json (key "lon_offsets"); teleport lấy mẫu gần nhất khi vẽ.

Usage (app 8001 đang chạy để lấy clip timeline):
  python3 scripts/calib_cache_offset.py <participant> <session_dir> [step_s=60]
"""
import cv2, glob, json, os, re, sys, urllib.request, urllib.parse
import numpy as np

def prep(im):
    g = cv2.cvtColor(cv2.resize(im, (1000, 500)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - g.mean()

def shift_deg(cap, local_t, cache_img):
    cap.set(cv2.CAP_PROP_POS_MSEC, local_t * 1000)
    ok, vf = cap.read()
    if not ok: return None
    cf = cv2.imread(cache_img)
    if cf is None: return None
    A, B = prep(vf), prep(cf)
    cc = np.fft.ifft(np.fft.fft(A, axis=1) * np.conj(np.fft.fft(B, axis=1)), axis=1).real.sum(axis=0)
    i = int(np.argmax(cc))
    # chất lượng đỉnh: (peak - median)/std — đỉnh tốt đo được ~3.8-4.0, nền mờ thấp hơn
    qual = float((cc[i] - np.median(cc)) / (cc.std() + 1e-9))
    sh = i - 1000 if i > 500 else i
    return sh / 1000 * 360.0, qual

def main(participant, sdir, step=60):
    ses = json.load(urllib.request.urlopen(
        f'http://localhost:8001/api/v3/session?participant={urllib.parse.quote(participant)}'))
    cache = os.path.join(sdir, 'pose_cache')
    p2d_f = os.path.join(sdir, 'people_2d_v3.json')
    p2d = json.load(open(p2d_f))
    lon_offsets = {}
    for c in ses['cameras']:
        cam = c['cameraId']
        if not cam.startswith('360'): continue
        ts_all = sorted(int(re.search(r'_(-?\d+)_4000x2000', os.path.basename(f)).group(1))
                        for f in glob.glob(f'{cache}/eq_{cam}_*_4000x2000.jpg'))
        if not ts_all: continue
        samples = []
        camdir = cam.replace('360-', '')
        for cl in c['clips']:
            base = re.sub(r'\.(insv|mp4|MP4|MOV)$', '', cl['filename'])
            hits = glob.glob(os.path.join(sdir, 'videos-360-equirectangular', camdir, base + '.mp4'))
            if not hits: continue
            cap = cv2.VideoCapture(hits[0])
            want = np.arange(cl['offset'] + 10, cl['end'] - 5, step)
            for w in want:
                t = min(ts_all, key=lambda x: abs(x - w))
                if abs(t - w) > step / 2: continue
                r = shift_deg(cap, t - cl['offset'], f'{cache}/eq_{cam}_{t}_4000x2000.jpg')
                if r is None: continue
                deg, qual = r
                if qual < 3.0: continue          # đỉnh yếu -> bỏ mẫu
                samples.append({'t': int(t), 'deg': round(deg, 1)})
            cap.release()
        samples.sort(key=lambda s: s['t'])
        lon_offsets[cam] = samples
        degs = [s['deg'] for s in samples]
        print(f'{cam}: {len(samples)} mẫu | đầu {degs[:3]} … cuối {degs[-3:]}')
    p2d['lon_offsets'] = lon_offsets
    json.dump(p2d, open(p2d_f, 'w'))
    print('updated', p2d_f)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60)
