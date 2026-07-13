#!/usr/bin/env python3
"""Trích frame 1fps cho pipeline tracking, THẲNG TỪ mp4 teleport đang phát
(videos-360-equirectangular) — nên tọa độ tracking khớp viewer tự nhiên,
không cần calib_cache_offset như lô Christina (cache cũ stitch thô).

- Đọc video từ ổ 8TB, GHI JPEG sang ổ 4TB (chính sách 2 ổ: song song I/O).
- Tên file: eq_<cameraId>_<t>_4000x2000.jpg với t = GIÂY SESSION (clip.offset + giây cục bộ,
  làm tròn; có thể âm nếu máy bấm trước mốc session) — cùng convention pose_cache cũ.
- Decode VideoToolbox (ffmpeg -hwaccel), scale về 4000x2000, JPEG q3.

Usage (app 8001 đang chạy để lấy clip timeline):
  python3 scripts/extract_frames_1fps.py <participant> <session_dir> <out_dir> [--limit-s N]
Ví dụ:
  python3 scripts/extract_frames_1fps.py Christina \
    "/Volumes/MAML 8TB 1/Phan Dissertation Data/Christina - ..." \
    "/Volumes/MAML 4TB 1/TrackingCache/Christina" --limit-s 30
"""
import glob, json, os, re, shutil, subprocess, sys, tempfile, urllib.parse, urllib.request

def run(participant, sdir, outdir, limit_s=None):
    ses = json.load(urllib.request.urlopen(
        f'http://localhost:8001/api/v3/session?participant={urllib.parse.quote(participant)}'))
    os.makedirs(outdir, exist_ok=True)
    total = 0
    for c in ses['cameras']:
        cam = c['cameraId']
        if not cam.startswith('360'): continue
        camdir = cam.replace('360-', '')
        for cl in c['clips']:
            base = re.sub(r'\.(insv|mp4|MP4|MOV)$', '', cl['filename'])
            hits = glob.glob(os.path.join(sdir, 'videos-360-equirectangular', camdir, base + '.mp4'))
            if not hits:
                print(f'BỎ QUA {cam} {base}: không có mp4 equirect'); continue
            tmp = tempfile.mkdtemp(prefix='eq1fps_')
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-hwaccel', 'videotoolbox',
                   '-i', hits[0]]
            if limit_s: cmd += ['-t', str(limit_s)]
            cmd += ['-vf', 'fps=1,scale=4000:2000', '-q:v', '3', os.path.join(tmp, 'f%06d.jpg')]
            subprocess.run(cmd, check=True)
            n = 0
            for f in sorted(glob.glob(os.path.join(tmp, 'f*.jpg'))):
                i = int(re.search(r'f(\d+)\.jpg', f).group(1)) - 1   # frame i ~ giây cục bộ i
                t = round(cl['offset'] + i)
                dst = os.path.join(outdir, f'eq_{cam}_{t}_4000x2000.jpg')
                if not os.path.exists(dst):   # clip chồng mép: giữ frame đầu tiên gặp
                    shutil.move(f, dst); n += 1
            shutil.rmtree(tmp, ignore_errors=True)
            total += n
            print(f'{cam} {base}: {n} frame')
    print(f'XONG: {total} frame -> {outdir}')

if __name__ == '__main__':
    a = sys.argv
    limit = int(a[a.index('--limit-s') + 1]) if '--limit-s' in a else None
    run(a[1], a[2], a[3], limit)
