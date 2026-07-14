#!/usr/bin/env python3
"""Chạy detector state iPad (v3) cho CẢ 26 buổi — CPU-only, không đụng GPU (YOLO26 đang cày).

Detector đã qua vòng chấm của user: 9/9 mốc user tự phán + 30/30 bài mù (2026-07-13).
Mỗi buổi: mọi video iPad-Screen → <session>/ipad_state_v3.json + strip-chart.
Idempotent: buổi nào đã có JSON thì bỏ qua (trừ --force).

Usage: python3 scripts/run_ipad_cohort.py [--only Tên1,Tên2] [--force]
Log:   /Volumes/MAML 4TB 1/TrackingResults/ipad_state_v3/cohort_ipad.log
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

BASE8 = Path('/Volumes/MAML 8TB 1/Phan Dissertation Data')
OUT4 = Path('/Volumes/MAML 4TB 1/TrackingResults/ipad_state_v3')
REPO = Path(__file__).resolve().parent.parent
LOG = OUT4 / 'cohort_ipad.log'


def log(m):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def ipad_videos(sdir):
    fs = []
    for e in ('*.mov', '*.MOV', '*.mp4', '*.MP4', '*.m4v'):
        fs += glob.glob(str(sdir / 'videos-original' / 'iPad-Screen' / e))
    return sorted(fs, key=os.path.getsize, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only'); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    OUT4.mkdir(parents=True, exist_ok=True)
    sess = [d for d in sorted(BASE8.iterdir())
            if d.is_dir() and ' - MultiCam Data - ' in d.name]
    if a.only:
        want = {w.strip().lower() for w in a.only.split(',')}
        sess = [d for d in sess if d.name.split(' - ')[0].lower() in want]
    log(f'=== iPad state v3: {len(sess)} buổi ===')
    done = 0
    for sdir in sess:
        p = sdir.name.split(' - ')[0]
        vids = ipad_videos(sdir)
        if not vids:
            log(f'{p}: KHÔNG có video iPad — bỏ qua'); continue
        out = sdir / 'ipad_state_v3.json'
        if out.exists() and not a.force:
            log(f'{p}: đã có ipad_state_v3.json — bỏ qua'); done += 1; continue
        merged = {'session': sdir.name, 'videos': []}
        for v in vids:                       # buổi nhiều clip iPad -> gộp, giữ mốc theo từng video
            tmp = OUT4 / f'{p}_{Path(v).stem}.json'
            rc = subprocess.run(
                ['python3', 'scripts/ipad_state_detect.py', '--video', v,
                 '--out', str(tmp), '--chart-dir', str(OUT4)],
                cwd=REPO).returncode
            if rc != 0 or not tmp.exists():
                log(f'!! {p}: LỖI trên {Path(v).name} (rc={rc})'); continue
            d = json.load(open(tmp))
            d['video'] = Path(v).name
            merged['videos'].append(d)
        if merged['videos']:
            json.dump(merged, open(out, 'w'))
            n = sum(len(v.get('timeline', [])) for v in merged['videos'])
            log(f'{p}: XONG — {len(merged["videos"])} video, {n} giây -> {out.name}')
            done += 1
    log(f'=== HẾT: {done}/{len(sess)} buổi có kết quả ===')


if __name__ == '__main__':
    main()
