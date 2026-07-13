#!/usr/bin/env python3
"""Chạy cả cohort ở 1fps: trích frame -> YOLO26 person-ID -> relabel -> blip-clean -> export.

- Christina chạy ĐẦU TIÊN (chuẩn đối chiếu với baseline 2s), rồi các buổi khác theo ABC.
- Mọi bước IDEMPOTENT (có marker/backup là bỏ qua) -> kill giữa chừng chạy lại vô tư.
- Data đích theo CLAUDE.md: JPEG cache -> 4TB TrackingCache/<p>; JSON kết quả -> 4TB
  TrackingResults/v1fps_2026-07-13/<p>; people_2d_v3.json -> folder buổi trên 8TB.
- Cần app 8001 đang chạy (extract + export đọc clip offsets qua API).

Usage:  python3 scripts/run_cohort_1fps.py [--only P1,P2] [--skip-extract]
Log:    /Volumes/MAML 4TB 1/TrackingResults/v1fps_2026-07-13/cohort_run.log
"""
import argparse, glob, json, os, re, subprocess, sys, time
from pathlib import Path

BASE8 = Path('/Volumes/MAML 8TB 1/Phan Dissertation Data')
CACHE4 = Path('/Volumes/MAML 4TB 1/TrackingCache')
RES4 = Path('/Volumes/MAML 4TB 1/TrackingResults/v1fps_2026-07-13')
REPO = Path(__file__).resolve().parent.parent
LOG = None

def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def sh(cmd, **kw):
    log('  $ ' + ' '.join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=REPO, **kw).returncode

def sessions():
    out = []
    for d in sorted(BASE8.iterdir()):
        if d.is_dir() and ' - MultiCam Data - ' in d.name \
           and (d / 'videos-360-equirectangular').is_dir():
            out.append((d.name.split(' - ')[0], d))
    out.sort(key=lambda x: (x[0] != 'Christina', x[0].lower()))   # Christina trước
    return out

def cams_of(sdir):
    return sorted('360-' + p.name for p in (sdir / 'videos-360-equirectangular').iterdir()
                  if p.is_dir() and p.name.startswith('Camera-'))

def run_one(p, sdir, skip_extract=False):
    cache = CACHE4 / p
    res = RES4 / p
    cache.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)
    marker = cache / '.extract_done'
    if not skip_extract and not marker.exists():
        rc = sh(['python3', 'scripts/extract_frames_1fps.py', p, sdir, cache])
        if rc != 0: log(f'!! {p}: extract LỖI (rc={rc}) — bỏ buổi này'); return False
        marker.touch()
    else:
        log(f'  {p}: extract đã xong trước đó' if marker.exists() else f'  {p}: extract bị skip')
    for cam in cams_of(sdir):
        out_json = res / f'people_id_{cam}.json'
        if not out_json.exists():
            rc = sh(['python3', 'scripts/person_id_from_cache.py', cache,
                     '--camera', cam, '--step', '1', '--out', res])
            if rc != 0: log(f'!! {p}/{cam}: person_id LỖI (rc={rc})'); continue
        if not (res / f'people_id_{cam}.json.pre_relabel.json').exists() and out_json.exists():
            sh(['python3', 'scripts/relabel_ids_global.py', cache,
                '--camera', cam, '--data-dir', res])
    if not list(res.glob('*.pre_blipclean.json')) and list(res.glob('people_id_*.json')):
        sh(['python3', 'scripts/clean_id_blips.py', res])
    # Export STAGING vào chính thư mục kết quả — KHÔNG đè people_2d_v3.json trên 8TB
    # (teleport đang dùng bản đã QC). Publish sang 8TB là bước riêng SAU khi kiểm nhãn
    # liên-camera. Lưu ý: cache mới trích từ mp4 chống rung nên bản publish mới sẽ
    # KHÔNG cần lon_offsets (khớp viewer tự nhiên).
    if list(res.glob('people_id_*.json')):
        sh(['python3', 'scripts/export_people2d_v3.py', res, res])
    log(f'== {p}: XONG ==')
    return True

def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('--only'); ap.add_argument('--skip-extract', action='store_true')
    a = ap.parse_args()
    RES4.mkdir(parents=True, exist_ok=True)
    LOG = RES4 / 'cohort_run.log'
    ss = sessions()
    if a.only:
        want = {w.strip().lower() for w in a.only.split(',')}
        ss = [s for s in ss if s[0].lower() in want]
    log(f'=== COHORT 1fps: {len(ss)} buổi: {[s[0] for s in ss]} ===')
    for p, sdir in ss:
        log(f'--- {p} ---')
        try: run_one(p, sdir, a.skip_extract)
        except Exception as e: log(f'!! {p}: exception {e!r}')
    log('=== HẾT COHORT ===')

if __name__ == '__main__':
    main()
