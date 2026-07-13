#!/usr/bin/env python3
"""Export people_id_<camera>.json (YOLO26x tiled + person-ID, pixel coords 4000x2000)
sang schema people_2d cua Alignment Tool viewer (goc lon/lat) -> people_2d_v3.json.

Viewer (teleport.html) da co san may ve box/skeleton theo goc:
  lon = (x + 0.5)/W * 360 - 180   (deg, seam o +-180)
  lat = 90 - (y + 0.5)/H * 180    (deg)
Moi nguoi (id 1/2/...) = MOT tracklet, frames sorted theo t. Keypoint conf < KP_MIN -> null.

Usage:
  python3 scripts/export_people2d_v3.py src/output/person_id_christina \
      "/Volumes/MAML 8TB 1/Phan Dissertation Data/Christina - MultiCam Data - Piano - 2025-09-19"
"""
import json, re, sys
from pathlib import Path

KP_MIN = 0.30

def px2lon(x, w): return (x + 0.5) / w * 360.0 - 180.0
def px2lat(y, h): return 90.0 - (y + 0.5) / h * 180.0

def convert(indir: Path):
    cameras = {}
    step_s = None
    for f in sorted(indir.glob('people_id_*.json')):
        cam = re.sub(r'^people_id_|\.json$', '', f.name)
        if '.pre_' in cam:      # bo qua backup .pre_relabel/.pre_blipclean/...
            continue
        d = json.load(open(f))
        w, h = d.get('stats', {}).get('frame_size', [4000, 2000])
        by_id = {}
        ts = sorted(e['t'] for e in d['timeline'])
        if len(ts) > 3:
            gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
            step_s = max(step_s or 0, gaps[len(gaps) // 2])
        for e in d['timeline']:
            for p in e['people']:
                x1, y1, x2, y2 = p['bbox']
                kp = []
                for k in (p.get('keypoints') or []):
                    if k and len(k) >= 3 and k[2] >= KP_MIN:
                        kp.append([round(px2lon(k[0] % w, w), 2), round(px2lat(k[1], h), 2)])
                    else:
                        kp.append(None)
                # bbox co the trum qua seam (x2 > w): lon tam va be rong tinh truoc khi wrap
                cx = (x1 + x2) / 2.0
                rec = {'t': round(float(e['t']), 2),
                       'lon': round(px2lon(cx % w, w), 2),
                       'headlat': round(px2lat(y1, h), 2),
                       'footlat': round(px2lat(y2, h), 2),
                       'wdeg': round((x2 - x1) / w * 360.0, 2),
                       'conf': round(float(p['conf']), 2)}
                if any(kp):
                    rec['kp'] = kp
                by_id.setdefault(int(p['id']), []).append(rec)
        cameras[cam] = {'tracklets': [
            {'pid': pid, 'frames': sorted(frs, key=lambda r: r['t'])}
            for pid, frs in sorted(by_id.items())]}
    return {'cameras': cameras, 'ver': 'v3-yolo26', 'step_s': step_s,
            'source': str(indir)}

if __name__ == '__main__':
    indir = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    out = convert(indir)
    of = outdir / 'people_2d_v3.json'
    if of.exists():   # giữ lon_offsets đã đo (calib_cache_offset.py) khi export lại
        try:
            old = json.load(open(of))
            if old.get('lon_offsets'): out['lon_offsets'] = old['lon_offsets']
        except Exception:
            pass
    json.dump(out, open(of, 'w'))
    n = {c: sum(len(t['frames']) for t in v['tracklets']) for c, v in out['cameras'].items()}
    print('wrote', of, '| frames per camera:', n, '| step_s:', out['step_s'])
