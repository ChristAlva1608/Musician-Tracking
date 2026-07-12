#!/usr/bin/env python3
"""
Local review UI: posture trend chart + skeleton overlay on the real frames.
Click anywhere on the chart (or use arrow keys) to see that exact moment
with P1/P2 skeletons drawn over the participants.

Usage:
    python3 scripts/review_ui.py "/Volumes/.../pose_cache" \
        [--data-dir src/output/person_id_christina] [--port 8765]
Then open http://localhost:8765
"""

import argparse
import io
import json
import re
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2

ROOT = Path(__file__).resolve().parents[1]
ARGS = None
FRAME_INDEX = {}  # cam -> {t: path}

SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(6,8),(7,9),(8,10),
            (5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]


def build_index(cache_dir: Path):
    pat = re.compile(r'^eq_(360-Camera-\d)_(-?\d+)_')
    for f in cache_dir.iterdir():
        if f.name.startswith('._'):
            continue
        m = pat.match(f.name)
        if m:
            FRAME_INDEX.setdefault(m.group(1), {})[int(m.group(2))] = f


@lru_cache(maxsize=64)
def render_frame(cam: str, t: int, width: int = 1800) -> bytes:
    path = FRAME_INDEX.get(cam, {}).get(t)
    if path is None:
        return b''
    img = cv2.imread(str(path))
    if img is None:
        return b''
    scale = width / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else b''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype='application/json', code=200):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == '/':
            self._send(HTML.encode(), 'text/html; charset=utf-8')
        elif u.path == '/data':
            cam = q['cam'][0]
            f = Path(ARGS.data_dir) / f'people_id_{cam}.json'
            self._send(f.read_bytes() if f.exists() else b'{}')
        elif u.path == '/frame':
            cam, t = q['cam'][0], int(q['t'][0])
            jpg = render_frame(cam, t)
            if jpg:
                self._send(jpg, 'image/jpeg')
            else:
                self._send(b'not found', 'text/plain', 404)
        elif u.path == '/times':
            cam = q['cam'][0]
            ts = sorted(FRAME_INDEX.get(cam, {}))
            self._send(json.dumps(ts).encode())
        else:
            self._send(b'not found', 'text/plain', 404)


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Kiểm posture — skeleton trên người thật</title>
<style>
:root{--bg:#16181a;--panel:#1d2023;--ink:#e6e8e3;--muted:#9aa190;--line:#33383b;
  --p1:#22c55e;--p2:#f97316}
body{background:var(--bg);color:var(--ink);margin:0;
  font:14px/1.45 "Avenir Next",-apple-system,sans-serif}
main{max-width:1860px;margin:0 auto;padding:14px 18px}
h1{font-size:17px;margin:2px 0 10px}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
button,select{background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:7px;padding:5px 12px;font:inherit;cursor:pointer}
button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
.time{font-variant-numeric:tabular-nums;color:var(--muted);min-width:120px}
#chartwrap{background:var(--panel);border:1px solid var(--line);border-radius:9px;
  padding:8px 10px 2px;margin-bottom:10px}
#chart{width:100%;height:150px;display:block;cursor:crosshair}
#viewer{position:relative;background:#000;border-radius:9px;overflow:hidden}
#photo,#overlay{width:100%;display:block}
#overlay{position:absolute;inset:0;pointer-events:none}
.vals{display:flex;gap:18px;margin-top:8px;color:var(--muted);
  font-variant-numeric:tabular-nums;flex-wrap:wrap}
.vals b{font-weight:600}
kbd{background:var(--panel);border:1px solid var(--line);border-radius:4px;
  padding:0 5px;font-size:12px}
</style></head><body><main>
<h1>Kiểm posture — bấm vào chart để xem skeleton trên frame thật <span style="color:var(--muted);font-weight:400">(&larr;/&rarr; = lùi/tới 1 frame, Shift = 10 frame)</span></h1>
<div class="bar">
  <span id="cams"></span>
  <select id="metric">
    <option value="trunk_lean_deg">Nghiêng thân (°)</option>
    <option value="head_forward">Đầu nhô trước</option>
    <option value="shoulder_tilt_deg">Lệch vai (°)</option>
  </select>
  <span class="time" id="time">–</span>
  <span class="vals" id="vals"></span>
</div>
<div id="chartwrap"><canvas id="chart"></canvas></div>
<div id="viewer"><img id="photo"><canvas id="overlay"></canvas></div>
</main><script>
const SK=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[6,8],[7,9],[8,10],
          [5,11],[6,12],[11,12],[11,13],[12,14],[13,15],[14,16]];
const P={1:getComputedStyle(document.documentElement).getPropertyValue('--p1'),
         2:getComputedStyle(document.documentElement).getPropertyValue('--p2')};
const CAMS=["360-Camera-1","360-Camera-2","360-Camera-3"];
let cam="360-Camera-2", data=null, times=[], cur=0, metric="trunk_lean_deg";
const $=id=>document.getElementById(id);

function fmt(s){return Math.floor(s/60)+":"+String(Math.abs(Math.round(s%60))).padStart(2,"0")}

async function loadCam(c){
  cam=c;
  const [d,ts]=await Promise.all([
    fetch('/data?cam='+cam).then(r=>r.json()),
    fetch('/times?cam='+cam).then(r=>r.json())]);
  data=d; times=ts.filter(t=>t>=0); cur=Math.min(cur,times.length-1);
  camsBar(); chart(); show();
}
function camsBar(){
  $('cams').innerHTML="";
  CAMS.forEach(c=>{
    const b=document.createElement('button');
    b.textContent=c.replace('360-','');
    b.setAttribute('aria-pressed',c===cam);
    b.onclick=()=>loadCam(c);
    $('cams').appendChild(b);
  });
}
function metricSeries(pid){
  const out=[];
  if(!data||!data.timeline) return out;
  for(const e of data.timeline){
    const p=e.people.find(x=>x.id===pid);
    if(!p||!p.keypoints.length) continue;
    const kp=p.keypoints;
    const mid=(a,b)=>kp[a][2]>.3&&kp[b][2]>.3?[(kp[a][0]+kp[b][0])/2,(kp[a][1]+kp[b][1])/2]:null;
    const sh=mid(5,6),hip=mid(11,12),ear=mid(3,4);
    let v=null;
    if(metric==='trunk_lean_deg'&&sh&&hip){
      v=Math.atan2(sh[0]-hip[0],hip[1]-sh[1])*180/Math.PI;
      if(Math.abs(v)>40) v=null;
    }else if(metric==='head_forward'&&sh&&hip&&ear){
      const torso=Math.hypot(sh[0]-hip[0],sh[1]-hip[1]);
      if(torso>10){v=(ear[0]-sh[0])/torso; if(Math.abs(v)>0.4)v=null;}
    }else if(metric==='shoulder_tilt_deg'&&kp[5][2]>.3&&kp[6][2]>.3&&sh&&hip){
      const dx=kp[6][0]-kp[5][0],dy=kp[6][1]-kp[5][1];
      const torso=Math.hypot(sh[0]-hip[0],sh[1]-hip[1]);
      if(Math.abs(dx)>0.3*torso){v=((Math.atan2(dy,dx)*180/Math.PI+90)%180)-90;
        if(Math.abs(v)>30)v=null;}
    }
    if(v!==null) out.push([e.t,v]);
  }
  return out;
}
function chart(){
  const cv=$('chart'),ctx=cv.getContext('2d');
  const w=cv.width=cv.clientWidth*devicePixelRatio, h=cv.height=150*devicePixelRatio;
  ctx.clearRect(0,0,w,h);
  const s1=metricSeries(1),s2=metricSeries(2),all=s1.concat(s2);
  if(!all.length) return;
  const tMax=times[times.length-1]||1;
  let lo=Math.min(...all.map(p=>p[1])),hi=Math.max(...all.map(p=>p[1]));
  const pad=(hi-lo)*0.1||1; lo-=pad; hi+=pad;
  const X=t=>t/tMax*w, Y=v=>(1-(v-lo)/(hi-lo))*(h-14*devicePixelRatio)+7*devicePixelRatio;
  ctx.strokeStyle='#33383b';
  ctx.beginPath();ctx.moveTo(0,Y(0));ctx.lineTo(w,Y(0));ctx.stroke();
  [[s1,P[1]],[s2,P[2]]].forEach(([s,col])=>{
    ctx.strokeStyle=col;ctx.lineWidth=1.4*devicePixelRatio;ctx.beginPath();
    s.forEach((p,i)=>{i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1]))});
    ctx.stroke();});
  ctx.strokeStyle='#e6e8e3';ctx.lineWidth=devicePixelRatio;
  const tx=X(times[cur]||0);
  ctx.beginPath();ctx.moveTo(tx,0);ctx.lineTo(tx,h);ctx.stroke();
}
async function show(){
  const t=times[cur];
  $('time').textContent=fmt(t)+" · frame "+(cur+1)+"/"+times.length;
  const img=$('photo');
  img.src='/frame?cam='+cam+'&t='+t;
  await img.decode().catch(()=>{});
  const ov=$('overlay'),ctx=ov.getContext('2d');
  ov.width=img.naturalWidth;ov.height=img.naturalHeight;
  ctx.clearRect(0,0,ov.width,ov.height);
  const entry=(data.timeline||[]).find(e=>e.t===t);
  const sc=img.naturalWidth/4000;
  let vals="";
  if(entry) for(const p of entry.people){
    const col=P[p.id]||'#888';
    ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=3;
    const kp=p.keypoints;
    const [x1,y1,x2,y2]=p.bbox;
    ctx.strokeRect(x1*sc,y1*sc,(x2-x1)*sc,(y2-y1)*sc);
    ctx.font='bold 20px sans-serif';
    ctx.fillText('P'+p.id+' '+p.conf.toFixed(2),x1*sc+4,Math.max(22,y1*sc-6));
    for(const [a,b] of SK){
      if(kp[a]&&kp[b]&&kp[a][2]>.3&&kp[b][2]>.3&&Math.abs(kp[a][0]-kp[b][0])<2000){
        ctx.beginPath();ctx.moveTo(kp[a][0]*sc,kp[a][1]*sc);
        ctx.lineTo(kp[b][0]*sc,kp[b][1]*sc);ctx.stroke();}}
    for(const [x,y,c] of kp) if(c>.3){
      ctx.beginPath();ctx.arc(x*sc,y*sc,4,0,7);ctx.fill();}
    vals+=`<span style="color:${col}"><b>P${p.id}</b></span> conf ${p.conf.toFixed(2)} · khớp tin cậy ${p.keypoints.filter(k=>k[2]>.3).length}/17 &nbsp;`;
  }
  $('vals').innerHTML=vals||'<i>không có người trong frame này</i>';
  chart();
}
$('chart').addEventListener('click',ev=>{
  const r=ev.currentTarget.getBoundingClientRect();
  const t=(ev.clientX-r.left)/r.width*(times[times.length-1]||1);
  let best=0,bd=1e9;
  times.forEach((x,i)=>{const d=Math.abs(x-t);if(d<bd){bd=d;best=i}});
  cur=best;show();
});
$('metric').onchange=e=>{metric=e.target.value;chart();};
document.addEventListener('keydown',ev=>{
  if(ev.key==='ArrowRight'){cur=Math.min(times.length-1,cur+(ev.shiftKey?10:1));show();ev.preventDefault();}
  if(ev.key==='ArrowLeft'){cur=Math.max(0,cur-(ev.shiftKey?10:1));show();ev.preventDefault();}
});
addEventListener('resize',chart);
loadCam(cam);
</script></body></html>"""


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('cache_dir')
    ap.add_argument('--data-dir', default=str(ROOT / 'src/output/person_id_christina'))
    ap.add_argument('--port', type=int, default=8765)
    ARGS = ap.parse_args()
    build_index(Path(ARGS.cache_dir))
    if not FRAME_INDEX:
        sys.exit('No frames found in cache dir')
    print(f"Cameras: {list(FRAME_INDEX)} — open http://localhost:{ARGS.port}")
    ThreadingHTTPServer(('127.0.0.1', ARGS.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
