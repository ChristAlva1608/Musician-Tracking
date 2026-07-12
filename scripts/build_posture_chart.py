#!/usr/bin/env python3
"""Build the self-contained posture-trend HTML page (data inlined)."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = (root / 'src/output/person_id_christina/posture_series.json').read_text()
stats = {}
for cam in ['360-Camera-1', '360-Camera-2', '360-Camera-3']:
    d = json.loads((root / f'src/output/person_id_christina/people_id_{cam}.json').read_text())
    stats[cam] = d['stats']

html = """<title>Christina — Posture theo thời gian</title>
<style>
:root{
  --bg:#fbfbf9; --panel:#ffffff; --ink:#20241f; --muted:#68705f; --grid:#e6e8e0;
  --line:#d8dbd0; --p1:#15803d; --p2:#c2410c; --chip:#f0f2ea;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#16181a; --panel:#1d2023; --ink:#e6e8e3; --muted:#9aa190; --grid:#2a2e30;
  --line:#33383b; --p1:#16a34a; --p2:#ea580c; --chip:#24282b;
}}
:root[data-theme="dark"]{
  --bg:#16181a; --panel:#1d2023; --ink:#e6e8e3; --muted:#9aa190; --grid:#2a2e30;
  --line:#33383b; --p1:#16a34a; --p2:#ea580c; --chip:#24282b;
}
:root[data-theme="light"]{
  --bg:#fbfbf9; --panel:#ffffff; --ink:#20241f; --muted:#68705f; --grid:#e6e8e0;
  --line:#d8dbd0; --p1:#15803d; --p2:#c2410c; --chip:#f0f2ea;
}
body{background:var(--bg);color:var(--ink);
  font:15px/1.5 "Avenir Next",Seravek,-apple-system,"Segoe UI",sans-serif;
  margin:0;padding:28px 20px 48px}
main{max-width:1060px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px;text-wrap:balance}
.sub{color:var(--muted);margin:0 0 20px;font-size:13.5px}
.tiles{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:10px 14px;min-width:130px}
.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}
.tile span{font-size:12px;color:var(--muted);letter-spacing:.02em}
.cams{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.cams button{background:var(--chip);border:1px solid var(--line);color:var(--ink);
  border-radius:999px;padding:6px 14px;font:inherit;font-size:13px;cursor:pointer}
.cams button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.cams button:focus-visible{outline:2px solid var(--p1);outline-offset:2px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px 6px;margin-bottom:16px}
.panel h2{font-size:14px;margin:0 0 2px}
.panel .desc{font-size:12px;color:var(--muted);margin:0 0 6px}
.legend{display:flex;gap:14px;font-size:12.5px;color:var(--muted);margin:4px 0 2px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
svg{width:100%;height:190px;display:block}
.tooltip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;padding:6px 9px;font-size:12px;display:none;z-index:9;
  box-shadow:0 2px 10px rgba(0,0,0,.12);font-variant-numeric:tabular-nums}
details{margin-top:10px}
summary{cursor:pointer;color:var(--muted);font-size:13px}
table{border-collapse:collapse;font-size:12.5px;margin-top:8px;font-variant-numeric:tabular-nums}
td,th{border:1px solid var(--line);padding:4px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
.note{font-size:12.5px;color:var(--muted);margin-top:14px;max-width:70ch}
.tblwrap{overflow-x:auto}
</style>
<main>
<h1>Christina — Posture của 2 người theo thời gian</h1>
<p class="sub">YOLO26x phân ô + seam 360 · tách người theo màu áo/quần · 1 frame/~2s ·
P1 = học viên (áo đen, quần đỏ nâu) · P2 = giáo viên (áo hoa văn) · đường = trung vị trượt 5 mẫu (~10s)</p>
<div class="tiles" id="tiles"></div>
<div class="cams" id="cams" role="group" aria-label="Chọn camera"></div>
<div id="panels"></div>
<details><summary>Xem dạng bảng (trung bình mỗi 5 phút)</summary><div class="tblwrap" id="tbl"></div></details>
<p class="note">Góc đo trên ảnh 2D equirectangular nên giá trị tuyệt đối lệch theo góc nhìn từng camera —
đọc <b>xu hướng theo thời gian trong cùng một camera</b>, đừng so trực tiếp giữa các camera.
Chưa tô pha intervention (chờ căn iPad-time → session-time).</p>
</main>
<div class="tooltip" id="tip"></div>
<script>
const DATA = __DATA__;
const STATS = __STATS__;
const METRICS = [
  ["trunk_lean_deg","Nghiêng thân (°)","mid-hông → mid-vai so với phương thẳng đứng; 0 = thẳng · trục khóa ±40° (giá trị ngoài = nhiễu, bị cắt)",[-40,40]],
  ["head_forward","Đầu nhô trước (tỉ lệ thân)","tai lệch ngang so với vai, chia chiều dài thân; dương = nhô trước",[-0.4,0.4]],
  ["shoulder_tilt_deg","Lệch vai (°)","đường nối 2 vai so với phương ngang (bỏ frame nhìn nghiêng)",[-30,30]]
];
const CAMS = Object.keys(DATA);
let cam = "360-Camera-2";
const P = {1:getComputedStyle(document.documentElement).getPropertyValue('--p1'),
           2:getComputedStyle(document.documentElement).getPropertyValue('--p2')};
const fmtT = s => Math.floor(s/60)+":"+String(Math.round(s%60)).padStart(2,"0");

function tiles(){
  const el = document.getElementById('tiles'); el.innerHTML = "";
  CAMS.forEach(c=>{
    const s = STATS[c];
    el.insertAdjacentHTML('beforeend',
      `<div class="tile"><b>${s.both_pct}%</b><span>${c.replace('360-','')} · đủ 2 người<br>${s.frames_processed} frame</span></div>`);
  });
  el.insertAdjacentHTML('beforeend',
    `<div class="tile"><b>99.3%</b><span>Gộp 3 camera<br>không lỗ &gt;6s</span></div>`);
}
function cams(){
  const el = document.getElementById('cams'); el.innerHTML = "";
  CAMS.forEach(c=>{
    const b = document.createElement('button');
    b.textContent = c.replace('360-',''); b.setAttribute('aria-pressed', c===cam);
    b.onclick = ()=>{cam=c; cams(); panels();};
    el.appendChild(b);
  });
}
function extent(vals,pad){let lo=Math.min(...vals),hi=Math.max(...vals);
  const q=vals.slice().sort((a,b)=>a-b);
  lo=q[Math.floor(q.length*0.01)];hi=q[Math.floor(q.length*0.99)];
  const m=(hi-lo)*pad||1;return [lo-m,hi+m];}
function panels(){
  const host = document.getElementById('panels'); host.innerHTML = "";
  METRICS.forEach(([key,label,desc,dom])=>{
    const s1 = DATA[cam]["1"][key]||[], s2 = DATA[cam]["2"][key]||[];
    const all = s1.concat(s2); if(!all.length) return;
    let [y0,y1] = extent(all.map(p=>p[1]),0.15);
    if(dom){y0=Math.max(y0,dom[0]);y1=Math.min(y1,dom[1]);}
    const tMax = Math.max(...all.map(p=>p[0]));
    const W=1000,H=190,L=46,R=40,T=12,B=24;
    const x=t=>L+(t/tMax)*(W-L-R), y=v=>T+(1-(v-y0)/(y1-y0))*(H-T-B);
    let grid="",lbl="";
    for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;
      grid+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/>`;
      lbl+=`<text x="${L-6}" y="${y(v)+4}" text-anchor="end" font-size="10.5" fill="var(--muted)">${v.toFixed(key==='head_forward'?2:0)}</text>`;}
    for(let m=0;m<=tMax/60;m+=10){
      grid+=`<line y1="${T}" y2="${H-B}" x1="${x(m*60)}" x2="${x(m*60)}" stroke="var(--grid)"/>`;
      lbl+=`<text x="${x(m*60)}" y="${H-8}" text-anchor="middle" font-size="10.5" fill="var(--muted)">${m}p</text>`;}
    const path=s=>s.length?"M"+s.map(p=>`${x(p[0]).toFixed(1)},${y(Math.max(y0,Math.min(y1,p[1]))).toFixed(1)}`).join("L"):"";
    const end=(s,id)=>s.length?`<text x="${W-R+5}" y="${y(s[s.length-1][1])+4}" font-size="11" fill="${P[id]}">P${id}</text>`:"";
    host.insertAdjacentHTML('beforeend',`<div class="panel">
      <h2>${label}</h2><p class="desc">${desc}</p>
      <div class="legend"><span><i style="background:${P[1]}"></i>P1 học viên</span>
      <span><i style="background:${P[2]}"></i>P2 giáo viên</span></div>
      <svg viewBox="0 0 ${W} ${H}" data-key="${key}" data-tmax="${tMax}" data-y0="${y0}" data-y1="${y1}">
        ${grid}${lbl}
        <path d="${path(s1)}" fill="none" stroke="${P[1]}" stroke-width="2"/>
        <path d="${path(s2)}" fill="none" stroke="${P[2]}" stroke-width="2"/>
        ${end(s1,1)}${end(s2,2)}
        <line class="cross" y1="${T}" y2="${H-B}" x1="-9" x2="-9" stroke="var(--muted)" stroke-dasharray="3,3"/>
      </svg></div>`);
  });
  hover(); table();
}
function nearest(s,t){let best=null,bd=1e9;
  for(const p of s){const d=Math.abs(p[0]-t);if(d<bd){bd=d;best=p;}}
  return bd<=15?best:null;}
function hover(){
  const tip=document.getElementById('tip');
  document.querySelectorAll('svg').forEach(svg=>{
    const W=1000,L=46,R=40,tMax=+svg.dataset.tmax,key=svg.dataset.key;
    svg.addEventListener('mousemove',ev=>{
      const r=svg.getBoundingClientRect();
      const px=(ev.clientX-r.left)/r.width*W;
      if(px<L||px>W-R){tip.style.display='none';return;}
      const t=(px-L)/(W-L-R)*tMax;
      svg.querySelector('.cross').setAttribute('x1',px);
      svg.querySelector('.cross').setAttribute('x2',px);
      const v1=nearest(DATA[cam]["1"][key]||[],t), v2=nearest(DATA[cam]["2"][key]||[],t);
      tip.innerHTML=`<b>${fmtT(t)}</b><br>`+
        (v1?`<span style="color:${P[1]}">P1</span> ${v1[1]}<br>`:"")+
        (v2?`<span style="color:${P[2]}">P2</span> ${v2[1]}`:"");
      tip.style.display='block';
      tip.style.left=Math.min(ev.clientX+14,innerWidth-140)+'px';
      tip.style.top=(ev.clientY+12)+'px';
    });
    svg.addEventListener('mouseleave',()=>{tip.style.display='none';
      svg.querySelector('.cross').setAttribute('x1',-9);svg.querySelector('.cross').setAttribute('x2',-9);});
  });
}
function table(){
  const host=document.getElementById('tbl');
  const bins={};
  METRICS.forEach(([key])=>{ [1,2].forEach(id=>{
    (DATA[cam][String(id)][key]||[]).forEach(([t,v])=>{
      const b=Math.floor(t/300);(bins[b]=bins[b]||{})[key+id]=(bins[b][key+id]||[]).concat(v);});});});
  const avg=a=>a?(a.reduce((x,y)=>x+y,0)/a.length).toFixed(2):"–";
  let rows=Object.keys(bins).sort((a,b)=>a-b).map(b=>{
    const r=bins[b];
    return `<tr><td>${b*5}–${b*5+5} phút</td>`+
      METRICS.map(([k])=>`<td>${avg(r[k+1])}</td><td>${avg(r[k+2])}</td>`).join("")+`</tr>`;}).join("");
  host.innerHTML=`<table><tr><th>Khoảng</th>${METRICS.map(([,l])=>`<th>${l} P1</th><th>P2</th>`).join("")}</tr>${rows}</table>`;
}
tiles();cams();panels();
new MutationObserver(()=>{P[1]=getComputedStyle(document.documentElement).getPropertyValue('--p1');
  P[2]=getComputedStyle(document.documentElement).getPropertyValue('--p2');panels();})
  .observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script>
"""

html = html.replace('__DATA__', data).replace('__STATS__', json.dumps(stats))
out = root / 'src/output/person_id_christina/posture_trend.html'
out.write_text(html)
print(out, out.stat().st_size // 1024, 'KB')
