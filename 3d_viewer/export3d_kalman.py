"""Tracker Kalman per-joint — KHÔNG box IoU (đó là mode greedy riêng).
- Association = so KHỚP-với-KHỚP trên MỌI cặp (track × detection). Không lọc box.
- Kalman mỗi khớp giữ 3 thứ: VỊ TRÍ (x,y) · VẬN TỐC (vx,vy) · ĐỘ CHẮC (hiệp phương sai P).
- "Quá xa = track mới" quyết bằng chính P (khoảng cách Mahalanobis, ngưỡng 3σ thống kê — KHÔNG px chế).
- Track mất khớp → không update khớp đó, không bịa. Track không nối được → coast NỘI BỘ (P phồng) để
  còn tái-bắt, nhưng KHÔNG xuất điểm bịa; P phồng quá (lạc hẳn) thì bỏ khỏi pool.
- wrap-aware ở x (seam 360): mở-cuộn số đo về gần estimate trước khi update; xuất %W.
- Xuất: mỗi det = [x0,y0,x1,y1,t,tid,conf, k] với k[j]=[x,y,conf,vx,vy,sig] (BACK-COMPAT: [0,1,2] như cũ)."""
import json,os,numpy as np
JS="/Volumes/MAML 4TB 1/DetectCache_6fps_v0.01/Christina - MultiCam Data - Piano - 2025-09-19/Camera-2/VID_20250919_140534_00_007.jsonl"
OUT="/private/tmp/claude-501/-Users-ngothanhnhan-Downloads-Multicam-Annotation-v2/77451e8a-fb74-4cd2-b15e-7452ef79dbfe/scratchpad/det3d_kalman.json"
W=4000.0
Q_POS=1.0; Q_VEL=4.0    # nhiễu quá trình (cho phép tăng tốc chút)
R=25.0                  # nhiễu đo ~ (5px)^2
GATE=3.0                # Mahalanobis 2D ~3σ: ngoài vùng bất-định thì KHÔNG nhận → track mới
LOST_SIG=250.0          # σ vị trí vượt (px) = lạc hẳn → bỏ track khỏi pool tái-bắt
def wdx(a,b): d=abs(a-b); return min(d,W-d)
def unwrap(z,ex): return ex+(((z-ex)+W/2)%W)-W/2
# KF 1 trục [pos,vel]: F=[[1,1],[0,1]] H=[1,0]
def kpred(p,v,P00,P01,P11):
    np_=p+v
    n00=P00+2*P01+P11+Q_POS; n01=P01+P11; n11=P11+Q_VEL
    return np_,v,n00,n01,n11
def kupd(p,v,P00,P01,P11,z):
    S=P00+R; K0=P00/S; K1=P01/S; inn=z-p
    return p+K0*inn, v+K1*inn, (1-K0)*P00, (1-K0)*P01, P11-K1*P01
class Joint:
    __slots__=('px','vx','Px00','Px01','Px11','py','vy','Py00','Py01','Py11','psig','ppx','ppy')
    def __init__(self,x,y):
        self.px,self.vx,self.Px00,self.Px01,self.Px11=x,0.0,R,0.0,100.0
        self.py,self.vy,self.Py00,self.Py01,self.Py11=y,0.0,R,0.0,100.0
        self.psig=R**0.5; self.ppx=x; self.ppy=y   # σ + VỊ TRÍ ĐOÁN (tâm vòng) TRƯỚC update
    def predict(self):
        self.px,self.vx,self.Px00,self.Px01,self.Px11=kpred(self.px,self.vx,self.Px00,self.Px01,self.Px11)
        self.py,self.vy,self.Py00,self.Py01,self.Py11=kpred(self.py,self.vy,self.Py00,self.Py01,self.Py11)
        self.psig=((self.Px00+self.Py00)/2)**0.5   # chốt độ-mù + tâm-đoán NGAY SAU predict, TRƯỚC update
        self.ppx=self.px; self.ppy=self.py
    def update(self,zx,zy):
        zx=unwrap(zx,self.px)
        self.px,self.vx,self.Px00,self.Px01,self.Px11=kupd(self.px,self.vx,self.Px00,self.Px01,self.Px11,zx)
        self.py,self.vy,self.Py00,self.Py01,self.Py11=kupd(self.py,self.vy,self.Py00,self.Py01,self.Py11,zy)
    def maha(self,zx,zy):   # khoảng cách Mahalanobis 2D tới số đo (dùng chính P → không cần box)
        dx=wdx(self.px,zx); dy=self.py-zy
        return (dx*dx/(self.Px00+R) + dy*dy/(self.Py00+R))**0.5
    def sig(self): return ((self.Px00+self.Py00)/2)**0.5
frames=[]
for line in open(JS):
    try: r=json.loads(line)
    except: continue
    frames.append((r["t"],[{"b":d["b"],"c":d["c"],"k":d["k"]} for d in r["d"]]))
frames.sort()
tracks=[]      # {joints:{j:Joint}, box, miss:int, id}
active=[]
dets_out=[]; tl={}
for t,ds in frames:
    for ai in active:
        for J in tracks[ai]["joints"].values(): J.predict()
    # cost MỌI cặp (track × det) bằng KHỚP — không box gate
    pairs=[]
    for ai in active:
        T=tracks[ai]["joints"]
        for di,d in enumerate(ds):
            tot=0.0;n=0
            for j in range(17):
                z=d["k"][j]
                if z[2]>0.05 and j in T:
                    tot+=T[j].maha(z[0],z[1]); n+=1
            if n>0: pairs.append((tot/n,ai,di))
    pairs.sort()
    usedA=set();usedD=set();assign={}
    for cost,ai,di in pairs:
        if ai in usedA or di in usedD or cost>GATE: continue
        usedA.add(ai);usedD.add(di);assign[di]=ai
    # update matched + đẻ track mới cho det không nối được
    matched=set(); born=[]
    for di,d in enumerate(ds):
        if di in assign:
            ai=assign[di]; T=tracks[ai]
            for j in range(17):
                z=d["k"][j]
                if z[2]>0.05:
                    if j in T["joints"]: T["joints"][j].update(z[0],z[1])
                    else: T["joints"][j]=Joint(z[0],z[1])
            T["box"]=d["b"]; T["miss"]=0; tid=ai; matched.add(ai)
        else:
            tid=len(tracks)
            js={j:Joint(d["k"][j][0],d["k"][j][1]) for j in range(17) if d["k"][j][2]>0.05}
            tracks.append({"joints":js,"box":d["b"],"miss":0,"id":tid}); born.append(tid)
        # xuất: vị trí ĐÃ MƯỢT + vận tốc + σ cho khớp CÓ; khớp mất → [0,0,0,0,0,0]
        T=tracks[tid]; ks=[]
        for j in range(17):
            z=d["k"][j]
            if z[2]>0.05 and j in T["joints"]:
                J=T["joints"][j]
                ks.append([round(J.px%W),round(J.py),round(float(z[2]),2),
                           round(J.vx,1),round(J.vy,1),round(J.psig,1),round(J.ppx%W),round(J.ppy)])  # +[6,7]=vị trí ĐOÁN (tâm vòng)
            else: ks.append([0,0,0.0,0,0,0,0,0])
        b=d["b"]; dets_out.append([round(b[0]),round(b[1]),round(b[2]),round(b[3]),round(t,2),tid,round(d["c"],3),ks])
    # pool frame kế: track vừa nối + track mới sinh + track coast chưa lạc hẳn (KHÔNG xuất điểm bịa)
    newactive=[]
    for ai in active:
        if ai in matched: newactive.append(ai); continue
        tracks[ai]["miss"]+=1
        js=tracks[ai]["joints"]
        med_sig=np.median([J.sig() for J in js.values()]) if js else 1e9
        if med_sig<LOST_SIG: newactive.append(ai)   # còn cứu được → giữ để tái-bắt
    active=newactive+born
for d in dets_out: tl[str(d[5])]=tl.get(str(d[5]),0)+1
json.dump({"dets":dets_out,"tl":tl},open(OUT,"w"))
print(f"dets {len(dets_out)} · tracks {len(tracks)} · {round(os.path.getsize(OUT)/1e6,1)} MB")
lens=sorted(tl.values(),reverse=True)
print("top track dài:",lens[:8],"· ≥60f:",sum(1 for v in lens if v>=60))
