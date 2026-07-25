"""Tracker JOINT ĐỘC LẬP + PHÂN NHÁNH — MỖI LOẠI KHỚP = 1 track, NHÁNH TRƯỚC/QUYẾT SAU (spec user 2026-07-25).
- KHÔNG gộp 17 loại thành bộ xương. Đầu chỉ nối đầu, vai chỉ nối vai → không ráp lại thành người → KHÔNG chimera.
- Mỗi track = 1 Kalman của 1 khớp (vị trí+vận tốc+σ). Nối trong ĐÚNG rổ loại đó, gate 3σ, conf→R.
- NHÁNH TRƯỚC: hễ có ≥2 số-đo TRONG GATE (Kalman thấy hợp lý) → đẻ nhánh MỌI cái (id mới), nuôi song song.
  KHÔNG có vạch "hoà" bịa (bỏ TIE): data cho thấy tỉ số cost #2/#1 TRƠN, không khe → không ngưỡng (luật 12).
- QUYẾT SAU: nhánh sai đoán trật → hết số-đo trong gate → coast rồi CHẾT; hoặc khi 2 nhánh về cùng chỗ (gộp)
  thì giữ nhánh avg-cost tốt hơn. → "đợi tương lai phá đối xứng" ở TỪNG khớp.
- ★NÉ BUG avg=0: gộp/cắt ƯU TIÊN track DÀI (n lớn) TRƯỚC rồi mới avg thấp → track mới-đẻ (n=0) KHÔNG đá văng track thật.
- Xuất schema ĐIỂM gọn: [px,py,px,py, t, tid, conf, jt, vx,vy,sig, ppx,ppy]  (viewer dùng chung, không đổi)."""
import json,os
JS="/Volumes/MAML 4TB 1/DetectCache_6fps_v0.01/Christina - MultiCam Data - Piano - 2025-09-19/Camera-2/VID_20250919_140534_00_007.jsonl"
OUT="/private/tmp/claude-501/-Users-ngothanhnhan-Downloads-Multicam-Annotation-v2/77451e8a-fb74-4cd2-b15e-7452ef79dbfe/scratchpad/det3d_jointindep.json"
W=4000.0; Q_POS=1.0; Q_VEL=4.0; R=25.0; GATE=3.0; COAST_CAP=6   # GATE=3σ = mức tin cậy chuẩn ~99.7%, dẫn từ σ Kalman (số duy nhất còn lại, có gốc)
MAXH=80        # trần hypo SỐNG mỗi loại khớp AN TOÀN chống nổ (log nếu chạm — không giấu)
def wdx(a,b): d=abs(a-b); return min(d,W-d)
def unwrap(z,ex): return ex+(((z-ex)+W/2)%W)-W/2
def kpred(p,v,P00,P01,P11): return p+v,v,P00+2*P01+P11+Q_POS,P01+P11,P11+Q_VEL
def kupd(p,v,P00,P01,P11,z,Rj):
    S=P00+Rj; K0=P00/S; K1=P01/S; inn=z-p
    return p+K0*inn, v+K1*inn, (1-K0)*P00, (1-K0)*P01, P11-K1*P01
NEXT=[0]; PA={}   # PA[id]=id CHA (gia phả): -1 = gốc; nhánh con nhớ cha để dựng lại chuỗi sinh→tử
class Trk:
    __slots__=('px','vx','Px00','Px01','Px11','py','vy','Py00','Py01','Py11','psig','ppx','ppy','jt','id','miss','acc','n','lastc','parent')
    def __init__(self,x,y,jt,c):
        self.px,self.vx,self.Px00,self.Px01,self.Px11=x,0.0,R,0.0,100.0
        self.py,self.vy,self.Py00,self.Py01,self.Py11=y,0.0,R,0.0,100.0
        self.psig=R**0.5; self.ppx=x; self.ppy=y; self.jt=jt; self.miss=0; self.acc=0.0; self.n=0; self.lastc=c
        self.id=NEXT[0]; NEXT[0]+=1; self.parent=-1; PA[self.id]=-1   # track mới sinh = GỐC (không cha)
    def clone(self):
        h=Trk.__new__(Trk)
        for s in ('px','vx','Px00','Px01','Px11','py','vy','Py00','Py01','Py11','psig','ppx','ppy','jt','miss','acc','n','lastc'):
            setattr(h,s,getattr(self,s))
        h.id=NEXT[0]; NEXT[0]+=1; h.parent=self.id; PA[h.id]=self.id; return h   # nhánh con nhớ CHA = track rẽ ra nó
    def predict(self):
        self.px,self.vx,self.Px00,self.Px01,self.Px11=kpred(self.px,self.vx,self.Px00,self.Px01,self.Px11)
        self.py,self.vy,self.Py00,self.Py01,self.Py11=kpred(self.py,self.vy,self.Py00,self.Py01,self.Py11)
        self.psig=((self.Px00+self.Py00)/2)**0.5; self.ppx=self.px; self.ppy=self.py
    def update(self,zx,zy,conf):
        Rj=R/max(conf,0.02); zx=unwrap(zx,self.px)
        self.px,self.vx,self.Px00,self.Px01,self.Px11=kupd(self.px,self.vx,self.Px00,self.Px01,self.Px11,zx,Rj)
        self.py,self.vy,self.Py00,self.Py01,self.Py11=kupd(self.py,self.vy,self.Py00,self.Py01,self.Py11,zy,Rj)
        self.miss=0; self.lastc=conf
    def maha(self,zx,zy):
        dx=wdx(self.px,zx); dy=self.py-zy
        return (dx*dx/(self.Px00+R)+dy*dy/(self.Py00+R))**0.5
    def avg(self): return self.acc/self.n if self.n else 1e9   # n=0 (newborn) = xấu nhất về cost, nhưng gộp/cắt xét n TRƯỚC
frames=[]
for line in open(JS):
    try: r=json.loads(line)
    except: continue
    frames.append((r["t"],[d["k"] for d in r["d"]]))
frames.sort()
live={j:[] for j in range(17)}   # LOẠI khớp -> list[Trk] sống
dets_out=[]; tl={}; capped=0
def emit(T,t):
    dets_out.append([round(T.px%W),round(T.py),round(T.px%W),round(T.py),round(t,2),T.id,round(float(T.lastc),3),T.jt,
                     round(T.vx,1),round(T.vy,1),round(T.psig,1),round(T.ppx%W),round(T.ppy)])
for t,ds in frames:
    for j in range(17):
        for T in live[j]: T.predict()
    meas={j:[] for j in range(17)}
    for k in ds:
        for j in range(17):
            if k[j][2]>0.01: meas[j].append((k[j][0],k[j][1],k[j][2]))
    for j in range(17):
        Ms=meas[j]; nxt=[]; used=set()
        for T in live[j]:
            cands=[]
            for mi,(mx,my,mc) in enumerate(Ms):
                md=T.maha(mx,my)
                if md<=GATE: cands.append((md,mi))
            cands.sort()
            if not cands:
                T.miss+=1
                if T.miss<=COAST_CAP: nxt.append(T)   # coast, KHÔNG xuất điểm bịa
                continue
            take=cands                                # RẼ MỌI số-đo trong gate (Kalman thấy hợp lý) — không vạch "hoà" bịa
            for _,mi in take: used.add(mi)
            if len(take)==1:
                md,mi=take[0]; mx,my,mc=Ms[mi]; T.update(mx,my,mc); T.acc+=md; T.n+=1; nxt.append(T)
            else:                                     # ≥2 trong gate → mỗi số-đo 1 nhánh (id mới), để tự chết sau
                for md,mi in take:
                    mx,my,mc=Ms[mi]; ch=T.clone(); ch.update(mx,my,mc); ch.acc+=md; ch.n+=1; nxt.append(ch)
        for mi,(mx,my,mc) in enumerate(Ms):           # số-đo chưa ai nhận → đẻ track mới CỦA LOẠI j
            if mi not in used: nxt.append(Trk(mx,my,j,mc))
        # GỘP nhánh về CÙNG chỗ (~15px): giữ track DÀI hơn (n lớn) TRƯỚC, rồi avg thấp → newborn (n=0) không đá văng track thật
        seen={}
        for T in nxt:
            key=(round(T.px/15),round(T.py/15))
            cur=seen.get(key)
            if cur is None or (T.n,-T.avg())>(cur.n,-cur.avg()): seen[key]=T
        surv=list(seen.values())
        if len(surv)>MAXH:                            # trần: bỏ track NGẮN & cost cao trước (giữ dài, tốt)
            surv.sort(key=lambda T:(-T.n,T.avg())); capped+=len(surv)-MAXH; surv=surv[:MAXH]
        live[j]=surv
        for T in surv:                                # XUẤT survivor vừa khớp/đẻ frame này (coast miss>0 → bỏ)
            if T.miss==0: emit(T,t)
for d in dets_out: tl[str(d[5])]=tl.get(str(d[5]),0)+1
pa={k:PA[int(k)] for k in tl}   # gia phả chỉ cho track có xuất điểm
json.dump({"dets":dets_out,"tl":tl,"pa":pa},open(OUT,"w"))
roots=sum(1 for v in pa.values() if v==-1)
print(f"điểm {len(dets_out)} · track {len(tl)} (gốc {roots}, nhánh-con {len(tl)-roots}) · hypo tạo {NEXT[0]} · cắt-trần {capped} · {round(os.path.getsize(OUT)/1e6,1)} MB")
lens=sorted(tl.values(),reverse=True)
print("top dài:",lens[:8],"· ≥6f:",sum(1 for v in lens if v>=6),"· ≥60f:",sum(1 for v in lens if v>=60),"· stub≤2f:",sum(1 for v in lens if v<=2))
