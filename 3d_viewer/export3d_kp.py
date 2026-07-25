"""Dedup + track (greedy, chỉ để tô màu tham khảo) NHƯNG MANG THEO 17 keypoints thật.
Xuất det3d.json: mỗi det = [x0,y0,x1,y1,t,tid,conf, k] với k = 17×[x,y,c] (khớp thật YOLO-pose)."""
import json,os,numpy as np
JS="/Volumes/MAML 4TB 1/DetectCache_6fps_v0.01/Christina - MultiCam Data - Piano - 2025-09-19/Camera-2/VID_20250919_140534_00_007.jsonl"
OUT="/private/tmp/claude-501/-Users-ngothanhnhan-Downloads-Multicam-Annotation-v2/77451e8a-fb74-4cd2-b15e-7452ef79dbfe/scratchpad/det3d.json"
W=4000.0; THR=1.01   # 1.01 = KHÔNG bao giờ gộp = RAW
frames=[]
for line in open(JS):
    try: r=json.loads(line)
    except: continue
    ds=[{"b":d["b"],"c":d["c"],"k":d["k"],"kp":sum(1 for k in d["k"] if k[2]>0.05)} for d in r["d"]]
    frames.append((r["t"],ds))
frames.sort()
def iou1(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    iw=max(0,x1-x0);ih=max(0,y1-y0);I=iw*ih;U=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-I
    return I/U if U>0 else 0
def iou(a,b): return max(iou1(a,b),iou1(a,[b[0]-W,b[1],b[2]-W,b[3]]),iou1(a,[b[0]+W,b[1],b[2]+W,b[3]]))
def dedup(ds):
    order=sorted(range(len(ds)),key=lambda i:(ds[i]["kp"],ds[i]["c"]),reverse=True)
    merged=set(); res=[]
    for i in order:
        if i in merged: continue
        res.append(ds[i])
        for j in range(len(ds)):
            if j!=i and j not in merged and iou(ds[i]["b"],ds[j]["b"])>=THR: merged.add(j)
    return res
tracks=[]; active=[]; dets=[]; tl={}
for fi,(t,ds0) in enumerate(frames):
    ds=dedup(ds0); assign=[None]*len(ds)
    if active:
        pr=[]
        for ai in active:
            lb=tracks[ai][-1]
            for j,d in enumerate(ds):
                v=iou(lb,d["b"])
                if v>0: pr.append((v,ai,j))
        pr.sort(reverse=True); uA=set(); uJ=set()
        for v,ai,j in pr:
            if ai in uA or j in uJ: continue
            uA.add(ai); uJ.add(j); assign[j]=ai; tracks[ai].append(ds[j]["b"])
    newactive=[ai for ai in active if len(tracks[ai]) and assign.count(ai)]
    # xây lại active đúng: track nào vừa được nối ở frame này
    got=set(a for a in assign if a is not None); newactive=list(got)
    for j,d in enumerate(ds):
        if assign[j] is None:
            nid=len(tracks); tracks.append([d["b"]]); assign[j]=nid; newactive.append(nid)
    active=newactive
    for j,d in enumerate(ds):
        b=d["b"]; k=[[round(x),round(y),round(float(c),2)] for x,y,c in d["k"]]
        dets.append([round(b[0]),round(b[1]),round(b[2]),round(b[3]),round(t,2),assign[j],round(d["c"],3),k])
for tid,tr in enumerate(tracks): tl[str(tid)]=len(tr)
json.dump({"dets":dets,"tl":tl},open(OUT,"w"))
print(f"dets {len(dets)} · tracks {len(tracks)} · {round(os.path.getsize(OUT)/1e6,1)} MB (có keypoints)")
