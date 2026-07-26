"""Tầng 0 — DETECTION THÔ (chưa track gì): box + 17 joints mỗi detection, MỌI frame.
- KHÔNG tracking, KHÔNG nối, KHÔNG lọc (giữ conf>0.02 = "YOLO có xuất số"; dưới nữa là rác toạ-độ-0).
- Mỗi detection = 1 dòng, id RIÊNG (did) → viewer KHÔNG nối thành track (không trail).
- Lục địa = MẬT ĐỘ: viewer vẽ điểm cộng-dồn (additive), chỗ đông tự sáng, ghost tự mờ. Không ngưỡng.
- Schema y greedy (3-tuple) để viewer dùng chung: [x0,y0,x1,y1, t, did, conf, k] · k[j]=[x,y,c]. Cờ "raw":1."""
import json,os
JS="/Volumes/MAML 4TB 1/DetectCache_6fps_v0.01/Christina - MultiCam Data - Piano - 2025-09-19/Camera-2/VID_20250919_140534_00_007.jsonl"
OUT="/private/tmp/claude-501/-Users-ngothanhnhan-Downloads-Multicam-Annotation-v2/77451e8a-fb74-4cd2-b15e-7452ef79dbfe/scratchpad/det3d_raw.json"
dets=[]; did=0; njoint=0
for line in open(JS):
    try: r=json.loads(line)
    except: continue
    t=round(r["t"],2)
    for d in r["d"]:
        b=d["b"]; k=d["k"]
        ks=[]
        for j in range(17):
            c=k[j][2]
            if c>0.02: ks.append([round(k[j][0]),round(k[j][1]),round(float(c),2)]); njoint+=1
            else: ks.append([0,0,0.0])
        dets.append([round(b[0]),round(b[1]),round(b[2]),round(b[3]),t,did,round(float(d.get("c",0)),3),ks]); did+=1
json.dump({"dets":dets,"tl":{str(d[5]):1 for d in dets},"raw":1},open(OUT,"w"))
print(f"detection {len(dets)} · joint-điểm {njoint} · {round(os.path.getsize(OUT)/1e6,1)} MB")
