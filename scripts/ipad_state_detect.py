#!/usr/bin/env python3
"""
ipad_state_detect.py — v3. Trang thai man hinh iPad "Live Multicam" theo tung giay.

THAY THE ipad_layout_detect.py (v2). Ly do thay:
  v2 tra ve 1 nhan hinh hoc (non_app/1_view/2_view/4_view) — THIET KE SAI TU GOC.
  Hinh hoc luoi KHONG bang so goc may participant thuc su NHIN THAY:
    - Christina: luoi 4 nhung chi 3 o co hinh (o thu 4 la the xam rong).
    - Bryan t~18-34: luoi 4 nhung CA 4 O deu "No Connection"/trong -> nguoi xem thay 0 goc.
    - Christina t~90-97: 2_view nhung dang MO MENU "Add Angles" de len man.
    - Christina t=0.4: app 1_view nhung nua tren bi NOTIFICATION CENTER cua iPad che.

  => v3 tach ro 2 chieu:
       grid    = hinh hoc luoi app (1 / 2 / 4 / null)
       n_live  = SO O CO HINH THAT (o den / No-Connection KHONG tinh)  <- con so nghien cuu
     + co (flags) cho cac trang thai de len: menu_open, system_overlay, no_signal,
       non_app, transition.

Schema moi giay:
  {"t": 91, "grid": 2, "n_live": 2, "live_tiles": ["H1","H2"], "flags": ["menu_open"]}

Ten o (theo src/output/ipad_cut_preview/ipad_cut_scheme.json):
  1_view: W          2_view: H1,H2          4_view: Q1,Q2,Q3,Q4

--------------------------------------------------------------------------------
BAI HOC KY THUAT GIU TU v2 (dung xoa):
  * iPad screen recording la VFR (fps danh nghia 120, thuc ~58). cv2.VideoCapture
    seek/POS_MSEC quy doi qua fps danh nghia -> LECH ~2x thoi gian. Vi vay MOI decode
    deu qua ffmpeg (real PTS): quet bang filter fps=1, trich frame le bang `-ss` input-seek.
  * Toa do NORMALIZED (0..1) -> chiu duoc phan giai khac 1920x1440.
  * ffmpeg autorotate (mac dinh) xu ly buoi co rotation metadata (Sarah, Dottie&Emmy: -180).
  * CPU-only: ffmpeg + numpy (cv2 chi de ve montage).

--------------------------------------------------------------------------------
DAC TRUNG PHAN LOP (do tren frame that cua 3 buoi, 960x720 gray — xem docstring tung ham):

  tb_flat   the xam phang 34 cua thanh tieu de "Live Multicam"  -> CO = dang o luoi (2/4)
            grid=0.84 | 1_view=0.00 | non_app=0.00-0.37 | overlay=0.05
  tb_black  vung tren cung den kit (letterbox cua 1_view)       -> CO = 1_view
            1_view=1.00 | grid=0.16 | non_app=0.00
  dock_med  do sang dai dock cuoi man hinh                      -> SANG = man hinh Home
            non_app=115-117 | moi trang thai app=0-6
  qbot      "the" o dai y=0.725..0.855 (chi luoi 4 & 1_view co)  -> tach 4 vs 2
            4_view=0.67-1.00 | 2_view=0.00
  sidemarg  le trai/phai x<0.018 (1_view trai het be ngang)     -> tach 1 vs grid khi
            1_view=0.50-1.00 | grid=0.00-0.06                      thanh tieu de bi che

  O "CO HINH" vs "O CHET": do std + mat do canh + ti le pixel sang trong long o.
            o song  : std 41-82, edge 3.3-8.6, lit 0.26-0.99
            o chet  : std 0-12,  edge 0.0-1.4, lit 0.00-0.035   (the trong phang 44,
                                                                 hoac "No Connection")
            -> nguong std>20 & edge>1.9 & lit>0.08 (bien an toan ~2x ca hai phia)

Cach dung:
  # quet full 1 video -> JSON + strip chart
  python3 scripts/ipad_state_detect.py --video <v.MP4> --out <s>/ipad_state_v3.json --chart-dir <dir>
  # validate tren cac moc ground-truth user da cham
  python3 scripts/ipad_state_detect.py --video <v.MP4> --gt "980.8=4/3,91.9=2/2:menu_open" [--debug]
"""
import argparse
import json
import os
import subprocess
from datetime import datetime

import numpy as np

# ---- kich thuoc frame phan tich (giu ti le 4:3 cua nguon 1920x1440) ----
AW, AH = 960, 720

# ---- nguong (calibrate tren frame that 3 buoi, xem docstring) ----
CARD = 18          # pixel > CARD  => co "the" (nen app), duoi = nen trang den
TB_FLAT = 0.50     # thanh tieu de luoi
TB_BLACK = 0.50    # letterbox den cua 1_view
DOCK_MED = 40      # dock man hinh Home
DOCK_LIT = 0.15    # ti le pixel sang o dai dock (bat them app-switcher: dock_med sat nguong
                   # 40.0 nhung dock_lit=0.46; con 1_view that su co dock_lit=0.00)
QBOT = 0.35        # tach luoi 4 vs luoi 2
SIDEMARG = 0.35    # 1_view trai het be ngang
BLANK = 0.12       # man hinh trong tron

# O SONG: std la bang chung chinh (o chet max 11.9 | o song min 39.9 -> nguong 22 = ~1.8x
# ca hai phia). edge/lit chi la chot chan phu: KHONG duoc bat buoc edge cao, vi canh
# quay tu tren xuong (vai/da/vai jean) rat MIN — Jennifer t=853/1030 co std 40-54 nhung
# edge chi 1.63; bat buoc edge>1.9 se ket luan sai la "o chet".
LIVE_STD = 22.0
LIVE_EDGE = 2.0    # chot phu (OR voi lit)
LIVE_LIT = 0.05    # chot phu (OR voi edge)
LIT = 60           # pixel > LIT coi la "sang"
# CUU O BI "CHAY TRANG": camera quay vao vung qua sang -> anh gan nhu trang phang,
# std tut xuong duoi 22 (Jennifer t~1021: lit 0.82->0.98, std 49->28). O CHET thi LUON TOI
# (lit <= 0.035, ca 'the trong' lan 'No Connection'), nen "rat sang" la bang chung o song.
LIVE_BRIGHT_LIT = 0.50
LIVE_BRIGHT_STD = 8.0

# ---- menu "Add Angles" (popover neo goc tren-phai) ----
MENU_BOX = (0.660, 0.075, 0.990, 0.690)
MFILL_BOX = (0.70, 0.092, 0.95, 0.145)   # dai giua thanh tieu de va dinh luoi:
                                         # o luoi day LUON den; co panel -> day xam
MFILL_MIN = 0.85
MENU_FLAT_MIN = 0.40
MENU_MED_MAX = 100

# ---- hinh hoc o (normalized, tu ipad_cut_scheme.json) ----
TILES = {
    "W":  (0.0,    0.125,  1.0,    0.875),
    "H1": (0.0229, 0.3299, 0.4979, 0.6931),
    "H2": (0.5016, 0.3299, 0.9771, 0.6931),
    "Q1": (0.0229, 0.1521, 0.4979, 0.5090),
    "Q2": (0.5016, 0.1521, 0.9771, 0.5090),
    "Q3": (0.0229, 0.5139, 0.4979, 0.8708),
    "Q4": (0.5016, 0.5139, 0.9771, 0.8708),
}
GRID_TILES = {1: ["W"], 2: ["H1", "H2"], 4: ["Q1", "Q2", "Q3", "Q4"]}

# long o: bo vien, bo dai badge ten thiet bi (duoi) va icon pin (tren-phai)
IN_X0, IN_X1, IN_Y0, IN_Y1 = 0.03, 0.93, 0.03, 0.82

FLAGS = ["non_app", "no_signal", "menu_open", "system_overlay", "transition"]


# --------------------------------------------------------------------------
# ffmpeg I/O — real PTS, autorotate
# --------------------------------------------------------------------------

def ffmpeg_gray_stream(path, fps=1.0, w=AW, h=AH, hwaccel=True):
    """Sinh (idx, gray HxW) moi 1/fps giay. Dung filter fps= -> bam real PTS (VFR-safe)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", path, "-vf", f"fps={fps},scale={w}:{h}",
            "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=w * h * 8)
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(w * h)
            if len(buf) < w * h:
                break
            yield idx, np.frombuffer(buf, np.uint8).reshape(h, w)
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


def ffmpeg_frame_at(path, t, gray=True, w=AW, h=AH):
    """Trich 1 frame tai giay t (input-seek -> chinh xac theo real PTS)."""
    fmt = "gray" if gray else "bgr24"
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", path,
         "-frames:v", "1", "-vf", f"scale={w}:{h}", "-pix_fmt", fmt,
         "-f", "rawvideo", "-"], capture_output=True).stdout
    ch = 1 if gray else 3
    need = w * h * ch
    if len(out) < need:
        return None
    a = np.frombuffer(out[:need], np.uint8)
    return a.reshape(h, w) if gray else a.reshape(h, w, 3)


def probe_meta(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration:stream_side_data=rotation",
         "-of", "json", path], capture_output=True, text=True).stdout
    st = (json.loads(out or "{}").get("streams") or [{}])[0]
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    return {"width": st.get("width"), "height": st.get("height"),
            "duration": float(st.get("duration", 0.0) or 0.0), "rotation": rot}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _sl(n, a, b):
    i0, i1 = int(round(a * n)), int(round(b * n))
    return slice(max(0, i0), min(n, max(i0 + 1, i1)))


def _box(g, x0, y0, x1, y1):
    H, W = g.shape[:2]
    return g[_sl(H, y0, y1), _sl(W, x0, x1)]


def _card(g, x0, y0, x1, y1, mask=None):
    """Ti le pixel co 'the' (> CARD) trong vung, chi tinh pixel KHONG bi che.
    Tra (gia_tri, ti_le_pixel_hop_le)."""
    a = _box(g, x0, y0, x1, y1)
    if mask is None:
        return float((a > CARD).mean()), 1.0
    m = _box(mask, x0, y0, x1, y1)
    ok = ~m
    n = int(ok.sum())
    if n == 0:
        return 0.0, 0.0
    return float((a[ok] > CARD).mean()), n / ok.size


# --------------------------------------------------------------------------
# lop che (overlay) — phai tim TRUOC khi doc hinh hoc
# --------------------------------------------------------------------------

def detect_menu(g):
    """Menu 'Add Angles': popover neo goc tren-phai, nen xam phang + chu.

    Bang chung quyet dinh (mfill): dai y=0.092..0.145 ben phai nam GIUA thanh tieu de
    va dinh luoi -> o MOI layout luoi day luon la nen den (mfill=0.00).
    Co panel de len -> day thanh xam (mfill=0.98-1.00).
    Phan biet voi 'the o trong mau xam' (Bryan t=18, panel-box cung phang!):
    the o trong nam DUOI dinh luoi (y>0.152) nen KHONG lam mfill sang.
    """
    mf = float((_box(g, *MFILL_BOX) > 25).mean())
    if mf < MFILL_MIN:
        return False, None
    P = _box(g, *MENU_BOX).astype(np.float32)
    med = float(np.median(P))
    flat = float((np.abs(P - med) <= 12).mean())
    if flat < MENU_FLAT_MIN or med > MENU_MED_MAX:
        return False, None
    return True, {"mfill": mf, "pan_med": med, "pan_flat": flat}


def overlay_bottom(g):
    """Uoc luong day cua lop che keo tu tren xuong.

    Notification/Control Center la panel MO (blur) phu kin be ngang: do sang cao
    nhung mat do canh THAP. Quet tu tren xuong, dung o dong dau tien pha vo tinh chat do.
    """
    H, W = g.shape
    a = g.astype(np.float32)
    rowmed = np.median(a, axis=1)
    rowedge = np.abs(np.diff(a, axis=1)).mean(axis=1)
    y0 = int(0.030 * H)
    ylim = int(0.80 * H)
    bad = 0
    y_ov = y0
    for y in range(y0, ylim):
        if rowmed[y] > 22 and rowedge[y] < 5.0:
            bad = 0
            y_ov = y
        else:
            bad += 1
            if bad >= 4:
                break
    return min(max(y_ov / H, 0.03), 0.80)


def build_mask(g, menu_on, sysovl_on):
    """Mask True = pixel bi lop che, khong duoc dung de doc hinh hoc / do o song."""
    H, W = g.shape
    mask = np.zeros((H, W), bool)
    if menu_on:
        x0, y0, x1, y1 = MENU_BOX
        mask[_sl(H, y0, y1), _sl(W, x0, x1)] = True
    if sysovl_on:
        mask[:_sl(H, 0, overlay_bottom(g)).stop, :] = True
    return mask


# --------------------------------------------------------------------------
# do "o co hinh" — con so nghien cuu
# --------------------------------------------------------------------------

def tile_live(g, name, mask=None):
    """O co hinh that khong?

    Anh camera that  -> std 40-82  (edge 1.6-8.6, lit 0.26-0.99)
    O chet           -> std 0-12   (edge 0.0-1.4, lit 0.00-0.035)
      (2 kieu chet: 'the trong' phang tuyet doi gia tri 44; va 'No Connection' =
       nen toi phang + vai dong chu -> std ~10 do chu, van thap hon o song ~4 lan)
    => std la bang chung chinh; edge HOAC lit chi de chot chan (canh smooth van la o song).
    """
    x0, y0, x1, y1 = TILES[name]
    tw, th = x1 - x0, y1 - y0
    bx0, bx1 = x0 + IN_X0 * tw, x0 + IN_X1 * tw
    by0, by1 = y0 + IN_Y0 * th, y0 + IN_Y1 * th
    a = _box(g, bx0, by0, bx1, by1).astype(np.float32)
    if a.size < 16:
        return False, {}
    covered = 0.0
    if mask is not None:
        m = _box(mask, bx0, by0, bx1, by1)
        covered = float(m.mean())
        if covered > 0.88:            # gan nhu bi che het -> khong the ket luan
            return None, {"covered": covered}
        if covered > 0.02:
            # do tren phan KHONG bi che: lay hang/cot con lai qua bounding box lon nhat
            keep_rows = ~m.all(axis=1)
            keep_cols = ~m.all(axis=0)
            if keep_rows.any() and keep_cols.any():
                a = a[np.ix_(keep_rows, keep_cols)]
                mm = m[np.ix_(keep_rows, keep_cols)]
                if mm.mean() > 0.5:
                    a = np.where(mm, np.nan, a)
    if a.size < 16:
        return None, {"covered": covered}
    v = a[~np.isnan(a)] if np.isnan(a).any() else a
    if v.size < 16:
        return None, {"covered": covered}
    std = float(v.std())
    lit = float((v > LIT).mean())
    b = np.nan_to_num(a, nan=float(np.median(v)))
    edge = float((np.abs(np.diff(b, axis=1)).mean() + np.abs(np.diff(b, axis=0)).mean()) / 2)
    live = (std > LIVE_STD and (lit > LIVE_LIT or edge > LIVE_EDGE)) \
        or (lit > LIVE_BRIGHT_LIT and std > LIVE_BRIGHT_STD)      # o bi chay trang
    return live, {"std": round(std, 1), "edge": round(edge, 2),
                  "lit": round(lit, 3), "covered": round(covered, 2)}


# --------------------------------------------------------------------------
# phan lop 1 frame
# --------------------------------------------------------------------------

def topbar_feats(g):
    tb = _box(g, 0.03, 0.050, 0.34, 0.076).astype(np.float32)
    return {"tb_flat": float((np.abs(tb - 34) <= 7).mean()),
            "tb_black": float((tb < 14).mean())}


def classify(g):
    """Tra (grid, n_live, live_tiles, flags, dbg) cho 1 frame gray.

    THU TU QUAN TRONG: doc THANH TREN CUNG TRUOC, dock sau.
      Neu test dock chay truoc se sai o 1_view khi camera quay DOC (anh chan dung cao,
      tran xuong tan dai dock -> dock_med 64-87) — Christina t=448-451. Thanh tren cung
      cua 1_view luon la letterbox DEN (tb_black 0.84-1.00) nen phan xu duoc ngay.
    """
    H, W = g.shape
    tb = topbar_feats(g)
    dock = _box(g, 0.15, 0.885, 0.80, 0.965)
    dock_med = float(np.median(dock))
    dock_lit = float((dock > LIT).mean())

    flags = []
    dbg = dict(tb, dock_med=dock_med, dock_lit=round(dock_lit, 2))

    in_grid = tb["tb_flat"] >= TB_FLAT        # the xam 34: thanh tieu de "Live Multicam"
    in_1view = tb["tb_black"] >= TB_BLACK     # den kit: letterbox tren o W

    # --- 1. thanh tren cung bi che => hoac man hinh he thong, hoac lop che keo xuong ---
    sysovl_on = False
    if not in_grid and not in_1view:
        if dock_med > DOCK_MED or dock_lit > DOCK_LIT:
            # man hinh Home / app switcher / Notification Center keo HET man
            return None, 0, [], ["non_app"], dict(dbg, why="dock")
        sysovl_on = True      # app van o duoi, chi bi lop che tren dau

    # --- 2. lop che ---
    menu_on, mdbg = detect_menu(g)
    mask = build_mask(g, menu_on, sysovl_on)
    if menu_on:
        flags.append("menu_open")
        dbg.update(mdbg or {})
    if sysovl_on:
        flags.append("system_overlay")

    # --- 3. hinh hoc (chi tren pixel khong bi che) ---
    qtop, _ = _card(g, 0.05, 0.175, 0.95, 0.295, mask)
    qbot, qbot_ok = _card(g, 0.05, 0.725, 0.95, 0.855, mask)
    mid, _ = _card(g, 0.05, 0.360, 0.95, 0.660, mask)
    smL, _ = _card(g, 0.002, 0.55, 0.018, 0.85, mask)
    smR, _ = _card(g, 0.982, 0.55, 0.998, 0.85, mask)
    sidemarg = (smL + smR) / 2
    dbg.update(qtop=round(qtop, 2), qbot=round(qbot, 2), mid=round(mid, 2),
               sidemarg=round(sidemarg, 2))

    # man hinh trong tron (app dang mo / man hinh tat) -> khong doc duoc gi
    if not in_grid and max(qtop, qbot, mid) < BLANK:
        return None, 0, [], ["non_app"], dict(dbg, why="blank")

    if in_grid:
        # thanh tieu de "Live Multicam" con nguyen -> chac chan dang o luoi 2 hoac 4
        grid = 4 if (qbot_ok > 0.3 and qbot >= QBOT) or qtop >= 0.60 else 2
        dbg["why"] = "titlebar"
    elif in_1view:
        grid = 1
        dbg["why"] = "letterbox"
    else:
        # thanh tieu de bi lop che an -> doc hinh hoc tu NUA DUOI (duoi vung bi che)
        if sidemarg >= SIDEMARG:
            grid = 1
        elif qbot_ok > 0.3 and qbot >= QBOT:
            grid = 4
        else:
            grid = 2
        dbg["why"] = "geom_below_overlay"

    # --- 4. dem o co hinh ---
    # QUY UOC n_live = so o co hinh MA NGUOI DUNG NHIN THAY. O bi lop che (menu /
    # notification) phu KIN thi tile_live tra None -> KHONG dem: dung y nghia nghien cuu
    # ("participant thay may goc"), vi o bi che kin thi ho khong thay that.
    # O bi che MOT PHAN van do tren phan con nhin thay duoc.
    live_tiles = []
    for name in GRID_TILES[grid]:
        live, tdbg = tile_live(g, name, mask)
        dbg[name] = tdbg
        if live:
            live_tiles.append(name)
    n_live = len(live_tiles)

    if n_live == 0:
        flags.append("no_signal")
    return grid, n_live, live_tiles, flags, dbg


# --------------------------------------------------------------------------
# timeline -> segments
# --------------------------------------------------------------------------

def mark_transitions(timeline):
    """Blip 1 mau khac ca hai hang xom = frame animation chuyen canh."""
    for i in range(1, len(timeline) - 1):
        k = lambda r: (r["grid"], r["n_live"])
        if k(timeline[i]) != k(timeline[i - 1]) and k(timeline[i]) != k(timeline[i + 1]) \
                and k(timeline[i - 1]) == k(timeline[i + 1]):
            if "transition" not in timeline[i]["flags"]:
                timeline[i]["flags"].append("transition")
    return timeline


def merge_segments(timeline):
    segs = []
    for r in timeline:
        key = (r["grid"], r["n_live"], tuple(r["live_tiles"]))
        if segs and segs[-1]["_k"] == key:
            segs[-1]["t1"] = r["t"]
            segs[-1]["_n"] += 1
            segs[-1]["_flags"].update(r["flags"])
        else:
            segs.append({"_k": key, "t0": r["t"], "t1": r["t"], "grid": r["grid"],
                         "n_live": r["n_live"], "live_tiles": list(r["live_tiles"]),
                         "_flags": set(r["flags"]), "_n": 1})
    out = []
    for s in segs:
        out.append({"t0": s["t0"], "t1": s["t1"], "dur": s["t1"] - s["t0"] + 1,
                    "grid": s["grid"], "n_live": s["n_live"],
                    "live_tiles": s["live_tiles"],
                    "flags": sorted(s["_flags"])})
    return out


# --------------------------------------------------------------------------
# strip chart
# --------------------------------------------------------------------------
NLIVE_COLOR = {0: "#c0392b", 1: "#4c78a8", 2: "#f2a53a", 3: "#4aa56c", 4: "#7e57c2"}
NONAPP_COLOR = "#9e9e9e"
FLAG_MARK = {"menu_open": ("#111111", "^"), "system_overlay": ("#0091ea", "v"),
             "transition": ("#8d6e63", "|")}


def render_strip(timeline, segments, duration, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(16, 2.8), dpi=120)
    for s in segments:
        c = NONAPP_COLOR if s["grid"] is None else NLIVE_COLOR.get(s["n_live"], "#000")
        ax.axvspan(s["t0"], s["t1"] + 1.0, ymin=0.30, ymax=1.0, color=c, lw=0)
    # grid geometry ribbon (mong, duoi cung)
    GC = {None: "#e0e0e0", 1: "#bbdefb", 2: "#ffe0b2", 4: "#c8e6c9"}
    for s in segments:
        ax.axvspan(s["t0"], s["t1"] + 1.0, ymin=0.0, ymax=0.22, color=GC[s["grid"]], lw=0)
    for fl, (c, mk) in FLAG_MARK.items():
        ts = [r["t"] for r in timeline if fl in r["flags"]]
        if ts:
            ax.plot(ts, [1.06] * len(ts), mk, color=c, ms=5, clip_on=False, lw=0)
    ax.set_xlim(0, max(duration, 1))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("thoi gian (s)")
    ax.set_title(title, fontsize=10)
    h = [Patch(color=NLIVE_COLOR[i], label=f"{i} man") for i in range(5)]
    h.append(Patch(color=NONAPP_COLOR, label="non_app"))
    h += [Line2D([], [], color=c, marker=mk, ls="", label=fl)
          for fl, (c, mk) in FLAG_MARK.items()]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.42),
              ncol=9, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def label_of(rec):
    """Nhan nguoi doc duoc: 'LUOI 4 · 3 MAN' + co."""
    if rec["grid"] is None:
        base = "NON-APP"
    else:
        base = f"LUOI {rec['grid']} - {rec['n_live']} MAN"
    fl = [f for f in rec["flags"] if f != "non_app"]
    return base + (" [" + ",".join(fl) + "]" if fl else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out")
    ap.add_argument("--chart-dir")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--gt", help="'t=grid/nlive[:flag|flag],...'  vd '980.8=4/3,91.9=2/2:menu_open'")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.gt:
        n_ok = 0
        items = [p for p in args.gt.split(",") if p.strip()]
        for pair in items:
            spec, exp = pair.split("=")
            t = float(spec)
            exp_fl = []
            if ":" in exp:
                exp, fls = exp.split(":")
                exp_fl = [f for f in fls.split("|") if f]
            eg, en = exp.split("/")
            eg = None if eg in ("none", "null", "-") else int(eg)
            en = int(en)
            g = ffmpeg_frame_at(args.video, t)
            if g is None:
                print(f"?? t={t:8.1f} NO_FRAME"); continue
            grid, n_live, tiles, flags, dbg = classify(g)
            ok = (grid == eg and n_live == en
                  and all(f in flags for f in exp_fl))
            n_ok += ok
            print(f"{'OK ' if ok else 'BAD'} t={t:8.1f}  gt=grid{eg}/{en}live"
                  f"{('+' + '|'.join(exp_fl)) if exp_fl else '':16s}"
                  f"  pred=grid{grid}/{n_live}live {tiles} {flags}")
            if args.debug or not ok:
                print(f"      {dbg}")
        print(f"\nkhop: {n_ok}/{len(items)}")
        return

    meta = probe_meta(args.video)
    if meta["width"] and abs(meta["width"] / meta["height"] - 4 / 3) > 0.02:
        print(f"[warn] nguon khong phai 4:3 ({meta['width']}x{meta['height']}) "
              f"— toa do o co the lech")
    print(f"[scan] {os.path.basename(args.video)}  {meta}")

    timeline = []
    for idx, g in ffmpeg_gray_stream(args.video, fps=args.fps):
        t = idx / args.fps
        grid, n_live, tiles, flags, _ = classify(g)
        timeline.append({"t": round(t, 3), "grid": grid, "n_live": n_live,
                         "live_tiles": tiles, "flags": flags})
        if int(t) % 300 == 0 and abs(t - int(t)) < 1e-6:
            print(f"  t={t:7.0f}s -> grid={grid} n_live={n_live} {flags}", flush=True)

    mark_transitions(timeline)
    segments = merge_segments(timeline)
    duration = timeline[-1]["t"] + 1 if timeline else 0.0

    from collections import Counter
    hist = Counter((r["grid"], r["n_live"]) for r in timeline)
    fl_hist = Counter(f for r in timeline for f in r["flags"])

    out = {
        "video": args.video,
        "detector": "ipad_state_detect.py v3 (grid + n_live + flags; ffmpeg real-PTS)",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "fps_sampled": args.fps,
        "source_resolution": [meta["width"], meta["height"]],
        "rotation_side_data": meta["rotation"],
        "duration_s": meta["duration"],
        "schema": {
            "grid": "hinh hoc luoi app: 1|2|4|null (null = non_app)",
            "n_live": "SO O CO HINH THAT 0..4 (o den / No-Connection khong tinh)",
            "live_tiles": "ten o co hinh: W | H1,H2 | Q1..Q4 (ipad_cut_scheme.json)",
            "flags": FLAGS,
        },
        "meta": {
            "analysis_size": [AW, AH],
            "n_samples": len(timeline),
            "thresholds": {"CARD": CARD, "TB_FLAT": TB_FLAT, "TB_BLACK": TB_BLACK,
                           "DOCK_MED": DOCK_MED, "QBOT": QBOT, "SIDEMARG": SIDEMARG,
                           "LIVE_STD": LIVE_STD, "LIVE_EDGE": LIVE_EDGE,
                           "LIVE_LIT": LIVE_LIT},
            "hist_grid_nlive": {f"grid{k[0]}/{k[1]}live": v
                                for k, v in sorted(hist.items(), key=lambda x: -x[1])},
            "hist_flags": dict(fl_hist),
            "seconds_by_n_live": {str(n): sum(1 for r in timeline if r["n_live"] == n)
                                  for n in range(5)},
        },
        "timeline": timeline,
        "segments": segments,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[json] {args.out}  ({len(timeline)} mau, {len(segments)} segment)")
    print(f"[hist] {out['meta']['hist_grid_nlive']}")
    print(f"[flags] {dict(fl_hist)}")

    if args.chart_dir:
        os.makedirs(args.chart_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.video))[0]
        p = os.path.join(args.chart_dir, f"{base}_state_v3_strip.png")
        render_strip(timeline, segments, duration, p,
                     f"{base} — iPad state v3 (mau = so o co hinh; dai duoi = hinh hoc luoi)")
        print(f"[chart] {p}")


if __name__ == "__main__":
    main()
