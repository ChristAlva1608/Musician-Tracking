# Quick Start Guide - Hybrid Alignment System

## 🚀 Quick Setup (3 Steps)

### Step 1: Validate Your Data

```bash
python3 src/tools/participant_folder_validator.py \
  "/Volumes/X10 Pro 1/Phan Dissertation Data" \
  --export validation_reports/
```

**What it does:**
- Scans all participant folders
- Detects suspicious/wrong files
- Generates 3 CSV reports

**Review these files:**
- `issues_report_*.csv` - Fix before processing
- `file_inventory_*.csv` - Full file listing
- `statistics_*.csv` - Summary statistics

---

### Step 2: Fix Issues Found

Common issues to fix:

| Issue | Action |
|-------|--------|
| Empty template folders | Delete them |
| Participant name in filename | Rename to standard format |
| macOS metadata files (`._*`) | Delete them |
| `.lrv` files (low-res preview) | Delete if not needed |
| Misplaced files | Move to correct folder |
| X3 missing lens files | Find and restore both 00 and 10 files |

---

### Step 3: Run Processing

```bash
python3 src/integrated_video_processor.py \
  --alignment-dir "Processed_Videos/P01_20250709" \
  --no-duration-limit
```

**The system automatically:**
✅ Groups chunks by camera
✅ Aligns first chunks using audio
✅ Detects gaps using timestamps + audio
✅ Reports gaps for your review
✅ Merges chunks preserving timeline
✅ Runs pose detection
✅ Generates output videos

---

## 📋 File Naming Quick Reference

### Raw 360° Files (Keep Original!)

```
VID_YYYYMMDD_HHMMSS_LL_CCC.insv

VID_20250709_191503_00_007.insv
    └── Date: 2025-07-09
        └── Time: 19:15:03
            └── LL (Lens): 00 (front/only lens for X4/X5, front for X3)
                └── CCC (Chunk): 007 (camera-assigned number)

X3 cameras create TWO files per chunk:
VID_20250705_163421_00_021.insv  ← Front hemisphere (LL=00)
VID_20250705_163421_10_021.insv  ← Back hemisphere (LL=10) - must stitch!
```

### Processed Files (After Conversion)

```
P##_Cam#_chunk###.mp4  ← ALL cameras use this format!

Examples:
P01_Cam1_chunk007.mp4  ← 360° camera (from VID_007.insv)
P01_Cam4_chunk001.mp4  ← 2D camera (from IMG_3571.MOV)
P03_Cam2_chunk021.mp4  ← X3 camera (stitched from _00_ + _10_)
└── P01: Participant 1
    └── Cam1: Camera 1
        └── chunk007: Chunk number (preserves 360° numbering, sequential for 2D)
```

### Output Files (System Generated)

```
20251020_143052_detection_P01_20250709_Cam1_X4.mp4
└── When processed: 2025-10-20 14:30:52
    └── Type: detection (pose annotations)
        └── Participant: P01, Date: 2025-07-09
            └── Camera: Cam1, Model: X4
```

---

## 🔍 Gap Detection Logic

| Gap Size | Detection Method | Action |
|----------|-----------------|--------|
| < 5 seconds | Timestamp | ✅ Continuous |
| 5-300 seconds | Timestamp + Audio verification | 🔍 Smart detection |
| > 300 seconds (5 min) | Timestamp | ⚠️ Large gap - preserve in timeline |

**Example Console Output:**

```
📹 Camera 'Cam1_X4': base offset 75.2s (4 chunks)
  ✅ [1] chunk007: 75.2s (audio aligned - FIRST CHUNK)
  ✅ [2] chunk008: 1875.2s (continuous, Δt=0.0s)
  ✅ [3] chunk009: 3675.2s (continuous, Δt=0.0s)
  ⚠️  [4] chunk010: 7212.2s (LARGE GAP: 537s = 8.9min)

Gap Detection Summary:
  ⚠️  Before chunk010: 537.0s (8.9min) [timestamp_large_gap]

🚨 LARGE GAPS (>5 minutes) - Consider splitting sessions:
   Cam1_X4/chunk010: 8.9 minutes
```

---

## ⚠️ When to Split Sessions

### ✅ Keep Together If:
- All chunks continuous (no gaps)
- Small gaps (<5 min) - probably pauses
- Same performance piece

### ⚠️ Consider Splitting If:
- Large gap (>5 min)
- Different pieces/songs
- Participant took break

### ❌ Definitely Split If:
- Gap > 30 minutes
- Different recording days
- Different content/context

---

## 🗂️ Folder Structure

```
Phan Dissertation Data/
├── _Raw_Data_Archive/              # Never modify!
│   └── Jennifer - MultiCam.../
│       ├── 360 Camera 1 - ... X4/
│       │   └── VID_007.insv, VID_008.insv...
│       ├── 360 Camera 2 - ... X5/
│       └── 2D Camera 1 - ... iPhone/
│           └── IMG_3571.MOV, IMG_3572.MOV...
│
├── Processed_Videos/               # Converted 2D videos (ALL use chunk### format!)
│   └── P01_20250709/
│       ├── Cam1_X4/
│       │   ├── P01_Cam1_chunk007.mp4
│       │   └── P01_Cam1_chunk010.mp4 (with gap before)
│       ├── Cam2_X5/
│       └── Cam4_iPhone/
│           ├── P01_Cam4_chunk001.mp4  ← From IMG_3571.MOV
│           └── P01_Cam4_chunk002.mp4  ← From IMG_3572.MOV
│
├── validation_reports/             # From validator tool
│   ├── file_inventory_*.csv
│   ├── issues_report_*.csv
│   └── statistics_*.csv
│
├── participant_mapping.csv         # PRIVATE! P01 → Jennifer
└── conversion_log.csv              # Raw → Processed mapping
```

---

## 🚨 Common Issues & Solutions

### Issue: "No timestamp available - assuming continuous"

**Cause:** Processing converted files without timestamps
**Solution:** Process original `.insv` files first

---

### Issue: "Audio verification failed"

**Cause:** Silent passage or audio extraction error
**Solution:** Review flagged chunks manually, verify gap is correct

---

### Issue: X3 camera missing lens files

**Cause:** X3 creates two files per chunk (00 and 10)
**Solution:** Find both files before processing:
```
VID_20250705_163421_00_021.insv  ← Front lens
VID_20250705_163421_10_021.insv  ← Back lens (MUST have both!)
```

---

### Issue: Privacy - participant name in filename

**Example:** `VID_Sarah short_104315_00_004.insv`
**Solution:** Rename to standard format before processing

---

### Issue: macOS metadata files (`._*`)

**Example:** `._VID_20250709_191503_00_007.insv`
**Solution:** Safe to delete - use validator to find all

---

## 📊 Understanding Validation Reports

### file_inventory_*.csv

```csv
Participant,Camera_Folder,Filename,File_Size_MB,Chunk_Number,Is_360,Is_Suspicious
Jennifer...,360 Camera 1...,VID_007.insv,4096.5,7,Yes,No
```

**Use for:** Complete file inventory, size analysis

---

### issues_report_*.csv

```csv
Participant,Camera_Folder,Issue_Type,Description
Sarah...,360 Camera 1...,File: VID_Sarah...,Non-standard filename (privacy issue)
```

**Use for:** Finding and fixing problems before processing

---

### statistics_*.csv

```csv
Metric,Value
total_participants,23
total_360_files,289
suspicious_files,15
```

**Use for:** Dataset overview and quality metrics

---

## 🎯 Processing Workflow

```
1. RAW DATA
   ↓
2. VALIDATE (participant_folder_validator.py)
   ↓
3. FIX ISSUES (delete templates, rename files, etc.)
   ↓
4. CONVERT 360° → 2D (conversion tool - to be implemented)
   ↓
5. PROCESS (integrated_video_processor.py)
   ↓ ↓ ↓
   ├── Scan & Group Chunks
   ├── Audio Align First Chunks
   ├── Detect Gaps (hybrid method)
   ├── Merge Chunks (preserve gaps)
   └── Run Pose Detection
   ↓
6. OUTPUT VIDEOS
```

---

## 📝 Essential Files to Maintain

### participant_mapping.csv (PRIVATE!)

```csv
participant_id,real_name,date,instrument
P01,Jennifer,2025-07-09,Violin and Piano
P02,Bryan,2025-07-16,Piano Dan Tranh and Voice
```

**Never commit to git! Never share publicly!**

---

### conversion_log.csv

```csv
participant_id,camera_id,raw_file,converted_file,frame_angle,conversion_timestamp
P01,Cam1,VID_007.insv,P01_Cam1_chunk007.mp4,180,2025-10-20 14:30:00
```

**Use for:** Traceability, debugging, reproducibility

---

### camera_setup.csv

```csv
participant_id,camera_id,model,position,num_chunks,has_gaps
P01,Cam1,X4,Front,4,yes
P01,Cam2,X5,Left side,3,no
```

**Use for:** Camera configuration, gap tracking

---

## 🆘 Need Help?

1. **Check console output** - Detailed error messages
2. **Review validation reports** - Pre-processing issues
3. **Read gap detection summary** - Alignment issues
4. **Consult HYBRID_ALIGNMENT_GUIDE.md** - Full documentation

---

**Remember:**
- ✅ Always validate before processing
- ✅ Keep raw data untouched
- ✅ Review gap detection results
- ✅ Maintain mapping files for traceability
- ✅ Document any manual fixes

**Version:** 1.0 | **Last Updated:** 2025-10-20
