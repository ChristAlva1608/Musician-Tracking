# Hybrid Chunk Alignment Guide (Solution 3)

## Overview

This system implements **Solution 3: Hybrid Alignment** for processing musician performance videos with multiple cameras and chunked recordings.

### Key Features

✅ **Audio alignment for first chunks** - Precise synchronization across cameras
✅ **Timestamp-based gap detection** - Automatic detection of recording breaks
✅ **Audio verification for ambiguous cases** - Smart detection of 5-300 second gaps
✅ **Automatic gap preservation** - Timeline integrity maintained
✅ **File validation and reporting** - Comprehensive data quality checks

---

## Table of Contents

1. [File Naming Conventions](#file-naming-conventions)
2. [Folder Structure](#folder-structure)
3. [How the Alignment Works](#how-the-alignment-works)
4. [Gap Detection Logic](#gap-detection-logic)
5. [File Validation](#file-validation)
6. [Running the System](#running-the-system)
7. [Interpreting Results](#interpreting-results)
8. [Common Issues](#common-issues)

---

## File Naming Conventions

### Raw 360° Video Files (Insta360)

**Format:** `VID_YYYYMMDD_HHMMSS_LL_CCC.insv`

| Component | Description | Example | Notes |
|-----------|-------------|---------|-------|
| `VID` | Prefix (fixed) | `VID` | All Insta360 files |
| `YYYYMMDD` | Recording date | `20250709` | July 9, 2025 |
| `HHMMSS` | Recording start time | `191503` | 19:15:03 (7:15:03 PM) |
| **LL** | **Lens identifier** | **`00`** or **`10`** | **00** = front/only lens (X4/X5/X3)<br>**10** = back lens (X3 ONLY) |
| **CCC** | **Chunk number** | **`007`**, **`021`** | Camera-assigned sequential number |

**Understanding LL (Lens Identifier):**
- **X4/X5 cameras:** Only use `00` (camera stitches internally, outputs ONE file)
- **X3 cameras:** Use both `00` AND `10` (outputs TWO files per chunk that must be stitched)

**Understanding CCC (Chunk Number):**
- Assigned by camera automatically
- Increments across the camera's lifetime (not per session!)
- May start at `007` if camera has recorded before
- **Same chunk number** on X3 `_00_` and `_10_` files = same recording moment

**Examples:**

**X4/X5 Cameras (single file per chunk):**
```
VID_20250709_191503_00_007.insv    # Jennifer, Camera 1, Chunk 7, started 19:15:03
VID_20250709_191503_00_008.insv    # Jennifer, Camera 1, Chunk 8, continuous
VID_20250709_203100_00_010.insv    # Jennifer, Camera 1, Chunk 10, started 20:31:00 (GAP!)
```

**X3 Cameras (dual files per chunk - must stitch!):**
```
VID_20250705_163421_00_021.insv    # David, X3 front hemisphere (LL=00)
VID_20250705_163421_10_021.insv    # David, X3 back hemisphere (LL=10) ← SAME chunk!
                            ^^                                    ^^^
                        Same timestamp                      Same chunk number

These TWO files must be stitched together to create one 360° video.
```

### Raw 2D Video Files

**iPhone:** `IMG_####.MOV`
```
IMG_3571.MOV    # Original iPhone naming
IMG_3572.MOV
IMG_3573.MOV
```

**GoPro:** `GX######.MP4`
```
GX010303.MP4    # Original GoPro naming
GX010304.MP4
GX010305.MP4
```

**Important:** These will be mapped to sequential chunk numbers during conversion.

---

## Conversion Workflow

### For X4/X5 Cameras (Single File)

```
1. Raw file:    VID_20250709_191503_00_007.insv
                ↓
2. Select frame angle (e.g., 180° for front view)
                ↓
3. Extract 2D frame from 360° video
                ↓
4. Output:      P01_Cam1_chunk007.mp4
```

### For X3 Cameras (Dual Files - Must Stitch First!)

```
1. Raw files:   VID_20250705_163421_00_021.insv (front hemisphere)
                VID_20250705_163421_10_021.insv (back hemisphere)
                ↓
2. Stitch together using Insta360 Studio
                ↓
3. Stitched:    VID_021_stitched.mp4 (full 360°)
                ↓
4. Select frame angle (e.g., 180°)
                ↓
5. Extract 2D frame from 360° video
                ↓
6. Output:      P03_Cam2_chunk021.mp4
```

**Why stitch first?**
- ✅ Full 360° coverage - can choose any viewing angle
- ✅ Same workflow as X4/X5 cameras
- ✅ Consistent quality

### For 2D Cameras (iPhone/GoPro - Copy with Renaming)

```
1. Raw files (chronological order):
   IMG_3571.MOV (earliest)
   IMG_3572.MOV
   IMG_3573.MOV (latest)
                ↓
2. Map to sequential chunk numbers:
   IMG_3571.MOV → chunk001
   IMG_3572.MOV → chunk002
   IMG_3573.MOV → chunk003
                ↓
3. Copy/transcode with new names:
   P01_Cam4_chunk001.mp4
   P01_Cam4_chunk002.mp4
   P01_Cam4_chunk003.mp4
```

**Benefits of chunk numbering for 2D cameras:**
- ✅ Consistent naming across all cameras
- ✅ Timeline alignment works the same way
- ✅ Easier to process and manage

### Processed Video Files (After Conversion)

**Format:** `P##_Cam#_chunk###.mp4`

| Component | Description | Example |
|-----------|-------------|---------|
| `P##` | Participant ID (anonymous) | `P01` = Jennifer |
| `Cam#` | Camera number | `Cam1` = Camera 1 |
| `chunk###` | Sequential chunk number | `chunk007` (preserves raw file numbering for 360°, sequential for 2D) |

**Examples (360° cameras):**
```
P01_Cam1_chunk007.mp4    # Jennifer, Camera 1, Chunk 7 (from VID_007.insv)
P01_Cam1_chunk008.mp4    # Jennifer, Camera 1, Chunk 8 (from VID_008.insv)
P01_Cam1_chunk010.mp4    # Jennifer, Camera 1, Chunk 10 (with gap before)
```

**Examples (X3 Cameras - stitch before conversion):**
```
P03_Cam2_chunk021.mp4    # David, Camera 2, Chunk 21 (stitched from _00_ + _10_ files)
P03_Cam2_chunk022.mp4    # David, Camera 2, Chunk 22 (stitched from _00_ + _10_ files)
```

**Examples (2D Cameras - iPhone/GoPro):**
```
P01_Cam4_chunk001.mp4    # Jennifer, iPhone, Chunk 1 (from IMG_3571.MOV)
P01_Cam4_chunk002.mp4    # Jennifer, iPhone, Chunk 2 (from IMG_3572.MOV)
P03_Cam3_chunk001.mp4    # David, GoPro, Chunk 1 (from GX010303.MP4)
```

**All cameras now use consistent `chunk###` format!**

### System Output Files

**Aligned Videos (merged chunks):**
```
20251020_143052_P01_20250709_Cam1_X4.mp4
```

**Format:** `TIMESTAMP_PARTICIPANT_DATE_CAMERA.mp4`
- `20251020_143052` = Processing timestamp (when you ran the script)
- `P01_20250709` = Participant and recording date
- `Cam1_X4` = Camera and model

**Detection Videos (with pose annotations):**
```
20251020_143052_detection_P01_20250709_Cam1_X4.mp4
```

---

## Folder Structure

### Recommended Organization

```
/Volumes/X10 Pro 1/Phan Dissertation Data/
│
├── _Raw_Data_Archive/                           # Original data (never modify!)
│   ├── Jennifer - MultiCam Data - Violin and Piano - 2025-07-09/
│   │   ├── 360 Camera 1 - Jennifer - Violin and Piano - 2025-07-09 X4/
│   │   │   ├── VID_20250709_191503_00_007.insv    # KEEP ORIGINAL
│   │   │   ├── VID_20250709_191503_00_008.insv
│   │   │   ├── VID_20250709_191503_00_009.insv
│   │   │   ├── VID_20250709_203100_00_010.insv    # Has gap before
│   │   │   └── LRV_20250709_191503_01_007.lrv     # Low-res preview (can delete)
│   │   ├── 360 Camera 2 - Jennifer - Violin and Piano - 2025-07-09 X5 1/
│   │   │   └── ... (3 chunks, continuous)
│   │   └── 2D Camera 1 - Jennifer - Violin and Piano - 2025-07-09 iPhone 16/
│   │       └── ... (iPhone videos)
│   └── ... (other participants)
│
├── Processed_Videos/                            # Converted 2D videos
│   ├── P01_20250709/                            # Jennifer
│   │   ├── Cam1_X4/
│   │   │   ├── P01_Cam1_chunk007.mp4
│   │   │   ├── P01_Cam1_chunk008.mp4
│   │   │   ├── P01_Cam1_chunk009.mp4
│   │   │   └── P01_Cam1_chunk010.mp4            # Gap before this
│   │   ├── Cam2_X5/
│   │   ├── Cam3_X5/
│   │   └── Cam4_iPhone/                         # 2D camera (consistent chunk naming)
│   │       ├── P01_Cam4_chunk001.mp4
│   │       ├── P01_Cam4_chunk002.mp4
│   │       └── P01_Cam4_chunk003.mp4
│   │
│   ├── P03_20250705/                            # David (has X3!)
│   │   ├── Cam1_X4/
│   │   ├── Cam2_X3/                             # X3 (stitched before conversion)
│   │   │   ├── P03_Cam2_chunk021.mp4            # Stitched from _00_ + _10_
│   │   │   └── P03_Cam2_chunk022.mp4
│   │   └── Cam3_GoPro/
│   │       ├── P03_Cam3_chunk001.mp4
│   │       └── P03_Cam3_chunk002.mp4
│   └── ...
│
├── participant_mapping.csv                       # P01 → Jennifer (PRIVATE!)
├── conversion_log.csv                            # Raw → Processed mapping
├── validation_reports/                          # From validator tool
│   ├── file_inventory_20251020_143052.csv
│   ├── issues_report_20251020_143052.csv
│   └── statistics_20251020_143052.csv
│
└── src/output/                                  # System outputs
    ├── aligned_videos/                          # Merged chunks
    │   ├── 20251020_143052_P01_20250709_Cam1_X4.mp4
    │   └── ...
    └── annotated_detection_videos/              # With pose detection
        └── ...
```

---

## How the Alignment Works

### Step-by-Step Process

#### **Step 1: Scan and Group Chunks**

```
Input: Processed_Videos/P01_20250709/

Scans all camera folders and groups chunks:
  Cam1_X4: 4 chunks (007, 008, 009, 010)
  Cam2_X5: 3 chunks (004, 005, 006)
  Cam3_X5: 3 chunks (004, 005, 006)
```

#### **Step 2: Audio Alignment of First Chunks**

```
Extracts audio from first chunk of each camera:
  Cam1_X4/chunk007.mp4 → Audio pattern analysis
  Cam2_X5/chunk004.mp4 → Audio pattern analysis
  Cam3_X5/chunk004.mp4 → Audio pattern analysis

Cross-correlates audio to detect synchronization:
  Cam1_X4: 0.0s (earliest start) ← REFERENCE
  Cam2_X5: 75.2s (started 75s later)
  Cam3_X5: 92.5s (started 92s later)
```

#### **Step 3: Hybrid Alignment for Subsequent Chunks**

For each subsequent chunk, the system uses a 3-tier approach:

##### **Tier 1: Timestamp Analysis**

```python
# Extract timestamps from filenames
chunk_time = 2025-07-09 20:31:00  # From VID_20250709_203100_00_010.insv
prev_chunk_end = 2025-07-09 20:22:03

# Calculate actual gap
actual_gap = chunk_time - prev_chunk_end = 537 seconds (8.95 minutes)
expected_gap = prev_chunk_duration = 0 seconds (if continuous)

timestamp_delta = 537 seconds
```

**Decision logic:**
- **Δt < 5 seconds**: Chunks are continuous ✅
- **Δt > 300 seconds**: Large gap, definitely separate ⚠️
- **5s < Δt < 300s**: Ambiguous, verify with audio 🔍

##### **Tier 2: Audio Verification (for ambiguous cases)**

```python
# Extract audio segments
prev_audio = last 10 seconds of previous chunk
curr_audio = first 10 seconds of current chunk

# Calculate similarity
similarity = cross_correlation(prev_audio, curr_audio)

if similarity > 0.3:
    # Audio is similar → continuous recording
    → Ignore timestamp gap (might be camera clock drift)
else:
    # Audio is different → true gap
    → Preserve gap in timeline
```

##### **Tier 3: Fallback**

If timestamp or audio verification fails:
- Assume continuous (old behavior)
- Mark with warning for manual review

#### **Step 4: Gap Detection Report**

```
Gap Detection Summary:
  Camera 'Cam1_X4': 1 gap detected
    ⚠️  Before P01_Cam1_chunk010.mp4: 537.0s (8.95min) [timestamp_large_gap]

  Total gaps: 1

  Large gaps (>5 min) - Consider splitting sessions:
    Cam1_X4/P01_Cam1_chunk010.mp4: 8.95 minutes
```

#### **Step 5: Timeline Verification**

```
Final Timeline:
  Cam1_X4_chunk007: 0.0s (REFERENCE)
  Cam2_X5_chunk004: 75.2s
  Cam3_X5_chunk004: 92.5s
  Cam1_X4_chunk008: 1800.0s
  Cam2_X5_chunk005: 1875.2s
  ...
  Cam1_X4_chunk010: 4137.0s [GAP: 537s before]  ← Gap preserved!
```

---

## Gap Detection Logic

### Continuous Chunks (No Gap)

**Example: Jennifer Camera 2**
```
VID_20250709_191618_00_004.insv  →  19:16:18 start
VID_20250709_191618_00_005.insv  →  19:46:18 start (30 min later, expected!)
VID_20250709_191618_00_006.insv  →  20:16:18 start (30 min later, expected!)

Result: All continuous ✅
```

### Small Gap (Ambiguous)

**Example: 30-second pause**
```
Chunk 3 ends:   20:00:00
Chunk 4 starts: 20:00:30  (30 second gap)

Timestamp delta: 30 seconds
→ Trigger audio verification 🔍

Audio similarity: 0.75 (high)
→ Decision: Continuous (probably just a pause in performance)
```

### Large Gap (Definite Break)

**Example: Jennifer Camera 1**
```
Chunk 009 ends:   20:22:03
Chunk 010 starts: 20:31:00  (8.95 minute gap!)

Timestamp delta: 537 seconds
→ Decision: Large gap ⚠️
→ Preserve gap in timeline
→ Suggest splitting into separate sessions
```

---

## File Validation

### Running the Validator

```bash
# Scan participant folders
python3 src/tools/participant_folder_validator.py \
  "/Volumes/X10 Pro 1/Phan Dissertation Data" \
  --export validation_reports/

# Output:
# ✅ File inventory exported to: validation_reports/file_inventory_20251020_143052.csv
# ✅ Issues report exported to: validation_reports/issues_report_20251020_143052.csv
# ✅ Statistics exported to: validation_reports/statistics_20251020_143052.csv
```

### What Gets Checked

#### ✅ **File Inventory**
- All video files detected
- File sizes and extensions
- Camera models identified
- Chunk numbers extracted
- Recording timestamps parsed

#### ⚠️ **Suspicious Files**
- Very small files (< 1MB) - possibly corrupted
- Unexpected file types in camera folders
- Non-standard filenames (privacy risk)
- macOS metadata files (`._*`)
- Low-resolution preview files (`.lrv`)

#### 🔍 **Folder Issues**
- Empty camera folders
- Mixed file types
- X3 cameras missing lens files (00 or 10)
- Misplaced files (GoPro files in iPhone folder)

### Example Validation Output

```
================================================================================
PARTICIPANT FOLDER VALIDATION
================================================================================
Base Directory: /Volumes/X10 Pro 1/Phan Dissertation Data

================================================================================
Scanning: Jennifer - MultiCam Data - Violin and Piano - 2025-07-09
================================================================================
  📁 360 Camera 1 - Jennifer - Violin and Piano - 2025-07-09 X4: 8 files (32.45GB)
  📁 360 Camera 2 - Jennifer - Violin and Piano - 2025-07-09 X5 1: 6 files (58.62GB)
  📁 360 Camera 3 - Jennifer - Violin and Piano - 2025-07-09 X5 2: 6 files (58.62GB)
  📁 2D Camera 1 - Jennifer - Violin and Piano - 2025-07-09 iPhone 16: 10 files (12.34GB)

================================================================================
VALIDATION SUMMARY
================================================================================

Total Participants: 23
Total Camera Folders: 71
Total Video Files: 457
  - 360° Files (.insv): 289
  - 2D Files (.mp4/.mov): 168
Total Data Size: 1245.67 GB

Suspicious Files: 15
Participants with Issues: 3

================================================================================
ISSUES DETECTED
================================================================================

⚠️  Sarah - MultiCam Data - Piano - 2025-07-05/360 Camera 1 - Sarah - Piano - 2025-07-05 X4/VID_Sarah short_104315_00_004.insv
    Issue: Non-standard Insta360 filename (possible privacy issue)

⚠️  David - MultiCam Data - Trombone and Dan Tranh - 2025-07-05/2D Camera 2 - David - Trombone and Dan Tranh - 2025-07-05 iPhone 12 Tuyen Moi
    Issue: Unexpected file type in 2D camera folder: .MP4 (GoPro files in iPhone folder)

⚠️  Tempate Name - MultiCam Data - Instrument - 2025-07-DD/360 Camera 1 - Name - Instrument - 2025-07-DD
    Issue: Empty camera folder - no video files found
```

---

## Running the System

### 1. Validate Raw Data First

```bash
python3 src/tools/participant_folder_validator.py \
  "/Volumes/X10 Pro 1/Phan Dissertation Data/_Raw_Data_Archive" \
  --export validation_reports/

# Review the reports:
# - Check issues_report_*.csv for problems
# - Fix suspicious files before processing
```

### 2. Convert 360° to 2D Videos

```bash
# (Conversion tool to be implemented)
# Will need to:
# - Select frame angle for each camera
# - Apply same angle to all chunks from same camera
# - Output to Processed_Videos/ folder
```

### 3. Run Integrated Video Processor

```bash
python3 src/integrated_video_processor.py \
  --alignment-dir "Processed_Videos/P01_20250709" \
  --no-duration-limit

# The system will:
# 1. Scan and group chunks by camera
# 2. Align first chunks using audio
# 3. Detect gaps with hybrid method
# 4. Report gaps and suggest actions
# 5. Merge chunks with gap preservation
# 6. Run pose detection
# 7. Generate final output videos
```

### 4. Review Alignment Results

Check the console output for:

```
================================================================================
HYBRID CHUNK ALIGNMENT (Solution 3)
================================================================================
Reference camera: Cam2_X5

📹 Camera 'Cam1_X4': base offset 75.2s (4 chunks)
  ✅ [1] P01_Cam1_chunk007.mp4: 75.2s (audio aligned - FIRST CHUNK)
  ✅ [2] P01_Cam1_chunk008.mp4: 1875.2s (continuous, Δt=0.0s)
  ✅ [3] P01_Cam1_chunk009.mp4: 3675.2s (continuous, Δt=0.0s)
  ⚠️  [4] P01_Cam1_chunk010.mp4: 7212.2s (LARGE GAP: 537s = 8.9min)

============================================================
Step 4: Gap Detection Summary
============================================================

📹 Camera 'Cam1_X4': 1 gap(s) detected
   ⚠️  Before P01_Cam1_chunk010.mp4: 537.0s (8.9min) [timestamp_large_gap]

⚠️  Total gaps detected: 1

🚨 LARGE GAPS (>5 minutes) - Consider splitting sessions:
   Cam1_X4/P01_Cam1_chunk010.mp4: 8.9 minutes
```

---

## Interpreting Results

### Gap Detection Methods

| Method | Meaning | Action |
|--------|---------|--------|
| `audio` | First chunk, aligned by audio | ✅ Trust this alignment |
| `timestamp_continuous` | Timestamp confirms continuous | ✅ Chunks are back-to-back |
| `timestamp_large_gap` | Large timestamp gap (>5min) | ⚠️ Gap preserved, consider splitting |
| `audio_verified_continuous` | Audio confirms continuous despite timestamp gap | ✅ Probably camera clock drift |
| `audio_verified_gap` | Audio confirms gap | ⚠️ True recording break |
| `timestamp_fallback` | Audio check failed, used timestamp | ⚠️ Review manually |
| `assumed_continuous` | No timestamp available | ⚠️ Review manually |

### When to Split Sessions

**✅ Keep as one session if:**
- All chunks are continuous (no gaps)
- Small gaps (<5 min) verified as continuous by audio
- Participant briefly paused but didn't stop recording

**⚠️ Consider splitting if:**
- Large gap (>5 minutes) between chunks
- Gap represents different pieces/songs
- Audio verification confirms distinct recording sessions
- Participant took long break

**❌ Definitely split if:**
- Gap > 30 minutes
- Different recording days
- Different performance contexts

---

## Common Issues

### Issue 1: "No timestamp available - assuming continuous"

**Cause:** Converted video files don't have embedded timestamps
**Impact:** System can't verify gaps automatically
**Solution:** Only process original `.insv` files first, before conversion

### Issue 2: "Audio verification failed"

**Cause:** Silent passages, audio extraction error
**Impact:** Falls back to timestamp-only detection
**Solution:** Review the flagged chunks manually

### Issue 3: X3 camera - missing lens files

**Cause:** X3 produces two files per chunk (`_00_` and `_10_`)
**Impact:** Missing half the 360° data
**Solution:** Ensure both lens files are present before processing

### Issue 4: Privacy issue - participant name in filename

**Example:** `VID_Sarah short_104315_00_004.insv`
**Solution:** Rename to standard format or exclude from processing

### Issue 5: macOS metadata files (`._*`)

**Cause:** macOS creates these when copying files
**Solution:** Safe to delete, use validation tool to identify them

---

## Metadata Tracking Files

### participant_mapping.csv (KEEP PRIVATE!)

```csv
participant_id,real_name,date,instrument,notes
P01,Jennifer,2025-07-09,Violin and Piano,"Camera had restarts"
P02,Bryan,2025-07-16,Piano Dan Tranh and Voice,"3 cameras"
P03,David,2025-07-05,Trombone and Dan Tranh,"Has X3 camera"
```

### conversion_log.csv

```csv
participant_id,camera_id,raw_file,converted_file,frame_angle,fov,conversion_timestamp
P01,Cam1,VID_20250709_191503_00_007.insv,P01_Cam1_chunk007.mp4,180,120,2025-10-20 14:30:00
P01,Cam1,VID_20250709_191503_00_008.insv,P01_Cam1_chunk008.mp4,180,120,2025-10-20 14:32:00
```

### camera_setup.csv

```csv
participant_id,camera_id,model,position,num_chunks,has_gaps
P01,Cam1,X4,Front,4,yes
P01,Cam2,X5,Left side,3,no
P01,Cam3,X5,Right side,3,no
P01,Cam4,iPhone,Overhead,10,no
```

---

## Next Steps

1. **Review validation reports** - Fix any issues found
2. **Delete template folders** - Clean up empty/template folders
3. **Check privacy** - Rename files with participant names
4. **Verify X3 cameras** - Ensure both lens files present
5. **Test on one participant** - Run full pipeline on P01
6. **Review gap detection** - Verify gaps are correctly identified
7. **Process all participants** - Batch process entire dataset

---

## Support

For questions or issues:
1. Check console output for detailed error messages
2. Review validation reports in `validation_reports/`
3. Check gap detection summary for alignment issues
4. Consult this guide for naming conventions and expected behavior

---

**Last Updated:** 2025-10-20
**Version:** 1.0 (Solution 3 - Hybrid Alignment)
