# Naming Convention Update Summary

## 📋 What Changed

Based on your clarifying questions, the naming convention has been updated for consistency across all camera types.

---

## ✅ Updated Naming Convention

### **ALL cameras now use `chunk###` format in processed files**

| Camera Type | Raw Format | Converted Format | Example |
|-------------|------------|------------------|---------|
| **X4/X5 (360°)** | `VID_YYYYMMDD_HHMMSS_00_CCC.insv` | `P##_Cam#_chunk###.mp4` | `P01_Cam1_chunk007.mp4` |
| **X3 (360° dual)** | `VID_..._00_CCC.insv` + `VID_..._10_CCC.insv` | `P##_Cam#_chunk###.mp4` | `P03_Cam2_chunk021.mp4` ✅ |
| **iPhone (2D)** | `IMG_####.MOV` | `P##_Cam#_chunk###.mp4` | `P01_Cam4_chunk001.mp4` ✅ |
| **GoPro (2D)** | `GX######.MP4` | `P##_Cam#_chunk###.mp4` | `P03_Cam3_chunk001.mp4` ✅ |

---

## 📖 Understanding the Format

### **Raw 360° Filenames: `VID_YYYYMMDD_HHMMSS_LL_CCC.insv`**

#### **LL = Lens Identifier (2 digits)**

| Value | Meaning | Used By |
|-------|---------|---------|
| **00** | Front/only lens | **X4, X5** (single file)<br>**X3** (front hemisphere) |
| **10** | Back lens | **X3 ONLY** (back hemisphere) |

**Examples:**
```
VID_20250709_191503_00_007.insv    ← X4/X5: Single file, LL=00
VID_20250705_163421_00_021.insv    ← X3: Front hemisphere, LL=00
VID_20250705_163421_10_021.insv    ← X3: Back hemisphere, LL=10 (SAME chunk!)
```

#### **CCC = Chunk Number (3 digits)**

- **Camera-assigned** sequential number
- **Not per-session** - continues across camera's lifetime
- May start at `007`, `021`, etc. (not always `001`)
- **Same chunk number** on X3 `_00_` and `_10_` files = same moment in time

---

## 🔧 Key Changes from Previous Version

### **Change 1: X3 Camera Handling**

**Before:**
```
❌ P03_Cam2Front_chunk021.mp4    # Separate front view
❌ P03_Cam2Back_chunk021.mp4     # Separate back view
```

**After:**
```
✅ P03_Cam2_chunk021.mp4          # Stitched FIRST, then converted
```

**Why:** Stitch `_00_` + `_10_` files into full 360° video BEFORE converting to 2D
- Same workflow as X4/X5
- Full 360° coverage for angle selection
- Consistent processing

### **Change 2: 2D Camera Naming**

**Before:**
```
❌ P01_Cam4_IMG3571.mp4           # Kept original iPhone name
```

**After:**
```
✅ P01_Cam4_chunk001.mp4          # Sequential chunk number
✅ P01_Cam4_chunk002.mp4
✅ P01_Cam4_chunk003.mp4
```

**Why:**
- ✅ Consistent with 360° cameras
- ✅ Aligns with timeline processing
- ✅ Easier to manage and process

**Mapping:**
```
IMG_3571.MOV (earliest) → P01_Cam4_chunk001.mp4
IMG_3572.MOV            → P01_Cam4_chunk002.mp4
IMG_3573.MOV (latest)   → P01_Cam4_chunk003.mp4
```

---

## 📂 Updated Folder Structure

```
Processed_Videos/
├── P01_20250709/                    # Jennifer
│   ├── Cam1_X4/                     # 360° camera
│   │   ├── P01_Cam1_chunk007.mp4    # From VID_007.insv
│   │   ├── P01_Cam1_chunk008.mp4
│   │   └── P01_Cam1_chunk010.mp4    # Gap before this
│   │
│   ├── Cam2_X5/                     # 360° camera
│   │   └── ...
│   │
│   └── Cam4_iPhone/                 # 2D camera ✅ NOW USES chunk###
│       ├── P01_Cam4_chunk001.mp4    # From IMG_3571.MOV
│       ├── P01_Cam4_chunk002.mp4    # From IMG_3572.MOV
│       └── P01_Cam4_chunk003.mp4    # From IMG_3573.MOV
│
└── P03_20250705/                    # David (has X3)
    ├── Cam1_X4/
    │
    ├── Cam2_X3/                     # X3 camera ✅ STITCHED FIRST
    │   ├── P03_Cam2_chunk021.mp4    # Stitched from _00_021 + _10_021
    │   └── P03_Cam2_chunk022.mp4    # Stitched from _00_022 + _10_022
    │
    └── Cam3_GoPro/                  # 2D camera ✅ NOW USES chunk###
        ├── P03_Cam3_chunk001.mp4    # From GX010303.MP4
        └── P03_Cam3_chunk002.mp4    # From GX010304.MP4
```

---

## 🔄 Updated Conversion Workflow

### **X4/X5 Cameras (Unchanged)**

```
VID_20250709_191503_00_007.insv
    ↓ Select angle (e.g., 180°)
    ↓ Extract 2D frame
P01_Cam1_chunk007.mp4
```

### **X3 Cameras (CHANGED - Stitch First!)**

```
VID_20250705_163421_00_021.insv (front, LL=00)
VID_20250705_163421_10_021.insv (back, LL=10)
    ↓ Stitch using Insta360 Studio
VID_021_stitched.mp4 (full 360°)
    ↓ Select angle (e.g., 180°)
    ↓ Extract 2D frame
P03_Cam2_chunk021.mp4  ✅ Single output file
```

### **iPhone/GoPro (CHANGED - Sequential Chunks)**

```
IMG_3571.MOV (file 1)
    ↓ Copy/transcode
P01_Cam4_chunk001.mp4  ✅ chunk001

IMG_3572.MOV (file 2)
    ↓ Copy/transcode
P01_Cam4_chunk002.mp4  ✅ chunk002

Mapping = chronological order → sequential chunk numbers
```

---

## 📊 Benefits of Updated Convention

### **1. Consistency**
- ✅ ALL cameras use same `chunk###` format
- ✅ System processes all cameras uniformly
- ✅ Easier to understand and maintain

### **2. Timeline Alignment**
- ✅ 2D camera chunks align with 360° timeline
- ✅ Hybrid alignment works across all camera types
- ✅ Gap detection applies to all cameras

### **3. Workflow Simplification**
- ✅ X3: Same workflow as X4/X5 after stitching
- ✅ 2D: Same chunk-based processing as 360°
- ✅ One conversion pipeline for all

### **4. Data Management**
- ✅ Predictable file names
- ✅ Easy to verify chunk sequences
- ✅ Consistent sorting and organization

---

## 📝 Updated Mapping File

**conversion_log.csv:**
```csv
participant_id,camera_id,raw_file,converted_file,frame_angle,notes
P01,Cam1,VID_20250709_191503_00_007.insv,P01_Cam1_chunk007.mp4,180,X4 camera
P01,Cam1,VID_20250709_191503_00_008.insv,P01_Cam1_chunk008.mp4,180,X4 camera
P01,Cam4,IMG_3571.MOV,P01_Cam4_chunk001.mp4,N/A,iPhone - mapped to chunk 001
P01,Cam4,IMG_3572.MOV,P01_Cam4_chunk002.mp4,N/A,iPhone - mapped to chunk 002
P03,Cam2,VID_00_021.insv+VID_10_021.insv,P03_Cam2_chunk021.mp4,180,X3 stitched
P03,Cam3,GX010303.MP4,P03_Cam3_chunk001.mp4,N/A,GoPro - mapped to chunk 001
```

---

## 📚 Updated Documentation Files

All documentation has been updated with the new convention:

✅ **HYBRID_ALIGNMENT_GUIDE.md**
- Added LL and CCC explanations
- Added X3 stitching workflow
- Updated all examples with 2D chunk naming
- Added "Conversion Workflow" section

✅ **QUICK_START_GUIDE.md**
- Updated file naming quick reference
- Added LL/CCC explanations
- Updated folder structure examples
- Updated all naming examples

✅ **IMPLEMENTATION_SUMMARY.md**
- Added naming convention summary table
- Updated technical details

✅ **participant_folder_validator.py**
- Updated header documentation
- Updated function docstrings
- Added X3 dual-file notes

---

## ✅ Action Items

### **For You:**

1. **Review the updated naming convention**
   - Read this document
   - Check HYBRID_ALIGNMENT_GUIDE.md for full details

2. **Understand X3 workflow**
   - X3 cameras need stitching BEFORE conversion
   - Use Insta360 Studio to stitch `_00_` + `_10_` files

3. **Plan 2D camera conversion**
   - iPhone/GoPro files get sequential chunk numbers
   - Map files in chronological order

4. **Update conversion tool** (when implementing)
   - Map 2D files to chunk numbers
   - Handle X3 stitching step
   - Maintain conversion_log.csv

### **System is Ready:**

✅ Hybrid alignment algorithm implemented
✅ File validator ready to use
✅ Documentation updated
✅ Naming convention clarified

**Next:** Test with one participant's data!

---

## 🆘 Questions Answered

### Q1: What is LL?
**A:** Lens identifier
- `00` = front/only lens (X4, X5, X3 front)
- `10` = back lens (X3 only)

### Q2: What is CCC?
**A:** Chunk number assigned by camera
- Camera's lifetime counter (not per-session)
- May start at 007, 021, etc.
- Same number on X3 `_00_` and `_10_` files

### Q3: Why doesn't X3 become 1 file like X5?
**A:** X3 doesn't stitch internally
- X4/X5: Camera stitches → outputs 1 file
- X3: No stitching → outputs 2 files (00 + 10)
- **Solution:** Stitch first using Insta360 Studio

### Q4: Why chunk naming for 2D cameras?
**A:** Consistency and timeline alignment
- Same format across all cameras
- Works with hybrid alignment system
- Easier processing and management

---

**Last Updated:** 2025-10-20
**Version:** 2.0 (Updated Convention)
**Status:** ✅ Complete - Ready for Testing
