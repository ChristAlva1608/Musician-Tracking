# Files to Check - Manual Review Required

Based on the scan of your dissertation data, here are the specific issues found that require your attention before processing.

---

## 🚨 Critical Issues (Fix Before Processing)

### 1. **Empty Template Folders - DELETE**

These folders should be deleted as they're templates with no actual data:

```bash
# DELETE THESE:
rm -rf "/Volumes/X10 Pro 1/Phan Dissertation Data/Tempate Name - MultiCam Data - Instrument - 2025-07-DD"
rm -rf "/Volumes/X10 Pro 1/Phan Dissertation Data/Dottie and Emmy shared with Dottie Folder - MultiCam Data - Violin Piano - 2025-07-05"
```

**Why:** Empty folders will cause processing errors and waste time.

---

### 2. **Privacy Issue - Participant Name in Filename**

**File:** `/Volumes/X10 Pro 1/Phan Dissertation Data/Sarah - MultiCam Data - Piano - 2025-07-05/360 Camera 1 - Sarah - Piano - 2025-07-05 X4/VID_Sarah short_104315_00_004.insv`

**Issue:** Filename contains participant's real name "Sarah" instead of standard Insta360 format

**Action Required:**
```bash
# Option 1: Rename to standard format (if you can determine the correct timestamp)
# Option 2: Exclude this file from processing
# Option 3: Keep but note in conversion_log.csv that this file needs special handling
```

**Dissertation Impact:** If published data includes this filename, participant anonymity is compromised.

---

### 3. **Misplaced GoPro Files in iPhone Folder**

**Location:** `/Volumes/X10 Pro 1/Phan Dissertation Data/David - MultiCam Data - Trombone and Dan Tranh - 2025-07-05/2D Camera 2 - David - Trombone and Dan Tranh - 2025-07-05 iPhone 12 Tuyen Moi/`

**Issue:** This folder is labeled "iPhone 12" but contains GoPro files:
- `GX010303.MP4`
- `GX010304.MP4`
- `GX010305.MP4`

**Action Required:**
```bash
# Option 1: Move GoPro files to correct GoPro folder
mv "/path/to/iPhone 12 Tuyen Moi/GX*.MP4" "/path/to/2D Camera 1 - David.../GoPro 1/"

# Option 2: Rename folder to reflect actual content
mv "2D Camera 2 - David ... iPhone 12 Tuyen Moi" "2D Camera 3 - David ... GoPro Extra"
```

**Impact:** Will confuse camera model detection and processing.

---

### 4. **Incomplete Folder Names**

**Location:** `/Volumes/X10 Pro 1/Phan Dissertation Data/Matt - MultiCam Data - Voice - 2025-07-14/2D Camera 1 - Name - Instrument - 2025-07-14`

**Issue:** Folder still has template placeholders "Name - Instrument" instead of "Matt - Voice"

**Action Required:**
```bash
# Rename to match participant data
mv "2D Camera 1 - Name - Instrument - 2025-07-14" \
   "2D Camera 1 - Matt - Voice - 2025-07-14"
```

---

### 5. **Date Incomplete in Folder Name**

**Location:** `/Volumes/X10 Pro 1/Phan Dissertation Data/Sarah - MultiCam Data - Piano - 2025-07-05/iPad Screen - Sarah - Piano - 2025-07-DD`

**Issue:** Date shows "2025-07-DD" instead of actual date "2025-07-05"

**Action Required:**
```bash
# Fix the date
mv "iPad Screen - Sarah - Piano - 2025-07-DD" \
   "iPad Screen - Sarah - Piano - 2025-07-05"
```

---

## ⚠️ Optional Cleanups (Recommended)

### 6. **macOS Metadata Files (._*)**

These are macOS artifacts created when copying files. Safe to delete:

**Affected participants:** Bryan, Colette, David, Dottie, Jennifer, Justin, Khue, Matt, Nhan, Nicole Collins, Philippe, Quan, Sarah, Sean, Wangling, Yipeng

**Example locations:**
```
Bryan/.../._IMG_3705.MOV
Bryan/.../._RPReplay_Final1752691779.MP4
Colette/.../._WWKI3939.MP4
David/.../._IMG_0163.MOV
... (many more)
```

**Action:**
```bash
# Delete all macOS metadata files
find "/Volumes/X10 Pro 1/Phan Dissertation Data" -name "._*" -type f -delete

# Verify before deleting (dry run):
find "/Volumes/X10 Pro 1/Phan Dissertation Data" -name "._*" -type f
```

**Impact:** These files don't affect processing but clutter the file listing. Safe to delete.

---

### 7. **Low-Resolution Preview Files (.lrv)**

Insta360 cameras create `.lrv` files (Low-Resolution Video) for quick preview.

**Locations:** Many camera folders have these

**Action:**
```bash
# If you don't need preview files, delete them to save space:
find "/Volumes/X10 Pro 1/Phan Dissertation Data" -name "*.lrv" -type f -delete

# Check total size first:
find "/Volumes/X10 Pro 1/Phan Dissertation Data" -name "*.lrv" -type f -exec du -ch {} + | grep total
```

**Impact:**
- **Keep if:** You use them for quick preview on mobile devices
- **Delete if:** You only need final high-res videos (saves ~100GB of space)

---

## 📋 X3 Camera Check (Important!)

### Participants with X3 Cameras

**David, Dottie, Sarah** have X3 cameras which create **dual-lens files**:

**Example from David:**
```
360 Camera 2 - David - Trombone and Dan Tranh - 2025-07-05 X3/
├── VID_20250705_163421_00_021.insv    ← Front lens (00)
├── VID_20250705_163421_10_021.insv    ← Back lens (10)
├── VID_20250705_163421_00_022.insv    ← Front lens
├── VID_20250705_163421_10_022.insv    ← Back lens
...
```

**Action Required:**
For each chunk number, verify BOTH files exist:

```bash
# Check David's X3 camera
cd "/Volumes/X10 Pro 1/Phan Dissertation Data/David - MultiCam Data - Trombone and Dan Tranh - 2025-07-05/360 Camera 2 - David - Trombone and Dan Tranh - 2025-07-05 X3/"

# List unique chunk numbers
ls VID_*.insv | sed 's/.*_\([0-9]*\)\.insv/\1/' | sort -u

# For each chunk, verify both 00 and 10 files exist
# Example: chunk 021 should have:
# - VID_20250705_163421_00_021.insv (front)
# - VID_20250705_163421_10_021.insv (back)
```

**If missing:**
- Missing `_00_` files: Front lens data lost
- Missing `_10_` files: Back lens data lost
- **Impact:** You'll only have 180° coverage instead of 360°

---

## 📊 Pre-Processing Checklist

Use this checklist before running the main processing pipeline:

### Stage 1: Critical Fixes

- [ ] Delete template folders (`Tempate Name` and `Dottie and Emmy shared`)
- [ ] Fix Sarah's non-standard filename or document exclusion
- [ ] Move David's misplaced GoPro files to correct folder
- [ ] Rename Matt's incomplete folder name
- [ ] Fix Sarah's iPad folder date

### Stage 2: Optional Cleanups

- [ ] Delete all macOS metadata files (`._*`)
- [ ] Decide on `.lrv` files (keep or delete)
- [ ] Verify X3 dual-lens files for David, Dottie, Sarah

### Stage 3: Validation

- [ ] Run validator tool:
  ```bash
  python3 src/tools/participant_folder_validator.py \
    "/Volumes/X10 Pro 1/Phan Dissertation Data" \
    --export validation_reports/
  ```

- [ ] Review generated reports:
  - [ ] `issues_report_*.csv` - Should show reduced issues after fixes
  - [ ] `file_inventory_*.csv` - Verify all expected files present
  - [ ] `statistics_*.csv` - Numbers make sense

### Stage 4: Create Mapping Files

- [ ] Create `participant_mapping.csv`:
  ```csv
  participant_id,real_name,date,instrument,notes
  P01,Jennifer,2025-07-09,Violin and Piano,"4 chunks on Cam1, chunk 10 has gap"
  P02,Bryan,2025-07-16,Piano Dan Tranh and Voice,"3 cameras"
  P03,David,2025-07-05,Trombone and Dan Tranh,"Has X3 camera, verify dual-lens"
  P04,Dottie,2025-07-05,Violin and Piano,"Has X3 camera"
  P04_02,Sarah,2025-07-05,Piano,"Has X3 camera, non-standard filename issue"
  ... (continue for all 23 participants)
  ```

- [ ] Document camera positions per participant
- [ ] Note any special issues or gaps

---

## 🎯 Verification After Fixes

After making changes, re-run the validator:

```bash
python3 src/tools/participant_folder_validator.py \
  "/Volumes/X10 Pro 1/Phan Dissertation Data" \
  --export validation_reports_after_fixes/
```

**Expected results:**
- Suspicious files: Should decrease from 15 to ~0-5
- Participants with issues: Should decrease from 3 to 0
- Empty template folders: Should be gone

---

## 📝 Documentation of Fixes

Keep a log of what you fixed:

**File:** `data_cleanup_log.txt`

```
Date: 2025-10-20

1. Deleted empty template folders:
   - Tempate Name - MultiCam Data - Instrument - 2025-07-DD
   - Dottie and Emmy shared with Dottie Folder - MultiCam Data - Violin Piano - 2025-07-05

2. Renamed folders:
   - Matt/.../2D Camera 1 - Name - Instrument → 2D Camera 1 - Matt - Voice
   - Sarah/.../iPad Screen - Sarah - Piano - 2025-07-DD → iPad Screen - Sarah - Piano - 2025-07-05

3. Moved misplaced files:
   - David/.../iPhone 12 Tuyen Moi/GX*.MP4 → David/.../GoPro 1/

4. Deleted macOS metadata:
   - Removed all ._* files (45 files total, ~500KB)

5. Sarah's non-standard filename:
   - VID_Sarah short_104315_00_004.insv
   - Decision: [Document your decision here]
   - Action taken: [Document what you did]

6. Low-resolution files:
   - Decision: [Kept/Deleted]
   - If deleted: Saved ~XXX GB of space

7. X3 camera verification:
   - David: All dual-lens pairs verified ✅
   - Dottie: All dual-lens pairs verified ✅
   - Sarah: All dual-lens pairs verified ✅
```

---

## 🆘 If You Need Help

For each issue, you can:

1. **Check the file directly** - Verify the issue exists
2. **Review HYBRID_ALIGNMENT_GUIDE.md** - See if there's guidance
3. **Run validator again** - Get updated status
4. **Document your decision** - Keep track of what you did

**Safety First:**
- ✅ **Always backup before deleting**
- ✅ **Test on one participant first**
- ✅ **Keep original raw data untouched**
- ✅ **Document all changes**

---

**Last Updated:** 2025-10-20
**Review Status:** Pending Manual Review
