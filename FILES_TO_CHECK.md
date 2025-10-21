# Files to Check - Manual Review Required

Based on the scan of your dissertation data, here are the specific issues found that require your attention before processing.

---

## ✅ FIXED Issues (Completed on 2025-10-21)

### 1. ~~**Empty Template Folders**~~ - ✅ FIXED
- Template folders have been relocated/deleted
- No action needed

### 2. ~~**Privacy Issue - Participant Name in Filename**~~ - ✅ FIXED
- **Old:** `LRV_Sarah short_104315_01_004.lrv`
- **New:** `LRV_20250709_104315_01_004.lrv`
- Renamed on all 3 drives (X10 Pro 1, MAML 26TB 1, MAML 26TB 2)
- Privacy issue RESOLVED

### 3. ~~**Misplaced GoPro Files in iPhone Folder**~~ - ✅ FIXED
- Removed ALL duplicate GoPro files from David's iPhone 12 folder
- Files removed: 17 files (10 LRV + 7 MP4) per drive
- Space freed: ~64.2 GB total across all 3 drives
- iPhone folders now contain only iPhone content (IMG_0101.MOV)

### 4. ~~**Incomplete Folder Names**~~ - ✅ FIXED
- Matt's folder: Already fixed (no longer has template placeholders)

### 5. ~~**Date Incomplete in Folder Name**~~ - ✅ FIXED
- Sarah's iPad folder: Already shows correct date "2025-07-05"

### 6. ~~**macOS Metadata Files (._*)**~~ - ✅ FIXED
- All `._*` files deleted from all 3 drives
- 45+ metadata files removed
- Directories now clean

---

## 🚨 NEW Critical Issues (Requires Action Before Processing)

### 7. **Wangling & Justine Stopby - File Split Required (July 9, 2025)**

**Status:** Files currently duplicated in both folders

**Background:**
- Wangling was recording when Justine stopped by during chunk 003
- Chunk 003 files now exist in BOTH participants' folders (as intended)

**Files Involved:**
- `VID_20250709_132529_00_003.insv` (X5 Camera 2)
- `VID_20250709_132654_00_003.insv` (X5 Camera 1)

**Post-Conversion Action Required:**
1. Convert chunk 003 to 2D first (must convert complete chunks)
2. Review the 2D video to find timestamp when Justine appears
3. Split the video at that point:
   - **First portion:** Keep with Wangling's data
   - **Second portion:** Keep with Justine's data (or both, depending on research needs)

**Decision Needed:**
- [ ] Determine split point after viewing converted video
- [ ] Decide which portion goes to which participant
- [ ] Document the split in processing logs

**Current Status on All Drives:**
- ✅ X10 Pro 1: Wangling has chunks 001, 002, 003
- ✅ MAML 26TB 1: Wangling has chunks 001, 002, 003
- ✅ MAML 26TB 2: Wangling has chunks 001, 002, 003
- ✅ Justine still retains chunk 003 (duplicated)

---

### 8. **Philippe Missing 360 Camera 1 (July 16, 2025)**

**Issue:** Philippe only has 2 cameras but they're numbered Camera 2 and Camera 3

**Current folders:**
- ✅ 360 Camera 2 - Philippe - Drum - 2025-07-16 X5 1
- ✅ 360 Camera 3 - Philippe - Drum - 2025-07-16 X5 2
- ❌ Missing: 360 Camera 1 - Philippe - Drum - 2025-07-16 X4

**Files timestamps:** 13:04:03 - 13:04:29 (correct for Philippe's session)

**Possible explanations:**
1. Philippe only used 2 cameras (X5 cameras), X4 wasn't available
2. There's a missing X4 camera folder somewhere
3. Folder naming issue (should be renumbered to Camera 1 and Camera 2)

**Action Required:**
- [ ] **REMINDER:** Search for missing `360 Camera 1 - Philippe - Drum - 2025-07-16 X4` folder
- [ ] Verify if Philippe actually used 3 cameras or only 2
- [ ] If only 2 cameras used, consider renaming:
  - Camera 2 → Camera 1
  - Camera 3 → Camera 2

---

## ⚠️ Optional Cleanups (Can Skip)

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

- [x] ~~Delete template folders (`Tempate Name` and `Dottie and Emmy shared`)~~ ✅
- [x] ~~Fix Sarah's non-standard filename or document exclusion~~ ✅
- [x] ~~Move David's misplaced GoPro files to correct folder~~ ✅
- [x] ~~Rename Matt's incomplete folder name~~ ✅
- [x] ~~Fix Sarah's iPad folder date~~ ✅

### Stage 2: New Critical Issues

- [ ] **Split Wangling/Justine chunk 003 after conversion to 2D**
- [ ] **Search for Philippe's missing 360 Camera 1 (X4)**

### Stage 3: Optional Cleanups

- [x] ~~Delete all macOS metadata files (`._*`)~~ ✅
- [ ] Decide on `.lrv` files (keep or delete) - Can skip
- [ ] Verify X3 dual-lens files for David, Dottie, Sarah - Can verify during processing

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

---

## 📊 Summary Statistics

**Total Issues Found:** 8
- **Fixed:** 6 ✅
- **Requires Action:** 2 🚨
- **Optional:** 1 ⚠️

**Fixes Applied:**
- Removed duplicate GoPro files: ~64.2 GB freed
- Deleted macOS metadata: 45+ files
- Fixed privacy issue: 1 file renamed
- Duplicated Wangling/Justine chunk 003 files across all drives

**Space Freed:** ~64.2 GB total

---

**Last Updated:** 2025-10-21
**Review Status:** Most Critical Issues Fixed - 2 Remaining Tasks
