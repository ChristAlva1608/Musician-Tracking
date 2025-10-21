# 360° to 2D Conversion - Semi-Automated Workflow

## Participant: Quan - Violin (2025-07-25)

---

## STEP 1: Insta360 Studio Export (Manual)

### Camera 1 (X5 1) - Export Settings

**Input Files:**
- `VID_20250725_142824_00_020.insv` (19 GB, ~28.5 min)
- `VID_20250725_142824_00_021.insv` (16 GB, ~24 min)

**Insta360 Studio Process:**

### EXPORT EACH CHUNK SEPARATELY (Recommended to preserve gaps)

Export chunks **one at a time** to maintain temporal accuracy and preserve any gaps between recordings.

---

### **Export Chunk 020:**

1. **Open Insta360 Studio**

2. **Import first chunk:**
   - File → Import → `VID_20250725_142824_00_020.insv`

3. **Preview and Select View Angle:**
   - Scrub through timeline to check if camera moved
   - Choose your preferred angle:
     - Front view (facing musician)
     - Side view (left/right)
     - Overhead
     - Custom angle
   - **IMPORTANT:** Write down your chosen angle for consistency

4. **Export Settings - CRITICAL:**

   **Basic Settings:**
   - **Resolution:** 3840×2160 (4K) or 1920×1080 (1080p)
   - **Frame Rate:** 30fps (or match original - check with right-click → Properties)
   - **Codec:** H.265 (HEVC) for smaller files, or H.264 for compatibility
   - **Bitrate:** High (80-100 Mbps for 4K, 30-50 Mbps for 1080p)
   - **Format:** MP4

   **Advanced Settings (Look for "Advanced" or "More Settings" button):**
   - ✅ **Enable Dewarp** - Removes fisheye distortion, makes straight lines straight
   - ✅ **Direction Lock** - Keeps view stable even if camera physically moved
   - ❌ **Optical Flow Stabilization** - OFF (can cause artifacts)
   - **Viewing Angle:** [Your chosen angle - write it down!]

   **Output Folder:**
   ```
   /Volumes/X10 Pro 1/Phan Dissertation Data/Quan - MultiCam Data - Violin - 2025-07-25/2D Converted - 360 Camera 1 - Quan - Violin - 2025-07-25 X5 1/
   ```

5. **Start Export** - This will take 10-30 minutes depending on your computer

6. **After export finishes, rename immediately:**
   ```bash
   cd "/Users/ywcm/Downloads/Musician Tracking V3"
   ./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1" "020"
   ```

---

### **Export Chunk 021:**

7. **Import second chunk:**
   - File → Import → `VID_20250725_142824_00_021.insv`

8. **Use EXACT SAME settings as chunk 020:**
   - Same resolution
   - Same frame rate
   - Same codec
   - **Same viewing angle** (this is critical!)
   - Same dewarp setting
   - Same direction lock

9. **Export to same folder**

10. **After export finishes, rename:**
    ```bash
    ./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1" "021"
    ```

---

## STEP 2: Automated Concatenation (Script)

After Insta360 Studio finishes exporting all chunks, run the concatenation script.

**Script Location:**
```bash
/Users/ywcm/Downloads/Musician Tracking V3/concat_360_chunks.sh
```

**Usage:**
```bash
./concat_360_chunks.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1"
```

**What the script does:**
1. Finds all exported MP4 chunks in the output folder
2. Concatenates them in order using ffmpeg
3. Names the final file according to convention:
   ```
   2D_from_360_Camera_1-Quan-Violin-20250725-X5_1.mp4
   ```
4. Moves individual chunks to a `_chunks_backup/` subfolder
5. Verifies the final video duration and file integrity

---

## STEP 3: Repeat for Other Cameras

### Camera 2 (X5 2)
**Input Files:**
- `VID_20250725_142811_00_018.insv`
- `VID_20250725_142811_00_019.insv`

**Output Folder:**
```
/Volumes/X10 Pro 1/Phan Dissertation Data/Quan - MultiCam Data - Violin - 2025-07-25/2D Converted - 360 Camera 2 - Quan - Violin - 2025-07-25 X5 2/
```

**Script Command:**
```bash
./concat_360_chunks.sh "Quan" "Violin" "2025-07-25" "Camera 2" "X5 2"
```

### Camera 3 (X5 3)
**Input Files:**
- `VID_20250725_142833_00_007.insv`
- `VID_20250725_142833_00_008.insv`

**Output Folder:**
```
/Volumes/X10 Pro 1/Phan Dissertation Data/Quan - MultiCam Data - Violin - 2025-07-25/2D Converted - 360 Camera 3 - Quan - Violin - 2025-07-25 X5 3/
```

**Script Command:**
```bash
./concat_360_chunks.sh "Quan" "Violin" "2025-07-25" "Camera 3" "X5 3"
```

---

## Final Output Files

### Option A: Keep Chunks Separate (If gaps exist between recordings)

```
2D Converted - 360 Camera 1 - Quan - Violin - 2025-07-25 X5 1/
  ├── 2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_020.mp4 (~28.5 min)
  └── 2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_021.mp4 (~24 min)

2D Converted - 360 Camera 2 - Quan - Violin - 2025-07-25 X5 2/
  ├── 2D_from_360_Camera_2-Quan-Violin-20250725-X5_2-chunk_018.mp4 (~28.5 min)
  └── 2D_from_360_Camera_2-Quan-Violin-20250725-X5_2-chunk_019.mp4 (~24 min)

2D Converted - 360 Camera 3 - Quan - Violin - 2025-07-25 X5 3/
  ├── 2D_from_360_Camera_3-Quan-Violin-20250725-X5_3-chunk_007.mp4 (~30 min)
  └── 2D_from_360_Camera_3-Quan-Violin-20250725-X5_3-chunk_008.mp4 (~25.5 min)
```

**Align later in video editor** with exact timestamps preserved.

---

### Option B: Concatenated (If chunks are continuous with NO gaps)

After running concatenation script, you get:

```
2D Converted - 360 Camera 1 - Quan - Violin - 2025-07-25 X5 1/
  └── 2D_from_360_Camera_1-Quan-Violin-20250725-X5_1.mp4 (~52 min)
  └── _chunks_backup/
      ├── 2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_020.mp4
      └── 2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_021.mp4
```

---

## Tips for Consistent Conversion Across Participants

1. **Document your chosen angle** for each camera position
2. **Use the same export settings** for all participants
3. **Keep notes** on which camera angle works best for each instrument
4. **Verify duration** matches the calculated time from file sizes
5. **Check sync** between multiple cameras using audio waveforms

---

## Troubleshooting

**If chunks don't concatenate smoothly:**
- Ensure all chunks have identical resolution, frame rate, codec
- Check that Insta360 Studio exported all chunks with same settings
- Use the script's lossless mode: add `--lossless` flag

**If export takes too long:**
- Consider exporting at 1080p instead of 4K
- Use H.264 instead of H.265 (faster encoding)
- Close other applications to free up CPU/GPU

**If file sizes are huge:**
- Use H.265 codec (better compression)
- Reduce bitrate slightly (but not below 30 Mbps for 4K)
- Consider 1080p if 4K is not required for analysis
