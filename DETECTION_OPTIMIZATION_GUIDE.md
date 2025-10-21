# Detection Optimization Guide for 360° to 2D Conversion

## Goal: Maximize Human Detection Accuracy for Computer Vision

---

## 1. Subject Size Optimization

### **Rule: Subject should occupy 30-60% of frame height**

**Why this matters:**
- Detection models (YOLO, MediaPipe, OpenPose) work best when subject is **200-800 pixels tall**
- Too small: model misses details (hands, fingers, bow position)
- Too large: model loses context (can't see full body movement)

### **How to Achieve in Insta360 Studio:**

#### **Field of View (FOV) Settings:**

When choosing your viewing angle in Insta360 Studio:

1. **Narrow FOV (Recommended for seated musicians)**
   - FOV: 90-110 degrees
   - Subject fills more of frame
   - Better detail for hand/finger tracking
   - **Best for:** Piano, violin, guitar, seated performers

2. **Medium FOV (Good for standing/moving musicians)**
   - FOV: 110-130 degrees
   - Shows more body movement
   - Still maintains good detail
   - **Best for:** Drums, standing performers, larger movements

3. **Wide FOV (Use sparingly)**
   - FOV: 130-150+ degrees
   - Subject is smaller
   - Shows full context but loses detail
   - **Only if:** Need to see multiple people or full stage

#### **Where to Set FOV in Insta360 Studio:**

```
Export Settings Panel:
┌─────────────────────────────────┐
│ Resolution: [4K ▼]              │
│ Frame Rate: [30fps ▼]           │
│                                 │
│ ⚙️ Advanced ▼                   │
│   ✅ Enable Dewarp              │
│   ✅ Direction Lock             │
│   🎯 Viewing Angle: [Custom]    │
│   📐 Field of View: [100° ▼]   │  ← ADJUST THIS!
│   🔍 Zoom: [1.0-2.0x]           │  ← OR USE ZOOM
│                                 │
└─────────────────────────────────┘
```

**If FOV slider not available:**
- Use the **preview window** to zoom in/out before export
- Drag to adjust framing
- Some versions use "Zoom" instead of FOV (inverse relationship)

---

## 2. Optimal Camera Angles for Detection

### **For Quan (Violin):**

**Recommended Primary Angle: Front-Side (45° angle)**
- **Why:** Captures bow arm movement + finger positions + body posture
- **FOV:** 90-100°
- **Subject size:** Musician should fill ~50% of frame height

**Alternative Angle: Pure Front**
- **Why:** Best for facial expressions + body symmetry
- **FOV:** 100-110°
- **Limitation:** Loses bow depth perception

**Avoid: Pure Side**
- Bow movement is compressed (depth issue)
- One arm hidden behind body

### **Camera Placement Priority:**

**Camera 1 (Primary Detection Camera):**
- Front-side angle, narrow FOV
- Subject largest on screen
- **Use this for main tracking**

**Camera 2 (Context Camera):**
- Different angle (e.g., overhead or opposite side)
- Slightly wider FOV
- Captures movement you might miss in Camera 1

**Camera 3 (Backup/Wide Shot):**
- Wide angle showing full scene
- Can be lower priority for detection

---

## 3. Export Settings for Maximum Detection Accuracy

### **Resolution:**

| Resolution | Pros | Cons | Recommended For |
|------------|------|------|-----------------|
| **4K (3840×2160)** | Best detail, tracks fingers/bow accurately | Large files, slower processing | Primary detection camera |
| **1080p (1920×1080)** | Faster processing, smaller files | May miss fine details | Secondary cameras, quick tests |
| **2.7K (2704×1520)** | Good balance | Non-standard, some tools don't support | Alternative middle ground |

**Recommendation for Quan:**
- **Camera 1:** 4K (primary tracking)
- **Camera 2-3:** 1080p (context/backup)

---

### **Frame Rate:**

| Frame Rate | Pros | Cons | Recommended For |
|------------|------|------|-----------------|
| **30 fps** | Standard, compatible, smaller files | May miss very fast movements | Most instruments, standard analysis |
| **60 fps** | Captures fast bow movements, smooth | 2x larger files, slower processing | Violin, drums, rapid movements |
| **25 fps** | Cinematic look | Not standard for analysis | Avoid for detection work |

**Recommendation for Quan (Violin):**
- **30 fps** is sufficient for bow tracking
- Use **60 fps** only if you need ultra-precise bow motion analysis

---

### **Codec and Bitrate:**

**For Detection Work:**

```
Codec: H.264 (NOT H.265)
  ↑ Why: Better compatibility with OpenCV, MediaPipe

Bitrate: 50-80 Mbps (4K) or 25-40 Mbps (1080p)
  ↑ Why: Preserves edge detail for skeleton detection

Color Space: Rec. 709 (Standard)
  ↑ Why: Expected by most CV models
```

**Important:** While H.265 creates smaller files, some detection tools struggle with it. Use H.264 for detection cameras.

---

## 4. Lighting and Visual Optimization

### **Settings to Enable:**

- ✅ **Dewarp:** ON (straight lines = better pose estimation)
- ✅ **Direction Lock:** ON (stable horizon = consistent skeleton tracking)
- ❌ **HDR/Tone Mapping:** OFF (can create motion artifacts)
- ❌ **Color Grading:** OFF (neutral colors = better detection)
- ❌ **Vignette/Filters:** OFF (even lighting across frame)

### **Exposure Check:**

Before exporting, check preview for:
- Subject not too dark (underexposed)
- Subject not blown out (overexposed)
- High contrast between subject and background helps detection

If footage has exposure issues, some Insta360 versions allow:
- Brightness adjustment: Keep at 0 or +5 to +10 max
- Contrast: Keep at 0 (don't enhance)
- Sharpness: 0 to +10 (slight sharpen helps edge detection)

---

## 5. Framing Guidelines for Each Instrument

### **Violin (Quan):**

```
Ideal Framing (Front-Side Angle):
┌─────────────────────────────────┐
│         [ceiling/wall]          │  ← Small headroom (10-15% of frame)
│                                 │
│         👤 Musician             │  ← Head at ~80% from bottom
│        🎻 Violin + Bow          │  ← Instrument fully visible
│      (full upper body)          │  ← Torso ~50% of frame height
│                                 │
│    [chair/floor visible]        │  ← Small floor space (10-15%)
└─────────────────────────────────┘

Key Points to Capture:
✅ Full bow movement (left to right)
✅ Both hands (left hand fingers + right hand bow grip)
✅ Shoulder/elbow positioning
✅ Head/neck posture
✅ Torso movement
❌ Don't need: Full legs (unless standing), floor details
```

**Zoom Level Test:**
1. Pause on a frame where musician is playing
2. Check: Can you clearly see individual fingers on fingerboard?
3. Check: Can you see bow contact point on string?
4. If NO → zoom in more (narrower FOV)
5. If bow/hands go out of frame → zoom out slightly

---

## 6. Multi-Camera Strategy for Detection

### **Recommended Setup for Quan:**

| Camera | Angle | FOV | Resolution | Purpose |
|--------|-------|-----|------------|---------|
| **Camera 1** | Front-Side (45°) | 90-100° | 4K | Primary tracking: bow arm, fingers, posture |
| **Camera 2** | Overhead (looking down) | 100-110° | 1080p | Bow trajectory, violin position |
| **Camera 3** | Front (0°) | 100-110° | 1080p | Facial expression, body symmetry |

**Export Priority:**
1. Export Camera 1 first (primary)
2. Test detection on Camera 1 before exporting others
3. If Camera 1 works well, use same FOV for Camera 2-3

---

## 7. Quality Control Checklist

### **Before Exporting All Chunks:**

Export just **first 30 seconds** of chunk 020 as a test:

- [ ] Load test video in your detection software
- [ ] Run detection on 5-10 frames
- [ ] Check: Subject detected in >95% of frames?
- [ ] Check: Keypoints (hands, elbows, shoulders) tracked accurately?
- [ ] Check: Bow visible and trackable?

**If detection fails:**
- Subject too small → Narrow FOV / zoom in
- Subject too blurry → Increase bitrate or use 4K
- Pose estimation incorrect → Enable dewarp
- Tracking jumpy → Enable direction lock

**If detection succeeds:**
- ✅ Use exact same settings for all remaining chunks
- ✅ Document FOV and zoom level for other participants

---

## 8. Insta360 Studio Export Workflow (Detection-Optimized)

### **Step-by-Step for Quan Camera 1:**

1. **Import:** `VID_20250725_142824_00_020.insv`

2. **Frame the shot:**
   - Drag preview to find best angle (front-side 45°)
   - Adjust FOV slider: 90-100°
   - **Test:** Musician fills ~50% of frame height
   - **Test:** Can you see fingers clearly in preview?

3. **Lock the view:**
   - Enable "Direction Lock"
   - Scrub timeline to verify view stays stable

4. **Export Settings:**
   ```
   Resolution: 3840×2160 (4K)
   Frame Rate: 30 fps
   Codec: H.264
   Bitrate: 60-80 Mbps
   Format: MP4

   Advanced:
   ✅ Enable Dewarp
   ✅ Direction Lock
   ❌ HDR
   ❌ Filters
   FOV: 95° (adjust to taste)
   ```

5. **Export 30-second test clip first**

6. **Verify detection works**

7. **If good → export full chunk with same settings**

---

## 9. Detection Model Considerations

### **Expected Performance:**

**Good Framing (subject 40-60% of frame):**
- YOLO detection: 98-99% detection rate
- MediaPipe Pose: All 33 keypoints tracked
- Hand tracking: Fingers visible and trackable

**Poor Framing (subject <20% of frame):**
- YOLO detection: 60-80% (misses frames)
- MediaPipe Pose: Missing keypoints (hands, feet)
- Hand tracking: Fails entirely

### **Recommended Detection Models:**

**For musician tracking:**
1. **YOLOv8-pose** - Fast, accurate full body tracking
2. **MediaPipe Pose** - Good for detailed joint tracking
3. **MMPose** - Best for fine-grained instrument-specific tracking

**All models prefer:**
- Subject 200-600 pixels tall (in 1080p) or 400-1200 pixels (in 4K)
- High contrast
- Stable camera (direction lock)
- No fisheye distortion (dewarp enabled)

---

## 10. File Naming Convention (Updated)

When you have multiple camera angles with different FOV settings, include it in notes:

```
2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_020.mp4
  ↑ This is your PRIMARY tracking camera

README.txt or notes:
Camera 1: Front-side 45°, FOV 95°, 4K, H.264 - PRIMARY DETECTION
Camera 2: Overhead, FOV 110°, 1080p, H.264 - Bow trajectory
Camera 3: Front 0°, FOV 105°, 1080p, H.264 - Facial/body
```

---

## 11. Quality Check: Verify Exported File Meets Requirements

### **Automated Quality Check Script**

After exporting, verify your file meets detection requirements:

```bash
cd "/Users/ywcm/Downloads/Musician Tracking V3"
./verify_export_quality.sh "/path/to/your/exported_file.mp4"
```

**The script checks:**
- ✅ Resolution (1080p minimum, 4K preferred)
- ✅ Codec (H.264 or H.265)
- ✅ Frame rate (30fps recommended)
- ✅ Bitrate (sufficient for detection)
- ⚠️ File integrity (no corruption)

### **Manual Visual Check:**

Load the exported video and check random frames:

```
Frame Check at: 00:30, 05:00, 10:00, 15:00 seconds

For each frame:
- [ ] Subject clearly visible (not blurry)
- [ ] Fingers/hands have crisp edges
- [ ] No fisheye distortion (straight lines are straight)
- [ ] Horizon is level (not tilted)
- [ ] Subject fills 40-60% of frame height
- [ ] No banding or compression artifacts
```

**If ANY check fails → Re-export with corrected settings**

---

## 12. Reframing: Handling Frame Jumps Between Chunks

### **The Problem:**

If you export chunk 020 with one viewing angle/FOV, then export chunk 021 with a different angle/FOV, you'll get a **sudden jump** when concatenating:

```
Chunk 020: [Subject centered, zoomed in]
           ↓
[JUMP!]    ← Frame suddenly shifts/zooms
           ↓
Chunk 021: [Subject off-center, zoomed out]
```

### **Is Frame Jumping OK?**

**Short answer:** ⚠️ **It depends on your detection workflow**

#### **Frame Jump is ACCEPTABLE if:**

✅ **You're keeping chunks separate** (not concatenating)
- Each chunk analyzed independently
- Detection runs on each chunk separately
- No cross-chunk tracking needed
- **Example:** Analyze each chunk's pose data separately, merge later in post-processing

✅ **Chunks have recording gaps** (camera was stopped/restarted)
- Jump aligns with actual temporal gap
- Natural break in the data
- **Example:** Chunk 020 = practice session, gap, Chunk 021 = performance

✅ **You're only using one chunk for detection**
- Other chunks are backup/reference only
- **Example:** Chunk 020 has best angle, use only that one

#### **Frame Jump is PROBLEMATIC if:**

❌ **Concatenating for continuous detection**
- Tracking IDs get confused at jump point
- Kalman filters break at discontinuity
- Trajectory analysis has artificial spike
- **Solution:** Must use SAME angle/FOV for all chunks

❌ **Cross-chunk temporal analysis**
- Comparing movement patterns across time
- Need smooth transitions
- **Solution:** Must use SAME angle/FOV for all chunks

❌ **Video will be shown to humans**
- Jarring visual experience
- Looks unprofessional
- **Solution:** Use SAME angle/FOV for all chunks

---

### **How to Avoid Frame Jumps:**

#### **Method 1: Lock Settings for All Chunks (Recommended)**

**When exporting chunk 020:**
1. Choose your angle and FOV
2. **Write down the exact settings:**
   ```
   Camera 1, Chunk 020:
   - Viewing Angle: Front-side 45° (azimuth: 45°, elevation: 0°)
   - FOV: 95°
   - Zoom: 1.0x
   - Direction Lock: ON
   - Dewarp: ON
   ```

3. **Export chunk 020**

4. **For chunk 021:** Use IDENTICAL settings
   - Import chunk 021
   - Set viewing angle to SAME position (45°, 0°)
   - Set FOV to SAME value (95°)
   - Same zoom (1.0x)
   - Export

**Result:** Seamless transition, no jump

---

#### **Method 2: Use Insta360 Studio's "Reframe" Feature**

Some versions of Insta360 Studio allow you to:

1. **Import all chunks from same session**
2. **Set keyframes on first chunk** (define your viewing angle)
3. **Copy reframe settings to other chunks**
4. Export all with identical framing

**How to do this:**
```
Insta360 Studio Pro (if available):
1. Import VID_20250725_142824_00_020.insv and 021.insv
2. Click on chunk 020 → Set viewing angle
3. Right-click on timeline → "Copy Reframe"
4. Click on chunk 021 → Right-click → "Paste Reframe"
5. Verify both chunks have identical framing
6. Batch export
```

**Note:** Not all Insta360 Studio versions have this feature. Check your version.

---

#### **Method 3: Save Viewing Angle as Preset**

**If your Insta360 Studio supports presets:**

1. Export chunk 020 with desired settings
2. Before clicking Export: "Save as Preset" or "Save Template"
3. Name it: `Quan_Violin_Camera1_Detection`
4. For chunk 021: Load the preset
5. Export

---

### **What if Chunks Already Have Different Framing?**

**You have 3 options:**

#### **Option A: Keep Separate, Don't Concatenate**
- Analyze each chunk independently
- Merge detection results in post-processing
- **Pro:** Works with existing exports
- **Con:** More complex analysis workflow

#### **Option B: Re-export Problem Chunks**
- Identify which chunks have wrong framing
- Re-export those chunks with correct settings
- **Pro:** Clean, seamless result
- **Con:** Takes time to re-export

#### **Option C: Digital Stabilization (Advanced)**
- Use video editing software to digitally reframe
- Match crop/zoom between chunks
- **Pro:** No need to re-export from .insv
- **Con:** May lose resolution, requires manual work
- **Tools:** After Effects, DaVinci Resolve, FFmpeg

---

### **Detecting Frame Jumps in Exported Files:**

Run this check to see if your chunks have consistent framing:

```bash
# Check resolution and crop of both chunks
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 chunk_020.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 chunk_021.mp4

# Should output identical values, e.g.:
# 3840,2160
# 3840,2160
```

**Visual check:**
```bash
# Extract first frame of chunk 021 and last frame of chunk 020
ffmpeg -i chunk_020.mp4 -vf "select='eq(n,last_frame)'" -frames:v 1 last_frame_020.jpg
ffmpeg -i chunk_021.mp4 -vf "select='eq(n,0)'" -frames:v 1 first_frame_021.jpg

# Open both images side-by-side
# Check: Subject in same position/size?
```

---

### **Recommendation for Quan:**

**Best Practice:**

1. **Export chunk 020 first** (your test export)
2. **Verify it meets detection requirements**
3. **Document the exact settings used:**
   ```
   Quan - Camera 1 - Export Settings:
   - Angle: Front-side 45°
   - FOV: 95°
   - Resolution: 4K
   - Codec: H.264
   - Dewarp: ON
   - Direction Lock: ON
   ```
4. **Use IDENTICAL settings for chunk 021**
5. **Visual check:** Play last 5 seconds of chunk 020, then first 5 seconds of chunk 021
   - Should transition smoothly
   - No sudden jumps in framing

**If you're unsure:** Export chunk 020 and 021 separately with same settings, then run the verification script to confirm consistency before moving to Camera 2.

---

## Summary: Quick Checklist

**For optimal detection:**

- [ ] Subject fills **40-60%** of frame height
- [ ] **FOV: 90-110°** (narrow to medium)
- [ ] **Dewarp: ON**
- [ ] **Direction Lock: ON**
- [ ] **4K for primary camera**, 1080p for others
- [ ] **H.264 codec** (not H.265)
- [ ] **30 fps** (or 60 fps for very fast movements)
- [ ] **Export 30-second test** before full export
- [ ] **Verify detection works** before continuing
- [ ] **Document exact settings** (angle, FOV, zoom)
- [ ] **Use SAME settings for all chunks** from same camera
- [ ] **Run quality verification script** after export

**Remember:** You can always re-export with different settings later. Start with a conservative (slightly zoomed in) approach for Camera 1, test detection, then adjust for Camera 2-3.

**Frame Jumps:** Only acceptable if keeping chunks separate OR if there's a natural recording gap. For continuous tracking, use identical framing for all chunks from same camera.
