# Simplified Export Settings for Insta360 Studio (When No Advanced Options Available)

## Your Situation:
- ❌ Can't find FOV slider
- ❌ Can't find Dewarp option
- ❌ Can't find Advanced settings button
- ❌ Can't find Bitrate slider

**Good News:** Your version likely applies dewarp automatically during flat export!

---

## What You WILL See in Export Panel:

### **Basic Export Settings (What's Available):**

```
Insta360 Studio Export Panel:
┌─────────────────────────────┐
│ Resolution: [4K ▼]          │  ← THIS
│ Frame Rate: [30fps ▼]       │  ← THIS
│ Codec: [H.265 ▼]            │  ← THIS (might say "Quality" instead)
│                             │
│ Output: [Browse...]         │
│                             │
│      [Export]               │
└─────────────────────────────┘
```

---

## Optimal Settings for Detection (Simplified)

### **1. Resolution**

| Setting | File Size | Detection Quality | Recommended For |
|---------|-----------|-------------------|-----------------|
| **3840×2160 (4K)** | Large (~5-7 GB/30min) | Best - fingers/bow detail visible | Primary camera (Camera 1) |
| **1920×1080 (1080p)** | Medium (~2-3 GB/30min) | Good - general tracking works | Secondary cameras (Camera 2-3) |
| **2704×1520 (2.7K)** | Medium-large | Good - balance | Alternative |

**Recommendation for Quan:**
- **Camera 1:** Use **4K** (you need to see violin fingers clearly)
- **Camera 2-3:** Use **1080p** (save disk space)

---

### **2. Frame Rate**

| Setting | Detection Quality | File Size | Recommended For |
|---------|-------------------|-----------|-----------------|
| **30 fps** | Standard - works for most tracking | Normal | Most instruments (recommended) |
| **60 fps** | Better for fast motion | 2x larger | Only if analyzing very fast bow movements |
| **25 fps** | Cinematic | Smaller | Avoid for detection |

**Recommendation for Quan (Violin):**
- Use **30 fps** (sufficient for bow tracking)
- Only use 60 fps if you specifically need frame-by-frame bow analysis

---

### **3. Codec / Quality**

**What you might see:**

| If it says "Codec" | Choose |
|-------------------|--------|
| H.264 | **✓ Choose this** (best compatibility) |
| H.265 / HEVC | OK (smaller files, might have issues with some tools) |

| If it says "Quality" instead | Choose |
|------------------------------|--------|
| High / Best | **✓ Choose this** |
| Medium / Standard | OK (smaller files) |
| Low | ❌ Avoid |

**Recommendation:**
- **If "Codec" option exists:** Choose **H.264**
- **If "Quality" slider exists:** Choose **High** or **Best**

---

### **4. Bitrate (If No Slider Available)**

**Don't worry!** If you can't adjust bitrate manually, Insta360 Studio uses these defaults:

| Resolution | Codec | Automatic Bitrate |
|------------|-------|-------------------|
| 4K | H.265 (Quality: High) | ~80-100 Mbps ✓ |
| 4K | H.264 (Quality: High) | ~100-120 Mbps ✓ |
| 1080p | H.265 (Quality: High) | ~30-40 Mbps ✓ |
| 1080p | H.264 (Quality: High) | ~40-50 Mbps ✓ |

**These automatic bitrates are EXCELLENT for detection!**

**If you DO have a bitrate slider (sometimes labeled "Quality" with a number):**

| Resolution | Minimum | Recommended | Best |
|------------|---------|-------------|------|
| 4K | 50 Mbps | **80 Mbps** | 100 Mbps |
| 1080p | 20 Mbps | **30 Mbps** | 50 Mbps |

---

## Simplified Export Workflow for Quan

### **Step 1: Frame Your Shot in Insta360 Studio**

1. **Import chunk:** `VID_20250725_142824_00_020.insv`

2. **Choose viewing angle:**
   - **Drag the preview** to rotate your view
   - Find angle where:
     - Quan is facing towards you (front-side angle)
     - You can see fingers on violin clearly
     - Bow movement is visible
     - Subject fills ~50% of frame height

3. **Control FOV with mouse/trackpad:**
   - **Pinch to zoom in** (two-finger pinch) = narrows FOV, subject larger
   - **Pinch to zoom out** = wider FOV, subject smaller
   - **Goal:** Subject fills ~40-60% of frame height

4. **Once you have the right framing:**
   - Take a **screenshot** of the preview
   - Write down where you're looking (e.g., "front-side, slightly zoomed in")
   - This helps you replicate for chunk 021

---

### **Step 2: Export Settings (Simple)**

**Camera 1 (Primary Detection):**
```
Resolution: 3840×2160 (4K)
Frame Rate: 30 fps
Codec: H.264 (or Quality: High if no codec option)
[No other settings needed - defaults are fine]
```

**Camera 2-3 (Context/Backup):**
```
Resolution: 1920×1080 (1080p)
Frame Rate: 30 fps
Codec: H.264 (or Quality: High)
```

---

### **Step 3: Export & Rename**

1. Click **Export**
2. Wait for export to complete
3. Run rename script:
   ```bash
   cd "/Users/ywcm/Downloads/Musician Tracking V3"
   ./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1" "020"
   ```
4. Verify quality:
   ```bash
   ./verify_export_quality.sh "/Volumes/X10 Pro 1/Phan Dissertation Data/Quan - MultiCam Data - Violin - 2025-07-25/2D Converted - 360 Camera 1 - Quan - Violin - 2025-07-25 X5 1/2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_020.mp4"
   ```

---

### **Step 4: Replicate for Chunk 021**

**Critical:** Use the **same zoom/viewing angle** as chunk 020

1. Import chunk 021
2. Look at your screenshot from chunk 020
3. Drag preview to **match the exact same angle**
4. Pinch to zoom to **match the exact same zoom level**
5. Use **SAME export settings:**
   - Resolution: 4K
   - Frame Rate: 30 fps
   - Codec: H.264 (or Quality: High)
6. Export

**How to check if angles match:**
- Play last 5 seconds of chunk 020
- Then play first 5 seconds of chunk 021
- Subject should be in same position/size (no jump)

---

## What About Dewarp?

**If you can't find a dewarp setting, one of these is true:**

### **Scenario A: Dewarp Applied Automatically (Most Likely)**
- When you export to "flat" 2D video, dewarp is automatic
- Your exported video will NOT have fisheye distortion
- **Check:** Look at straight lines (walls, floor) in exported video
  - If they're straight → dewarp was applied ✓
  - If they're curved → you might be exporting "fisheye mode"

### **Scenario B: Dewarp in Viewing Mode**
- Some versions apply dewarp when you choose viewing angle
- The moment you drag the preview, it's dewarped
- **Check:** Does preview look "normal" (not fisheye)? Then it's dewarped ✓

### **How to Verify Dewarp Was Applied:**

After exporting chunk 020:

1. Open the exported MP4
2. Pause on a frame with straight lines (wall, doorframe, floor edge)
3. Check: Are the lines straight or curved?
   - **Straight** → ✓ Dewarp was applied (good!)
   - **Curved** → ❌ Fisheye still present (need to fix)

**If lines are still curved:**
- Look for "Flat" vs "Fisheye" export mode
- Choose "Flat" or "Perspective" mode (not "Fisheye" mode)

---

## Expected File Sizes (for Reference)

**Quan's Camera 1 (4K, H.264 or H.265 Quality: High):**

| Chunk | Original .insv | Expected Exported (4K) |
|-------|----------------|------------------------|
| 020 (~28.5 min) | 19 GB | ~5-7 GB |
| 021 (~24 min) | 16 GB | ~4-6 GB |

**If your files are much different:**
- **Much smaller** (< 2 GB for 4K): Bitrate too low or resolution wrong
- **Much larger** (> 10 GB): Very high bitrate (OK, just uses more space)

---

## Troubleshooting

### **"I can't match the viewing angle between chunks"**

**Solution:** Take reference screenshots

1. When exporting chunk 020:
   - Pause export
   - Take screenshot of preview window
   - Draw a reference point (e.g., "Quan's head should be at this height")

2. When exporting chunk 021:
   - Load your reference screenshot on another monitor or phone
   - Match the preview to the screenshot
   - Verify subject is same size/position

### **"Exported video looks distorted/curved"**

**Check your export mode:**
- Look for dropdown that says "Flat" vs "Fisheye"
- Choose **"Flat"** or **"Perspective"** (NOT "Fisheye")

### **"File size is huge (> 10 GB for 30 minutes)"**

**Your bitrate might be very high (which is OK for quality, but uses space):**
- If disk space is limited: try 1080p instead of 4K
- Or: Choose H.265 instead of H.264 (50% smaller files)

### **"Export is very slow"**

**Normal export times:**
- 4K export: ~30-60 minutes for 30 minutes of footage
- 1080p export: ~15-30 minutes for 30 minutes of footage

**To speed up:**
- Close other applications
- Use 1080p instead of 4K
- Use H.265 (faster encoding than H.264 on newer Macs)

---

## Quick Reference Card

**Copy this and keep it visible while exporting:**

```
QUAN - CAMERA 1 - EXPORT SETTINGS
==================================

Chunk 020:
- Resolution: 3840×2160 (4K)
- Frame Rate: 30 fps
- Codec: H.264 (or Quality: High)
- Viewing Angle: [Write your angle here after first export]
- Zoom Level: [Write zoom level here]
- Export time: [Write actual time for reference]

Chunk 021:
- Use IDENTICAL settings as chunk 020
- Match viewing angle using screenshot
- Verify no frame jump when playing back to back

Output Folder:
/Volumes/X10 Pro 1/Phan Dissertation Data/Quan - MultiCam Data - Violin - 2025-07-25/2D Converted - 360 Camera 1 - Quan - Violin - 2025-07-25 X5 1/

After Export:
1. Rename: ./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1" "020"
2. Verify: ./verify_export_quality.sh [filepath]
3. Check dewarp: straight lines are straight?
```

---

## Bottom Line

**For Quan's export, you only need to choose:**

1. **Resolution:** 4K (Camera 1) or 1080p (Camera 2-3)
2. **Frame Rate:** 30 fps
3. **Codec/Quality:** H.264 or "High" quality
4. **Viewing Angle:** Whatever shows Quan best (front-side)
5. **Zoom:** Subject fills ~50% of frame

**Bitrate will be automatic and will be fine!**

**Dewarp will likely be automatic when exporting flat 2D video!**

The only thing you need to manually control is the **viewing angle and zoom** (by dragging/pinching the preview), and you need to **match these between chunk 020 and 021**.
