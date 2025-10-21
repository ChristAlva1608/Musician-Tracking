# Export Settings Checklist - What to Adjust and Where

## Two Separate Places to Configure:

### **1. KEYFRAME Settings (Controls the VIEW)**
**Where:** Click on the timeline → Set keyframe → Keyframe panel

**What to set:**
- ✅ **Projection: Linear** (controls distortion correction)
- ✅ **FOV: 95-100°** (controls zoom/how much you see)
- ✅ **Viewing Angle:** Drag preview to frame subject
- ✅ **Direction Lock: ON** (keeps horizon level)

**This controls:** What the image LOOKS like (angle, zoom, distortion)

---

### **2. EXPORT Settings (Controls the FILE QUALITY)**
**Where:** Export panel (right side when you click Export button)

**What to set:**
- ✅ **Resolution: 3840×2160 (4K)** or 1920×1080 (1080p)
- ✅ **Frame Rate: 30 fps**
- ✅ **Codec: H.264** (or H.265)
- ✅ **Bitrate: 80 Mbps** (for 4K) or 30 Mbps (for 1080p)
  - OR **Quality: High** (if no bitrate slider)

**This controls:** File size, sharpness, detail quality

---

## Yes, You Need to Adjust BOTH!

### **Common Confusion:**

❌ **Wrong thinking:** "I set 4K resolution, so quality is automatically optimal"
✅ **Correct thinking:** "I need 4K resolution AND high bitrate for optimal quality"

**Why both matter:**

| Setting | What It Controls | Example |
|---------|------------------|---------|
| **Resolution** | Number of pixels (dimensions) | 3840×2160 = 8.3 million pixels |
| **Bitrate** | Data per second (compression) | 80 Mbps = how much detail those pixels contain |

**Analogy:**
- **Resolution** = Size of the canvas (4K = big canvas)
- **Bitrate** = Quality of the paint (80 Mbps = fine detail paint)

You can have:
- 4K resolution + low bitrate (30 Mbps) = Big blurry image ❌
- 4K resolution + high bitrate (80 Mbps) = Big sharp image ✅
- 1080p + high bitrate (50 Mbps) = Small sharp image ✅

---

## Complete Export Workflow:

### **Step 1: Set Keyframe (Controls VIEW)**

```
Timeline → Set Keyframe → Keyframe Panel:
┌─────────────────────────────┐
│ Viewing Angle: [Drag view]  │ ← Position camera angle
│ FOV: [95-100° ▼]            │ ← Set field of view (zoom)
│ Projection: [Linear ▼]      │ ← Set to Linear
│ Direction Lock: [ON ☑]      │ ← Enable if available
└─────────────────────────────┘
```

**This answers:** "What do I see?" "How zoomed in?" "Is it distorted?"

---

### **Step 2: Configure Export (Controls QUALITY)**

```
Export Panel (right side):
┌─────────────────────────────┐
│ Resolution: [4K ▼]          │ ← Set resolution
│ Frame Rate: [30fps ▼]       │ ← Set frame rate
│ Codec: [H.264 ▼]            │ ← Set codec
│ Bitrate: [80 Mbps]          │ ← Set bitrate (IMPORTANT!)
│   OR                        │
│ Quality: [High ▼]           │ ← OR set quality level
│                             │
│ Output: [Browse...]         │
│      [Export]               │
└─────────────────────────────┘
```

**This answers:** "How big is the file?" "How sharp are the edges?" "How much detail?"

---

## For Quan - Camera 1 (Primary Detection):

### **Keyframe Settings:**
```
☑ Projection: Linear
☑ FOV: 95-100° (adjust until subject fills ~50% of frame)
☑ Viewing Angle: Front-side (drag preview)
☑ Direction Lock: ON
```

### **Export Settings:**
```
☑ Resolution: 3840×2160 (4K)
☑ Frame Rate: 30 fps
☑ Codec: H.264
☑ Bitrate: 80 Mbps
  OR
☑ Quality: High (if no bitrate option)
```

### **Expected Result:**
- File size: ~5-7 GB for 30 minutes
- Detection rate: 98%+ with MediaPipe
- Fingers/bow clearly visible

---

## Why Bitrate is Independent from Resolution:

**Resolution determines HOW MANY pixels:**
- 1080p = 2,073,600 pixels per frame
- 4K = 8,294,400 pixels per frame (4× more pixels)

**Bitrate determines HOW MUCH DATA per pixel:**
- 30 Mbps / 8.3M pixels = 3.6 bits per pixel (low detail)
- 80 Mbps / 8.3M pixels = 9.6 bits per pixel (high detail)

**You can set them independently:**
- 4K @ 30 Mbps = Many pixels, little detail per pixel = Blurry
- 4K @ 80 Mbps = Many pixels, good detail per pixel = Sharp ✓
- 1080p @ 80 Mbps = Fewer pixels, lots of detail per pixel = Very sharp

---

## What Happens if You Only Set Resolution (and Ignore Bitrate)?

**Scenario:** You set 4K but leave bitrate at default (might be 30-40 Mbps)

**Result:**
```
Resolution: 4K ✓ (correct)
Bitrate: 30 Mbps (TOO LOW for 4K)

Effect:
- File looks OK when viewing small
- But zooming in shows blockiness/blur
- MediaPipe detection: 85-90% (should be 98%)
- Finger tracking: Intermittent (should be stable)
- Edge clarity: Poor (should be sharp)
```

**Fix:** Increase bitrate to 80 Mbps

---

## What Happens if Bitrate is Too High?

**Scenario:** You set 4K @ 150 Mbps

**Result:**
```
Resolution: 4K ✓
Bitrate: 150 Mbps (VERY HIGH)

Effect:
- File looks excellent
- MediaPipe detection: 98% (same as 80 Mbps - no improvement)
- File size: HUGE (~30 GB for 30 min instead of ~6 GB)
- Export time: Longer
- No benefit for detection
```

**Conclusion:** 80 Mbps is the "sweet spot" for 4K detection work

---

## Quick Decision Guide:

### **"Should I adjust bitrate if I already set 4K resolution?"**

**YES!** Check what your default bitrate is:

**If you see a bitrate slider/number:**
- Current bitrate < 60 Mbps → **Increase to 80 Mbps**
- Current bitrate 60-100 Mbps → **Already good**
- Current bitrate > 120 Mbps → **Can reduce to 80 Mbps** (saves space, no quality loss for detection)

**If you see a "Quality" dropdown instead:**
- Low/Medium → **Change to High**
- High/Best → **Already good**

**If you don't see bitrate OR quality:**
- Insta360 uses automatic bitrate
- Usually ~80-100 Mbps for 4K (good)
- No action needed

---

## How to Check Bitrate After Export:

```bash
# After exporting, verify actual bitrate:
ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 your_file.mp4 | awk '{print $1/1000000 " Mbps"}'
```

**Expected output:**
- 4K file: 60-100 Mbps ✓
- 1080p file: 25-40 Mbps ✓

**If lower:** Re-export with higher bitrate/quality setting

---

## Summary:

| What to Set | Where | Value | Why |
|-------------|-------|-------|-----|
| **Projection** | Keyframe panel | Linear | No distortion (critical for detection) |
| **FOV** | Keyframe panel | 95-100° | Subject fills ~50% of frame |
| **Resolution** | Export panel | 4K | Enough pixels to see fingers |
| **Bitrate** | Export panel | 80 Mbps | Sharp edges for keypoint detection |

**Both resolution AND bitrate matter independently!**

---

## Checklist Before Clicking Export:

```
Keyframe Settings:
☐ Projection: Linear
☐ FOV: 95-100°
☐ Direction Lock: ON
☐ Subject fills ~50% of frame

Export Settings:
☐ Resolution: 4K (3840×2160)
☐ Frame Rate: 30 fps
☐ Codec: H.264
☐ Bitrate: 80 Mbps (or Quality: High)
☐ Output folder: Correct path

Ready to Export!
```
