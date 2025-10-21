# Quick Reference: 360° to 2D Export

## Critical Dewarp Settings in Insta360 Studio

### Where to Find Settings:

**Insta360 Studio Interface:**
```
1. Import .insv file
2. Select viewing angle by dragging the preview
3. Click "Export" button
4. Export panel appears on right side →

   Export Settings Panel:
   ┌─────────────────────────────┐
   │ Resolution: [4K ▼]          │
   │ Frame Rate: [30fps ▼]       │
   │ Codec: [H.265 ▼]            │
   │                             │
   │ ⚙️ Advanced ▼               │  ← CLICK HERE!
   │   ✅ Enable Dewarp          │  ← CHECK THIS
   │   ✅ Direction Lock         │  ← CHECK THIS
   │   ❌ Optical Flow Stab.     │  ← UNCHECK
   │   🎯 Viewing Angle: [...]   │
   │                             │
   │ Output: [Browse...]         │
   │                             │
   │      [Export]               │
   └─────────────────────────────┘
```

---

## Export Checklist for Each Chunk

### Before Exporting:
- [ ] Choose viewing angle (front/side/overhead)
- [ ] Write down exact angle for consistency
- [ ] Check "Advanced" settings

### Export Settings:
- [ ] **Resolution:** 4K (3840×2160) or 1080p (1920×1080)
- [ ] **Frame Rate:** 30fps (or match original)
- [ ] **Codec:** H.265 (HEVC)
- [ ] **Dewarp:** ✅ ON
- [ ] **Direction Lock:** ✅ ON
- [ ] **Optical Flow Stabilization:** ❌ OFF
- [ ] **Output Folder:** Correct "2D Converted" folder

### After Export:
- [ ] Rename immediately using script
- [ ] Verify file size and duration
- [ ] Use SAME settings for next chunk

---

## Workflow for Quan Camera 1

### Chunk 020:
```bash
# 1. Export in Insta360 Studio (with dewarp ON, direction lock ON)
# 2. Then run:
cd "/Users/ywcm/Downloads/Musician Tracking V3"
./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1" "020"
```

### Chunk 021:
```bash
# 1. Export with SAME settings as chunk 020
# 2. Then run:
./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1" "021"
```

### Camera 2:
```bash
# Chunk 018:
./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 2" "X5 2" "018"

# Chunk 019:
./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 2" "X5 2" "019"
```

### Camera 3:
```bash
# Chunk 007:
./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 3" "X5 3" "007"

# Chunk 008:
./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 3" "X5 3" "008"
```

---

## Optional: Concatenate Chunks (If NO gaps)

**Only if chunks are continuous recordings with no stops:**

```bash
# Camera 1 - concatenate chunks 020 + 021:
./concat_360_chunks.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1"

# Camera 2 - concatenate chunks 018 + 019:
./concat_360_chunks.sh "Quan" "Violin" "2025-07-25" "Camera 2" "X5 2"

# Camera 3 - concatenate chunks 007 + 008:
./concat_360_chunks.sh "Quan" "Violin" "2025-07-25" "Camera 3" "X5 3"
```

---

## What Each Setting Does

| Setting | Effect | Recommended |
|---------|--------|-------------|
| **Dewarp** | Removes fisheye/curved distortion from 360° footage. Makes straight lines appear straight instead of curved. | ✅ ON (almost always) |
| **Direction Lock** | Locks the viewing direction even if camera physically rotated. Keeps horizon level and view stable. | ✅ ON (if camera moved) |
| **Optical Flow Stabilization** | AI-based smoothing of camera motion. Can cause warping artifacts at edges. | ❌ OFF (for analysis) |
| **Follow Mode** | Camera view follows detected subject (like musician). Alternative to Direction Lock. | ⚠️ Optional (test first) |

---

## Expected File Sizes

**For reference (Quan's session):**

| Chunk | Original .insv | Expected 2D (4K H.265) | Expected 2D (1080p H.265) |
|-------|----------------|------------------------|---------------------------|
| 020 | 19 GB (~28.5 min) | ~5-7 GB | ~2-3 GB |
| 021 | 16 GB (~24 min) | ~4-6 GB | ~1.5-2.5 GB |

**If your exported files are much larger:**
- Codec might be H.264 instead of H.265
- Bitrate might be too high
- Resolution might be higher than expected

**If export is very slow:**
- Close other apps
- Use 1080p instead of 4K
- Switch to H.264 (faster encoding)

---

## Troubleshooting

**Can't find "Dewarp" setting:**
- Look for "Advanced Settings" button/dropdown
- Might be called "Lens Correction" or "Distortion Correction"
- Some older versions: enabled by default (check documentation)

**Video looks curved/distorted:**
- Dewarp is OFF → turn it ON
- Or viewing angle is pointing at edge of 360° sphere

**Horizon is tilted:**
- Direction Lock is OFF → turn it ON
- Or camera was physically tilted during recording

**Viewing angle changes during video:**
- Direction Lock is OFF
- Or "Follow Mode" is affecting view

**Export fails or crashes:**
- Free up disk space (need ~3x the original file size)
- Update Insta360 Studio to latest version
- Try lower resolution (1080p instead of 4K)
