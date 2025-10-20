# Video Conversion Guide: 360° to 2D for Pose Detection

## Overview

This guide explains how to convert 360° videos and process 2D camera videos to prepare them for pose detection analysis using MediaPipe.

### Why Convert 360° Videos?

MediaPipe pose detection works best with standard perspective (2D) videos. Converting 360° equirectangular videos to 2D offers:

- ✅ **Better accuracy**: MediaPipe is trained on normal camera perspectives
- ✅ **Reduced distortion**: Equirectangular format has severe edge distortion
- ✅ **Smaller file sizes**: 50-70% reduction after conversion
- ✅ **Faster processing**: Smaller resolution and standard format
- ✅ **Focused view**: Select optimal viewing angle for the musician

## Workflow Overview

```
Raw Data Collection
        ↓
1. Scan Videos
   - Identify 360° cameras (X3, X4, X5)
   - Identify 2D cameras (iPhone, GoPro)
   - Check for missing files or issues
        ↓
2. [360° Only] Select Viewing Angles
   - Interactive preview with arrow key controls
   - Save angle settings per camera
        ↓
3. [360° Only] Batch Convert to 2D
   - Apply saved angles to all chunks
   - FFmpeg perspective projection
   - Output: 1920x1080 MP4 files
        ↓
4. [2D Cameras] Rename to Chunk Format
   - Sort by timestamp
   - Rename to chunk### format
   - Optional: transcode to standard format
        ↓
5. Run Hybrid Alignment
   - Detect gaps between chunks
   - Synchronize all cameras
        ↓
6. Run Pose Detection
   - Process all cameras
   - Export 3D landmark data
```

## Installation

### Prerequisites

1. **FFmpeg** (for video conversion)
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Python packages**
   ```bash
   pip install opencv-python numpy
   ```

3. **Insta360 Studio** (ONLY for X3 cameras - optional)
   - Download from: https://www.insta360.com/download
   - Required to stitch X3 dual files (_00_ + _10_)
   - X4/X5 cameras produce pre-stitched files, no need for this step

## Usage

### Quick Start (Recommended)

Run the complete workflow for a participant:

```bash
python src/video_conversion_workflow.py --participant P03
```

This interactive workflow will guide you through all steps.

### Step-by-Step Manual Usage

#### Step 1: Scan Videos

Check what videos are available for a participant:

```bash
python src/video_converter_360_to_2d.py --mode scan --participant P03
```

Example output:
```
📊 Scan Results for P03:

X4: 4 videos
  - VID_20250705_163421_00_018.mp4
  - VID_20250705_163421_00_019.mp4
  - VID_20250705_163421_00_020.mp4
  ... and 1 more

iPhone: 3 videos
  - IMG_1234.mov
  - IMG_1235.mov
  - IMG_1236.mov
```

#### Step 2: Interactive Angle Selection (360° Only)

Select viewing angles for each 360° camera:

```bash
python src/video_converter_360_to_2d.py --mode interactive --participant P03
```

**Interactive Controls:**
- **Arrow Keys**: Adjust yaw (left/right) and pitch (up/down)
- **Q/E**: Adjust roll (rotation)
- **+/-**: Adjust field of view (FOV)
- **SPACE**: Accept current angle
- **ESC**: Cancel

**Tips for Angle Selection:**
- Center the musician in the view
- Use FOV 120° for good balance (wider = more context, narrower = less distortion)
- Slight downward pitch often works well for standing musicians
- Save angles once - they'll be applied to all chunks from that camera

Angles are saved to: `Processed_Videos/camera_angles.json`

#### Step 3: Batch Convert 360° Videos

Convert all 360° videos using saved angles:

```bash
python src/video_converter_360_to_2d.py --mode batch --participant P03
```

This will:
- Load saved angles for each camera
- Convert all chunks to 2D perspective
- Output to: `Processed_Videos/P03/[DATE]/Cam#/`
- Create log: `Processed_Videos/conversion_log.csv`

**Conversion Settings:**
- Output resolution: 1920x1080
- Projection: Perspective (from equirectangular)
- Codec: H.264
- Quality: CRF 23 (good balance)

#### Step 4: Process 2D Cameras

Rename iPhone/GoPro videos to chunk format:

```bash
# Option 1: Copy with rename (fast)
python src/camera_2d_renamer.py --mode rename --participant P03

# Option 2: Transcode to standard format (slower, standardizes resolution)
python src/camera_2d_renamer.py --mode transcode --participant P03
```

This will:
- Sort videos by creation timestamp
- Rename to sequential chunk### format
- Output to: `Processed_Videos/P03/[DATE]/Cam#/`

## File Naming Convention

After conversion, all videos follow the standard naming:

```
P03_Cam1_chunk001.mp4   # 360° Camera 1 (e.g., X3)
P03_Cam1_chunk002.mp4
P03_Cam2_chunk001.mp4   # 360° Camera 2 (e.g., X4)
P03_Cam3_chunk001.mp4   # 360° Camera 3 (e.g., X5)
P03_Cam4_chunk001.mp4   # 2D Camera 1 (e.g., iPhone)
P03_Cam5_chunk001.mp4   # 2D Camera 2 (e.g., GoPro)
```

## Directory Structure

```
Processed_Videos/
├── camera_angles.json          # Saved viewing angles
├── conversion_log.csv          # 360° conversion log
├── 2d_camera_log.csv          # 2D camera processing log
├── conversion_report.json      # Workflow summary
└── P03/
    └── 20250705/
        ├── Cam1/               # X3 converted
        │   ├── P03_Cam1_chunk001.mp4
        │   ├── P03_Cam1_chunk002.mp4
        │   └── ...
        ├── Cam2/               # X4 converted
        │   └── ...
        ├── Cam3/               # X5 converted
        │   └── ...
        ├── Cam4/               # iPhone
        │   └── ...
        └── Cam5/               # GoPro
            └── ...
```

## Special Case: X3 Cameras

Insta360 X3 cameras save dual files for front/back hemispheres:

```
VID_20250705_163421_00_021.insv  # Front (LL=00)
VID_20250705_163421_10_021.insv  # Back (LL=10)
```

### Option 1: Use Insta360 Studio (Recommended)

1. Open Insta360 Studio
2. Import both files (Studio auto-detects pairs)
3. Export as stitched 360° MP4
4. Place stitched video in camera folder
5. Run conversion workflow

### Option 2: Command Line (If Insta360 CLI Available)

```bash
# Check if CLI is installed
insta360-stitch --help

# Batch stitch (if supported)
insta360-stitch --input VID_*_00_*.insv --output stitched/
```

## Conversion Quality Settings

### Current Settings (Optimized for Pose Detection)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Output Resolution | 1920x1080 | Good balance for MediaPipe |
| Projection | Perspective | Unwarp from equirectangular |
| FOV | 120° (adjustable) | User-selected during interactive mode |
| Codec | H.264 (libx264) | Universal compatibility |
| CRF | 23 | Good quality (lower = better) |
| Preset | medium | Encoding speed |

### To Modify Settings

Edit `src/video_converter_360_to_2d.py`:

```python
self.default_settings = {
    "projection": "perspective",
    "fov_h": 120,        # Adjust default FOV
    "fov_v": 90,
    "width": 1920,       # Change resolution
    "height": 1080,
    "crf": 23,           # Lower = better quality, larger files
    "preset": "medium"   # faster/slower encoding
}
```

## FFmpeg Command Reference

The tool uses this FFmpeg command for conversion:

```bash
ffmpeg -i input_360.mp4 \
  -vf "v360=e:flat:iv_fov=360:ih_fov=180:yaw=180:pitch=0:roll=0:w=1920:h=1080:interp=linear" \
  -c:v libx264 -preset medium -crf 23 \
  -c:a copy \
  -y output_2d.mp4
```

**Parameters:**
- `e:flat` = equirectangular to flat (perspective)
- `yaw`, `pitch`, `roll` = viewing angle
- `w`, `h` = output dimensions
- `interp=linear` = interpolation method

## Troubleshooting

### Issue: "No saved angles found"

**Solution:** Run interactive mode first:
```bash
python src/video_converter_360_to_2d.py --mode interactive --participant P03
```

### Issue: "Cannot open video"

**Possible causes:**
- Video file is corrupted
- Missing FFmpeg installation
- X3 video not stitched yet

**Solution for X3:**
1. Check if you have both _00_ and _10_ files
2. Stitch using Insta360 Studio
3. Replace dual files with stitched version

### Issue: Preview window not responding

**Solution:**
- Click on the preview window to focus it
- Press ESC to exit, restart the tool
- On macOS, grant Terminal camera/screen recording permissions if needed

### Issue: Conversion very slow

**Solutions:**
- Use faster preset: Change `preset` to `fast` or `ultrafast`
- Reduce output resolution: Change to 1280x720
- Process fewer videos in parallel
- Use hardware acceleration (if available):
  ```bash
  # Add to FFmpeg command
  -hwaccel auto
  ```

### Issue: Output files very large

**Solutions:**
- Increase CRF value (e.g., 28 for smaller files)
- Reduce resolution
- Use 2-pass encoding for better compression

## Advanced Usage

### Batch Process Multiple Participants

```bash
for participant in P03 P04 P05; do
    python src/video_conversion_workflow.py \
        --participant $participant \
        --skip-interactive
done
```

### Custom Data Path

```bash
python src/video_conversion_workflow.py \
    --participant P03 \
    --data-root "/Volumes/MyDrive/Data" \
    --output-root "./output"
```

### Process Specific Date Folder

```bash
python src/video_converter_360_to_2d.py \
    --mode batch \
    --participant P03 \
    --date-folder 20250705
```

## Integration with Pose Detection Workflow

After conversion, proceed with:

1. **Hybrid Alignment** (detect gaps, synchronize)
   ```bash
   python src/hybrid_video_aligner.py \
       --input-dir Processed_Videos/P03/20250705 \
       --output alignment_output.json
   ```

2. **Pose Detection** (extract landmarks)
   ```bash
   python src/detect_v2_3d.py \
       --mode yolo+mediapipe \
       --video Processed_Videos/P03/20250705/Cam1/P03_Cam1_chunk001.mp4
   ```

## Performance Estimates

Based on typical video sizes:

| Step | Time per Chunk (30s) | Notes |
|------|---------------------|-------|
| Interactive angle selection | 1-2 min (once per camera) | Save and reuse |
| 360° to 2D conversion | 2-5 min | Depends on resolution |
| 2D camera copy | 5-10 sec | Fast |
| 2D camera transcode | 1-3 min | Optional |

**Example workflow for Jennifer (4 chunks, 1 camera):**
- Scan: 5 seconds
- Interactive: 2 minutes (once)
- Batch convert: 10-20 minutes
- **Total: ~15-25 minutes**

## Camera Angle Tips

### For Solo Musicians

- **Frontal view**: yaw=0°, pitch=-10° (slightly looking down)
- **Side view**: yaw=90° or 270°, pitch=0°
- **Over-shoulder**: yaw=45°, pitch=-15°

### For Ensembles

- **Wide view**: Use larger FOV (140-160°)
- **Focus on section**: Narrow FOV (90-100°), adjust yaw to center

### For Instrument-Specific Analysis

- **Hands/fingers**: Higher pitch (looking down), narrow FOV
- **Full body**: Neutral pitch, wider FOV
- **Embouchure**: Close-up frontal, narrow FOV

## Next Steps

1. ✅ Convert videos using this guide
2. ✅ Review conversion logs for any failures
3. ⏭️ Run hybrid alignment (see `HYBRID_ALIGNMENT_GUIDE.md`)
4. ⏭️ Run pose detection (see `INTEGRATED_VIDEO_PROCESSOR_README.md`)
5. ⏭️ Export and analyze landmark data

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review conversion logs in `Processed_Videos/`
3. Test with a single chunk first before batch processing
4. Verify FFmpeg installation: `ffmpeg -version`

---

**Created:** 2025-10-20
**Last Updated:** 2025-10-20
**Version:** 1.0
