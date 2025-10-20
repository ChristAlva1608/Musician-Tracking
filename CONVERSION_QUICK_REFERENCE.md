# Video Conversion Quick Reference

## One-Command Workflow (Recommended)

```bash
# Interactive workflow with all steps
python src/video_conversion_workflow.py --participant P03
```

## Individual Commands

### 1. Scan Videos
```bash
python src/video_converter_360_to_2d.py --mode scan --participant P03
```

### 2. Select Angles (360° cameras)
```bash
python src/video_converter_360_to_2d.py --mode interactive --participant P03
```

**Preview Controls:**
- Arrow keys: adjust yaw/pitch
- +/- : adjust FOV
- SPACE: accept
- ESC: cancel

### 3. Convert 360° to 2D
```bash
python src/video_converter_360_to_2d.py --mode batch --participant P03
```

### 4. Process 2D Cameras
```bash
# Copy and rename
python src/camera_2d_renamer.py --mode rename --participant P03

# Transcode to standard format
python src/camera_2d_renamer.py --mode transcode --participant P03
```

## Common Options

```bash
# Custom data location
--data-root "/Volumes/MyDrive/Data"

# Custom output location
--output-root "./output"

# Specific date folder
--date-folder 20250705

# Skip interactive mode (use saved angles)
--skip-interactive
```

## Output Locations

```
Processed_Videos/
├── camera_angles.json       # Saved angles
├── conversion_log.csv       # 360° conversion log
├── 2d_camera_log.csv       # 2D processing log
└── P03/
    └── 20250705/
        ├── Cam1/           # Converted videos
        ├── Cam2/
        └── ...
```

## File Naming

After conversion:
```
P03_Cam1_chunk001.mp4
P03_Cam1_chunk002.mp4
P03_Cam2_chunk001.mp4
```

## Next Steps

1. Convert videos (this guide)
2. Align and sync:
   ```bash
   python src/hybrid_video_aligner.py \
       --input-dir Processed_Videos/P03/20250705 \
       --output alignment.json
   ```
3. Pose detection:
   ```bash
   python src/detect_v2_3d.py \
       --mode yolo+mediapipe \
       --video Processed_Videos/P03/20250705/Cam1/P03_Cam1_chunk001.mp4
   ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No saved angles | Run `--mode interactive` first |
| Cannot open video | Check FFmpeg: `ffmpeg -version` |
| X3 not working | Stitch with Insta360 Studio first |
| Preview frozen | Press ESC, restart tool |

## Camera Angle Recommendations

| View Type | Yaw | Pitch | FOV |
|-----------|-----|-------|-----|
| Frontal | 0° | -10° | 120° |
| Side | 90° | 0° | 120° |
| Wide | 0° | 0° | 150° |
| Close-up | (adjust) | -15° | 90° |

---

For full documentation, see: `VIDEO_CONVERSION_GUIDE.md`
