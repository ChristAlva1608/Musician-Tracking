# Unified Video Test Scripts

This directory contains test scripts for the unified video creation functionality.

## Test Scripts

### 1. `test_unified_video.py`
Tests the complete unified video workflow including the IntegratedVideoProcessor setup.

**Usage:**
```bash
python test/test_unified_video.py [video_directory]
```

**What it tests:**
- Complete IntegratedVideoProcessor initialization
- Mock detection video setup
- Full unified video creation workflow
- Configuration handling

### 2. `test_video_stacking.py`
Tests the specific video stacking functionality (`_create_stacked_video_with_sync` method).

**Usage:**
```bash
python test/test_video_stacking.py [video_directory]
```

**What it tests:**
- Direct video stacking function
- Video info extraction
- Frame synchronization
- Vertical/horizontal stacking
- Camera label overlays

## Default Test Directory
Both scripts default to:
```
/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/video/multi-cam video/vid_shot1/chunk_video
```

## Output
Test results are saved to:
```
test/output/
```

## Requirements
- OpenCV (cv2)
- All project dependencies from requirements.txt

## Example Usage

Test with default directory:
```bash
python test/test_video_stacking.py
```

Test with custom directory:
```bash
python test/test_video_stacking.py "/path/to/your/videos"
```

## Troubleshooting

If you get import errors, make sure you're running from the project root directory:
```bash
cd "/Volumes/Extreme_Pro/Mitou Project/Musician Tracking"
python test/test_video_stacking.py
```