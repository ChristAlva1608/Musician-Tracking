# Installation Guide for Video Conversion Tools

## Prerequisites

### 1. Python Environment

Ensure you have Python 3.8 or later:
```bash
python3 --version
```

### 2. Install FFmpeg

FFmpeg is required for video conversion.

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add to PATH: `C:\ffmpeg\bin`

**Verify installation:**
```bash
ffmpeg -version
```

### 3. Install Python Dependencies

**Option 1: Install only conversion tool dependencies**
```bash
pip3 install opencv-python numpy
```

**Option 2: Install all project dependencies** (includes pose detection)
```bash
pip3 install -r requirements.txt
```

## Quick Test

Test that everything is installed correctly:

```bash
# Test Python imports
python3 -c "import cv2, numpy; print('✓ Dependencies OK')"

# Test FFmpeg
ffmpeg -version

# Test scripts
python3 src/video_conversion_workflow.py --help
```

## Troubleshooting

### Issue: "No module named 'cv2'"

**Solution:**
```bash
pip3 install opencv-python
```

### Issue: "ffmpeg: command not found"

**Solution:**
- Install FFmpeg using instructions above
- Verify it's in your PATH: `which ffmpeg`

### Issue: Permission denied

**Solution:**
```bash
chmod +x src/video_converter_360_to_2d.py
chmod +x src/camera_2d_renamer.py
chmod +x src/video_conversion_workflow.py
```

### Issue: ImportError on macOS

**Solution:**
If you see Qt or GUI-related errors:
```bash
pip3 uninstall opencv-python
pip3 install opencv-python-headless
```

Then edit the scripts to use headless version (no GUI preview available).

## Optional: Insta360 Studio

**Only required for X3 cameras** (to stitch dual _00_ and _10_ files).

Download from: https://www.insta360.com/download

- X4 and X5 cameras don't need this (they produce pre-stitched files)
- Alternative: Use Insta360 CLI if available

## Verify Setup

Run a quick scan to verify everything works:

```bash
python3 src/video_conversion_workflow.py \
    --participant P03 \
    --data-root "/path/to/your/data"
```

This will scan for videos without converting anything.

## Next Steps

Once installed, proceed to:
- `VIDEO_CONVERSION_GUIDE.md` - Full usage guide
- `CONVERSION_QUICK_REFERENCE.md` - Command reference
