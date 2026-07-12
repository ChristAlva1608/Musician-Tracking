# Modular Video Processing System

Refactored video processing pipeline with independent, testable components.

## Architecture

```
src/processors/
├── base_processor.py          # Base class with error handling
├── alignment_checker.py       # Step 1: Check existing alignment data
├── alignment_analyzer.py      # Step 2: Analyze video alignment
├── video_aligner.py          # Step 3: Create aligned videos
├── detection_processor.py    # Step 4: Run detection (pose, hand, face, emotion, transcript)
└── unified_video_creator.py  # Step 5: Create unified/stacked videos
```

## Key Features

✅ **Independent Testing**: Test each step separately
✅ **Fail-Fast**: Stops immediately on missing dependencies or errors
✅ **Clear Error Messages**: Provides installation instructions for missing packages
✅ **Modular Design**: Easy to maintain and extend

## Error Handling

Each processor validates:
1. **Dependencies**: Required Python packages
2. **Inputs**: Configuration and file paths
3. **Processing**: Step-specific logic

If any validation fails, the processor:
- Prints clear error message
- Provides fix instructions (e.g., pip install commands)
- Raises specific exception type
- **STOPS execution immediately** (no continuation)

## Testing Individual Processors

### Test Detection Processor (Includes Transcript)

The detection processor is the most important for testing as it includes all detection features (pose, hand, face, emotion, transcript).

```bash
# Basic test (processes full video)
python src/test_detection_processor.py \
  --video /path/to/video.mp4 \
  --camera cam_1

# Test with duration limit (first 60 seconds only)
python src/test_detection_processor.py \
  --video /path/to/video.mp4 \
  --camera cam_1 \
  --max-duration 60

# Test with offset (skip first 10 seconds)
python src/test_detection_processor.py \
  --video /path/to/video.mp4 \
  --camera cam_1 \
  --offset 10.0 \
  --processing-type use_offset

# Test with custom config
python src/test_detection_processor.py \
  --video /path/to/video.mp4 \
  --camera cam_1 \
  --config src/config/custom_config.yaml
```

### Test Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video, -v` | Path to video file | **Required** |
| `--camera, -c` | Camera prefix/name | `test_camera` |
| `--config` | Config file path | `src/config/config_v1.yaml` |
| `--offset` | Camera offset (seconds) | `0.0` |
| `--processing-type` | `use_offset` or `full_frames` | `full_frames` |
| `--max-duration` | Max duration to process (seconds) | None (full video) |

## Configuration for Transcript Testing

To test transcript processing specifically, ensure your config has:

```yaml
detection:
  transcript_model: "whisper"  # Enable transcript
  transcript_settings:
    whisper:
      model_size: "base"  # or "tiny", "small", "medium", "large"
      language: null  # Auto-detect, or specify "en", "ja", etc.

database:
  enabled: true  # To save transcript to database

video:
  save_output_video: true  # Save annotated video
  generate_report: true    # Generate analysis report
```

## Expected Output

When running detection processor test:

```
==============================================================================
DETECTION PROCESSOR TEST
==============================================================================
Video: /path/to/video.mp4
Camera: cam_1
Processing type: full_frames
Session ID: test_detection_20250122_143022

============================================================
DetectionProcessor - Dependency Validation
============================================================
   ✅ opencv-python available
   ✅ numpy available
   ✅ mediapipe available
   ✅ ultralytics available
   ✅ Whisper (transcript) available
   ⚠️  DeepFace not available - emotion detection may be limited
✅ All dependencies validated

============================================================
DetectionProcessor - Input Validation
============================================================
   📁 Output directory: src/output/annotated_detection_videos
   🎬 Processing 1 videos
✅ All inputs validated

============================================================
DetectionProcessor - Processing
============================================================
======================================================================
Processing: cam_1
Video: /path/to/video.mp4
======================================================================
⏱️  [Detection for cam_1] Starting...

🔧 Initializing detector...
✅ DetectorV2 initialized
📝 Session ID: test_detection_20250122_143022_cam_1
🤚 Hand Model: mediapipe
🏃 Pose Model: mediapipe
😊 Face Model: mediapipe
😢 Emotion Model: none
🎤 Transcript Model: whisper
⚠️ Bad Gestures: Enabled
💾 Database: Enabled

   🎬 Mode: Full frames (no duration limit)
   📊 Offset: 0.000s

   📋 Detection Configuration:
      📹 Input: /path/to/video.mp4
      🎬 Output: src/output/annotated_detection_videos/vid_cam_1_20250122_143022.mp4
      💾 Database: True
      📊 Session: test_detection_20250122_143022_cam_1
      ⏱️  Offset: 0.000s
      🎚️  Save video: True
      🔊 Audio: True
      📄 Report: True

🎬 Starting video processing...
[Detection progress updates...]
✅ Detection complete for cam_1

⏱️  [Detection for cam_1] Completed in 2m 34.56s

======================================================================
✅ Detection complete: 1/1 videos processed
======================================================================

✅ Processing completed successfully

==============================================================================
TEST COMPLETED SUCCESSFULLY
==============================================================================
✅ Processed 1/1 videos

Output videos:
  cam_1: src/output/annotated_detection_videos/vid_cam_1_20250122_143022.mp4
```

## Error Examples

### Missing Dependency

```
❌ DEPENDENCY ERROR: Missing required packages: openai-whisper
   Install with: pip install openai-whisper
   Processing stopped. Please install missing dependencies.
```

### Missing Video File

```
❌ VALIDATION ERROR: Video files not found:
  - cam_1: /path/to/missing.mp4
   Processing stopped. Please check your configuration.
```

### Processing Error

```
❌ PROCESSING ERROR: Face detection failed: No faces detected in video
   This may indicate no faces were detected or face model not properly configured.
   Processing stopped.
```

## Integration with Full Pipeline

The modular processors are designed to be used in `integrated_video_processor.py`. Each processor can be:

1. **Run independently** for testing
2. **Integrated** into the full pipeline
3. **Configured** via YAML config files
4. **Validated** before execution

## Next Steps

1. **Test detection processor**: Start with a short video to verify transcript works
2. **Check output**: Look for transcript data in database and output video
3. **Adjust config**: Fine-tune transcript settings (model size, language)
4. **Scale up**: Once working, process full videos

## Troubleshooting

### Whisper Not Found
```bash
pip install openai-whisper
```

### ffmpeg Not Found (for audio processing)
```bash
# Mac
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Database Connection Failed
```bash
# Check your database credentials in config
# Ensure PostgreSQL/Supabase is running
```

### Out of Memory (Whisper)
```yaml
# Use smaller whisper model in config
transcript_settings:
  whisper:
    model_size: "tiny"  # or "base" instead of "large"
```
