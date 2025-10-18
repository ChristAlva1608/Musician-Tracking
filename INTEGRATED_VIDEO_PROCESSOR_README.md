# Integrated Video Processor - Complete Guide

## Overview

The **Integrated Video Processor** (`src/integrated_video_processor.py`) is a comprehensive tool that combines video alignment and detection processing in a single automated workflow. It integrates three major components:

1. **Video Alignment** (`src/video_aligner/shape_based_aligner_multi.py`) - Audio-based synchronization
2. **Detection Processing** (`src/detect_v2_3d.py`) - Multi-modal detection (pose, hands, face, emotion, gestures, transcription)
3. **Unified Video Creation** - Multi-camera synchronized view generation

### Key Features

✅ **Automatic video alignment** using audio pattern matching
✅ **Database caching** of alignment data (PostgreSQL or Supabase)
✅ **Multi-modal detection** on synchronized videos
✅ **Chunk video support** (e.g., cam_1_1.mp4, cam_1_2.mp4)
✅ **Unified multi-camera views** with synchronized playback
✅ **Flexible processing modes** (full frames vs. synchronized portions)
✅ **Duration limiting** for quick testing
✅ **Audio preservation** in output videos

---

## Quick Start

### Basic Usage

```bash
# Run with default config (config_v1.yaml)
python src/integrated_video_processor.py

# Run with custom config
python src/integrated_video_processor.py --config path/to/config.yaml

# Override alignment directory from command line
python src/integrated_video_processor.py --alignment-dir /path/to/videos

# Alignment only (skip detection)
python src/integrated_video_processor.py --skip-detection

# Process full videos without duration limiting
python src/integrated_video_processor.py --no-duration-limit

# Process only first 30 seconds of each video (testing)
python src/integrated_video_processor.py --max-duration 30
```

### Configuration File

All settings are configured in `src/config/config_v1.yaml`. See [Configuration Structure](#configuration-structure) below for details.

---

## Configuration Structure

Configuration is managed through `src/config/config_v1.yaml`. Here's what each section controls:

### 1. Database Configuration

```yaml
database:
  enabled: true
  use_database_type: local  # Options: 'local' or 'supabase'

  # Local PostgreSQL Configuration (used when use_database_type: local)
  local:
    host: localhost
    port: 5432
    name: musician-tracking
    user: christalva
    password: ''

  # Supabase Configuration (used when use_database_type: supabase)
  supabase:
    url: 'https://your-project.supabase.co'
    anon_key: 'your-anon-key'

  # Database feature flags
  store_chunk_video_alignment: true      # Store chunk alignment data
  store_musician_frame_analysis: true    # Store detection frame data
  store_transcript_video: true           # Store transcript data
  link_frame_to_transcript: true         # Link frames to transcript segments
```

**When is each database table used?**

| Database Class | Config Flag | Use Case |
|----------------|-------------|----------|
| `ChunkVideoAlignmentDatabase` | `enable_chunk_processing: true` | For multi-chunk videos (cam_1_1.mp4, cam_1_2.mp4) |
| `VideoAlignmentDatabase` | `enable_chunk_processing: false` | For single-file videos (legacy) |
| `DatabaseManager` | `database.enabled: true` | For detection results (poses, hands, emotions, etc.) |

---

### 2. Video Aligner Configuration

```yaml
video_aligner:
  # Directory containing videos to align
  alignment_directory: /path/to/videos

  # Whether to save aligned video files
  save_output_videos: true

  # Alignment settings
  alignment:
    enable_chunk_processing: true        # Enable multi-chunk video support
    correlation_threshold: 0.3           # Audio correlation threshold
    max_offset_search: 60                # Max seconds to search for alignment
    reference_strategy: earliest_start   # Options: 'earliest_start' or 'latest_start'

  # Audio extraction settings
  audio:
    extraction_duration: 120             # Seconds of audio to analyze
    sample_rate: 22050                   # Audio sample rate
    window_ms: 100                       # Analysis window size

  # Chunk processing settings
  chunk_processing:
    chunk_gap_threshold: 60              # Max gap between chunks (seconds)
    combine_chunks: true                 # Combine chunks into single video
    gap_frame_repeat_duration: 1         # Duration for gap filling
```

**Chunk Processing Explained:**

- **`enable_chunk_processing: true`**: For videos split into chunks
  - Example: `cam_1_1.mp4`, `cam_1_2.mp4`, `cam_1_3.mp4`
  - Aligns each chunk individually
  - Stores per-chunk alignment data
  - Combines chunks into continuous video

- **`enable_chunk_processing: false`**: For single video files (legacy)
  - Example: `cam_1.mp4`, `cam_2.mp4`
  - Aligns complete videos
  - Stores simple offset per video

---

### 3. Integrated Processor Configuration

```yaml
integrated_processor:
  # === Core Workflow Control ===
  run_detection: true                    # Run detection after alignment
  create_aligned_videos: true            # Create aligned video outputs
  check_existing_alignment: true         # Check database before re-aligning

  # === Processing Type ===
  processing_type: full_frames           # Options: 'use_offset', 'full_frames'

  # === Duration Limiting ===
  limit_processing_duration: true        # Limit processing time per video
  max_processing_duration: 60            # Max seconds to process (if limiting enabled)

  # === Output Directories ===
  aligned_videos_dir: src/output/aligned_videos
  detection_videos_dir: src/output/annotated_detection_videos
  unified_videos_dir: src/output/unified_videos
  detection_output_format: mp4           # Output video format

  # === Unified Video Settings ===
  unified_videos: true                   # Create stacked multi-camera view
  unified_video_config:
    output_filename: unified_detection_output.mp4
    stack_direction: vertical            # Options: 'vertical', 'horizontal'
    add_camera_labels: true              # Add camera name labels
    label_color: [255, 255, 255]         # RGB color for labels
    label_font_scale: 1                  # Font size for labels
    background_color: [0, 0, 0]          # RGB background color
    freeze_frame_behavior: true          # Freeze frames when video ends

  # === Detection Overrides ===
  detector_config:
    database:
      enabled: true                      # Save detection results to DB
    display_output: false                # Show visualization window
    video:
      save_output_video: true            # Save annotated videos
      preserve_audio: true               # Keep original audio
      generate_report: true              # Generate text report
      process_matching_duration_only: true  # Process only aligned portion
```

---

## Processing Type Explained

### `processing_type: use_offset`

- **Starts processing from the alignment offset**
- **Skips frames before synchronization point**
- **Use case**: When you only want to process synchronized content across cameras
- **Example**:
  - cam_1 offset: 5.2s → starts processing at 5.2s
  - cam_2 offset: 0.0s → starts processing at 0.0s (reference)
  - cam_3 offset: 2.8s → starts processing at 2.8s

**Timeline visualization:**
```
cam_1: [--skip--][=====process=====]
cam_2: [=====process=====]
cam_3: [--skip--][=====process=====]
```

### `processing_type: full_frames`

- **Processes entire video from start to end**
- **Respects alignment for unified video creation**
- **Use case**: When you want complete analysis of each camera
- **Example**:
  - cam_1: processes all frames (0s → end)
  - cam_2: processes all frames (0s → end)
  - cam_3: processes all frames (0s → end)

**Timeline visualization:**
```
cam_1: [=====process all frames=====]
cam_2: [=====process all frames=====]
cam_3: [=====process all frames=====]
```

### **Important: Synced Time Calculation**

The `processing_type` setting directly controls how `synced_time` is calculated in the database:

**`processing_type: full_frames`**
```python
synced_time = original_time - start_time_offset
```
- Video starts at frame 0 (beginning)
- Subtract offset to align with other cameras
- Example: cam_1 at 10s with 5s offset → synced_time = 5s

**`processing_type: use_offset`**
```python
synced_time = original_time
```
- Video seeked to offset position
- Frame 0 corresponds to offset position
- No adjustment needed
- Example: cam_1 seeked to 5s, frame at 10s → synced_time = 10s

**Note**: `limit_processing_duration` no longer affects `synced_time` calculation - only `processing_type` matters.

---

## Auto-Pairing: processing_type ↔ reference_strategy

The system automatically pairs `processing_type` with the appropriate alignment strategy. **You no longer need to set `reference_strategy` manually** - it's determined automatically:

| Condition | Auto-Selected Strategy | Reason |
|-----------|------------------------|---------|
| `unified_videos: true` | **earliest_start** | Need full timeline for synchronization |
| `processing_type: full_frames` | **earliest_start** | Process all available content |
| `processing_type: use_offset` | **latest_start** | Process only synchronized portion |

**Example Console Output:**
```
🎯 Auto-selected alignment: earliest_start (full_frames processes all content)
   processing_type=full_frames, unified_videos=true
```

**Config Simplification:**
```yaml
integrated_processor:
  processing_type: full_frames  # Automatically uses earliest_start
  unified_videos: true

# reference_strategy is DEPRECATED - auto-determined by processing_type
```

---

## Duration Limiting

### `limit_processing_duration: true`

Processes only a limited duration per video:

```yaml
limit_processing_duration: true
max_processing_duration: 60  # Process only 60 seconds
```

**Combined with `processing_type`:**

| processing_type | limit_processing_duration | Result |
|----------------|---------------------------|--------|
| `use_offset` | `true` | Process 60s starting from offset |
| `use_offset` | `false` | Process from offset to end |
| `full_frames` | `true` | Process first 60s from start |
| `full_frames` | `false` | Process entire video |

---

## Unified Videos

### What are Unified Videos?

Unified videos combine multiple camera views into a single video with synchronized playback.

```yaml
unified_videos: true
unified_video_config:
  stack_direction: vertical  # or 'horizontal'
  add_camera_labels: true
  freeze_frame_behavior: true
```

**Behavior:**
- Videos are stacked vertically or horizontally
- All cameras play in sync based on alignment offsets
- If a camera ends early, its last frame is frozen
- If a camera starts late, its first frame is shown frozen
- Audio is taken from the reference camera (earliest start)

**Example with 3 cameras:**
```
┌─────────────────┐
│   cam_1 view    │  ← Camera 1 (offset: 5s)
├─────────────────┤
│   cam_2 view    │  ← Camera 2 (offset: 0s, reference)
├─────────────────┤
│   cam_3 view    │  ← Camera 3 (offset: 3s)
└─────────────────┘
```

---

## Complete Workflow

When you run `python src/integrated_video_processor.py`, the system executes this complete workflow:

```
┌─────────────────────────────────────────────────────────────┐
│  INTEGRATED VIDEO PROCESSOR - MAIN WORKFLOW                │
└─────────────────────────────────────────────────────────────┘

📂 STEP 1: CHECK EXISTING ALIGNMENT DATA
   ├─ Query database for cached alignment data
   ├─ If enable_chunk_processing=true:
   │  └─ Query chunk_video_alignment_offsets table
   └─ Else:
      └─ Query video_alignment_offsets table (legacy)

   ✅ Found? → Use cached data
   ❌ Not found? → Proceed to Step 2

📊 STEP 2: ANALYZE VIDEO ALIGNMENT (if no cache)
   ├─ Scan videos in alignment_directory
   ├─ Group videos by camera prefix (e.g., cam_1, cam_2)
   ├─ Extract audio from first chunk of each camera
   ├─ Calculate energy profiles (RMS audio patterns)
   ├─ Determine reference camera using audio correlation
   │  ├─ Auto-pair processing_type with reference_strategy:
   │  │  ├─ unified_videos=true → earliest_start
   │  │  ├─ processing_type=full_frames → earliest_start
   │  │  └─ processing_type=use_offset → latest_start
   │  └─ Select reference (camera with most/least content)
   ├─ Align all chunks to reference timeline
   ├─ Calculate absolute timeline positions
   └─ Save alignment data to database

🎬 STEP 3: CREATE ALIGNED VIDEOS
   ├─ Check if videos are chunked (e.g., cam_1_1.mp4, cam_1_2.mp4)
   ├─ If chunked:
   │  └─ Merge chunks into continuous videos using ffmpeg concat
   └─ If not chunked:
      └─ Map original video paths (no file copying)

🔍 STEP 4: RUN DETECTION ON ALIGNED VIDEOS
   For each camera video:
   ├─ Initialize DetectorV2 with config overrides
   ├─ Set processing_type for synced_time calculation:
   │  ├─ full_frames: synced_time = original_time + offset
   │  └─ use_offset: synced_time = original_time (video pre-seeked)
   ├─ Configure frame processing mode:
   │  ├─ full_frames: Process from frame 0 (no seeking)
   │  └─ use_offset: Seek to offset, then process
   ├─ Apply duration limiting (if enabled):
   │  ├─ limit_processing_duration=true: Process max_processing_duration seconds
   │  └─ limit_processing_duration=false: Process entire video
   ├─ Run detection models:
   │  ├─ MediaPipe/YOLO Pose Detection (3D world landmarks)
   │  ├─ MediaPipe/YOLO Hand Detection (3D world landmarks)
   │  ├─ YOLO+MediaPipe Face Mesh Detection
   │  ├─ DeepFace/GhostFaceNet Emotion Detection
   │  ├─ 3D Bad Gesture Detection (low wrists, turtle neck, etc.)
   │  └─ Whisper Transcription (with database caching)
   ├─ Save frame-level results to musician_frame_analysis table
   ├─ Save transcript segments to transcript_video table
   ├─ Generate annotated video with:
   │  ├─ Pose/hand/face landmarks overlays
   │  ├─ Bad gesture warnings
   │  ├─ Transcription subtitles
   │  └─ Processing time statistics
   └─ Preserve original audio using ffmpeg

📹 STEP 5: CREATE UNIFIED VIDEO (if unified_videos=true)
   ├─ Load all detection output videos
   ├─ Read alignment offsets for each camera
   ├─ Determine output dimensions based on stack_direction:
   │  ├─ vertical: stack videos top-to-bottom
   │  └─ horizontal: stack videos left-to-right
   ├─ Process frame-by-frame with synchronized playback:
   │  ├─ Before camera starts: Show first frame (frozen)
   │  ├─ During playback: Show actual video frames
   │  └─ After camera ends: Show last frame (frozen)
   ├─ Add camera labels if enabled
   ├─ Extract and concatenate audio from reference camera
   └─ Save unified video with synchronized audio

🧹 STEP 6: CLEANUP
   ├─ Remove temporary video files
   ├─ Remove temp audio files
   ├─ Close database connections
   └─ Generate processing report (if enabled)
```

### Key Processing Modes

**Processing Type** determines how videos are processed:

| Mode | Video Seeking | Frame Processing | Synced Time Calculation | Use Case |
|------|--------------|------------------|------------------------|----------|
| `full_frames` | No seeking, start at frame 0 | Process entire video | `synced_time = original_time + offset` | Complete analysis of all content |
| `use_offset` | Seek to offset position | Process from sync point | `synced_time = original_time` | Only synchronized portions |

**Duration Limiting** controls processing length:

| Setting | Behavior | Example |
|---------|----------|---------|
| `limit_processing_duration=false` | Process entire video | Full 5-minute video processed |
| `limit_processing_duration=true, max_processing_duration=60` | Process only first N seconds | Only first 60s processed |

---

## Command Line Options

The integrated processor supports the following command line arguments:

```bash
python src/integrated_video_processor.py [OPTIONS]

Arguments:
  --config, -c PATH          Path to configuration file
                             Default: src/config/config_v1.yaml
                             Example: --config my_config.yaml

  --alignment-dir, -a PATH   Override alignment_directory from config
                             Useful for processing different video sets
                             Example: --alignment-dir /path/to/vid_shot2

  --skip-detection           Skip detection processing entirely
                             Only performs alignment and video creation
                             Useful for: Testing alignment, preparing videos
                             Example: --skip-detection

  --skip-video-creation      Skip aligned video creation
                             Only performs alignment analysis
                             Useful for: Database population, testing alignment
                             Example: --skip-video-creation

  --max-duration SECONDS     Set maximum processing duration (seconds)
                             Overrides config's max_processing_duration
                             Enables limit_processing_duration automatically
                             Example: --max-duration 30

  --no-duration-limit        Disable duration limiting completely
                             Processes entire videos regardless of config
                             Overrides limit_processing_duration setting
                             Example: --no-duration-limit
```

### Command Line Override Behavior

Command line arguments **override** config file settings:

| Config Setting | Command Line | Final Behavior |
|----------------|--------------|----------------|
| `limit_processing_duration: true` | `--no-duration-limit` | ✅ Full video processing |
| `limit_processing_duration: false` | `--max-duration 60` | ✅ 60s processing limit |
| `run_detection: true` | `--skip-detection` | ✅ Detection skipped |
| `alignment_directory: /path/A` | `--alignment-dir /path/B` | ✅ Uses /path/B |

---

## Example Configurations

### Configuration 1: Quick Test (60s Processing)

```yaml
integrated_processor:
  run_detection: true
  processing_type: use_offset
  limit_processing_duration: true
  max_processing_duration: 60
  unified_videos: true
  detector_config:
    database:
      enabled: true
```

**Result**: Aligns videos, processes 60s from sync point, creates unified video, saves to database.

---

### Configuration 2: Full Analysis (Complete Videos)

```yaml
integrated_processor:
  run_detection: true
  processing_type: full_frames
  limit_processing_duration: false
  unified_videos: true
  detector_config:
    database:
      enabled: true
```

**Result**: Processes entire videos from start to finish, creates unified view.

---

### Configuration 3: Alignment Only (No Detection)

```yaml
integrated_processor:
  run_detection: false
  create_aligned_videos: true
  unified_videos: false
```

**Result**: Only performs video alignment, creates aligned outputs, no detection or unified video.

---

### Configuration 4: Detection Without Database

```yaml
integrated_processor:
  run_detection: true
  processing_type: use_offset
  unified_videos: false
  detector_config:
    database:
      enabled: false
    video:
      save_output_video: true
      preserve_audio: true
```

**Result**: Runs detection and creates annotated videos, but doesn't save to database.

---

## Detection Models Configuration

Detection models are configured in the `detection` section:

```yaml
detection:
  # Model selection
  pose_model: mediapipe              # Options: 'mediapipe', 'yolo'
  hand_model: mediapipe              # Options: 'mediapipe', 'yolo'
  facemesh_model: yolo+mediapipe     # Options: 'mediapipe', 'yolo', 'yolo+mediapipe'
  emotion_model: none                # Options: 'none', 'deepface', 'ghostfacenet'
  transcript_model: whisper          # Options: 'none', 'whisper'

  # Confidence thresholds
  pose_confidence: 0.5
  hand_confidence: 0.5
  face_confidence: 0.5
  facemesh_confidence: 0.5

  # Transcript settings
  transcript_settings:
    enabled: true
    model_size: tiny                 # Options: tiny, base, small, medium, large
    language: en
    chunk_duration: 15
```

---

## Transcript Validation and Reuse

The system intelligently validates existing transcript data in the database to avoid redundant processing:

### How Transcript Validation Works

When processing a video with transcription enabled:

1. **Check Database**: The system first checks if transcript segments already exist for the video file
2. **Calculate Video Duration**: Determines the total duration of the video being processed
3. **Validate Time Coverage**: Checks if the maximum `end_time` of all transcript records covers the full video duration:
   ```
   max_end_time >= (video_duration - tolerance)
   ```
   - Default tolerance: 2.0 seconds (configurable)

4. **Decision**:
   - **If complete coverage**: ✅ Reuse existing transcripts (skip processing)
   - **If incomplete coverage**: ⚠️ Reprocess from scratch with new session_id

### Why Time Coverage Instead of Segment Count?

Whisper doesn't create fixed-duration segments. Instead, it creates variable-length segments based on:
- Natural speech boundaries (sentences, pauses)
- Silence detection
- Language patterns

**Example**: A 90-second video might have:
- 25 small segments (3-5 seconds each) with speech
- 8 empty segments (duration=0) for silence
- **Total**: 33 records (not a predictable 6 segments)

Time coverage validation handles this correctly by checking if transcripts cover the timeline, regardless of segment count.

### Example Scenarios

**Scenario 1: Complete Transcript Data**
```
Video duration: 90.0s
Found records: 33 segments (25 with text, 8 empty/silent)
Time coverage: 0.0s → 89.8s
Max end_time: 89.8s >= (90.0 - 2.0) ✅
Result: ✅ Reuse existing transcripts (coverage complete)
```

**Scenario 2: Incomplete Transcript Data**
```
Video duration: 90.0s (full video being processed now)
Found records: 20 segments (previous processing stopped early)
Time coverage: 0.0s → 58.3s
Max end_time: 58.3s < (90.0 - 2.0) ❌
Missing: 31.7s at end
Result: ⚠️ Reprocess transcript from scratch with new session_id
```

**Scenario 3: Video with Silence at End**
```
Video duration: 90.0s
Found records: 18 segments (speech ends at 88s, rest is silence)
Time coverage: 0.0s → 88.0s
Max end_time: 88.0s >= (90.0 - 2.0) ✅
Result: ✅ Reuse existing transcripts (within tolerance)
```

### Benefits

- **Handles Variable Segments**: Works correctly with Whisper's dynamic segmentation
- **Ignores Empty Records**: Empty/silent segments don't break validation
- **Avoids Redundant Processing**: Saves time by reusing complete transcript data
- **Ensures Data Integrity**: Detects and fixes incomplete transcript coverage
- **Automatic Recovery**: If previous processing was interrupted, system automatically reprocesses
- **Configurable Tolerance**: Allows for silence at end of videos
- **Session Tracking**: New session_id is generated when reprocessing

### Configuration

Transcript validation is automatic when database is enabled:

```yaml
database:
  enabled: true
  store_transcript_video: true

detection:
  transcript_model: whisper
  transcript_settings:
    enabled: true
    chunk_duration: 15          # Duration for processing chunks (not segment size)
    validation_tolerance: 2.0   # Tolerance for coverage validation (seconds)
```

### Logging Output

When checking existing transcripts, you'll see detailed information:

```
🔍 Checking for existing transcripts for: cam_1.mp4
📹 Video duration: 90.23s
📊 Validation tolerance: 2.0s
✅ Found 33 existing transcript records in database
📊 Time coverage: 0.0s → 89.8s (89.8s)
📝 Segments: 25 with text, 8 empty/silent
✅ Transcript coverage is complete (99.5%)
✅ Loaded 33 transcript segments from database
```

---

## Bad Gesture Detection

```yaml
bad_gestures:
  detect_low_wrists: true
  detect_turtle_neck: true
  detect_hunched_back: true
  detect_fingers_pointing_up: true

  thresholds:
    low_wrist_threshold: 0.1
    turtle_neck_angle: 30
    finger_vertical_angle: 45
    spine_vertical_angle: 20
    upper_lower_spine_angle: 160
```

---

## Troubleshooting

### Issue: Database connection fails

**Solution:**
```yaml
# For local PostgreSQL
database:
  use_database_type: local
  local:
    host: localhost
    port: 5432
    name: musician-tracking
    user: your_username
    password: your_password  # Can be empty string

# For Supabase
database:
  use_database_type: supabase
  supabase:
    url: 'https://your-project.supabase.co'
    anon_key: 'your-anon-key'
```

### Issue: Videos not aligning correctly

**Check:**
1. `video_aligner.alignment.enable_chunk_processing` matches your video structure
2. Audio extraction duration is sufficient: `video_aligner.audio.extraction_duration`
3. Correlation threshold is not too high: `video_aligner.alignment.correlation_threshold`

### Issue: Detection taking too long

**Solutions:**
```yaml
# Limit processing duration
integrated_processor:
  limit_processing_duration: true
  max_processing_duration: 30  # Process only 30 seconds

# Disable heavy models
detection:
  emotion_model: none
  transcript_model: none

# Skip frames
video:
  skip_frames: 1  # Process every other frame
```

### Issue: No alignment data saved to database

**Check:**
```yaml
database:
  enabled: true
  store_chunk_video_alignment: true  # For chunk videos
```

And ensure database connection is successful (check console output).

---

## Output Files

### Aligned Videos
- **Location**: `src/output/aligned_videos/`
- **Format**: `YYYYMMDD_HHMMSS_<source>_<camera>.mp4`
- **Example**: `20250112_143022_vid_shot1_cam_1.mp4`

### Detection Videos
- **Location**: `src/output/annotated_detection_videos/`
- **Format**: `YYYYMMDD_HHMMSS_detection_<source>_<camera>.mp4`
- **Example**: `20250112_143022_detection_vid_shot1_cam_1.mp4`

### Unified Videos
- **Location**: `src/output/unified_videos/`
- **Format**: `YYYYMMDD_HHMMSS_unified_detection_output.mp4`
- **Example**: `20250112_143022_unified_detection_output.mp4`

---

## Database Tables

### chunk_video_alignment_offsets
Stores alignment data for multi-chunk videos:
- `source`: Video session (e.g., 'vid_shot1')
- `chunk_filename`: Chunk file name (e.g., 'cam_1_2.mp4')
- `camera_prefix`: Camera identifier (e.g., 'cam_1')
- `chunk_order`: Chunk sequence number
- `start_time_offset`: Timeline position in seconds
- `chunk_duration`: Duration in seconds
- `method_type`: 'earliest_start' or 'latest_start'

### video_alignment_offsets
Stores alignment data for single-file videos (legacy):
- `source`: Video identifier
- `start_time_offset`: Offset in seconds
- `matching_duration`: Duration of matched content
- `camera_type`: Camera type identifier

### musician_frame_analysis
Stores detection results per frame:
- Landmarks (hands, pose, face)
- Emotion scores
- Bad gesture flags
- Timestamps (original and synced)

---

## Summary

| Feature | Config Key | Options |
|---------|-----------|---------|
| **Database Type** | `database.use_database_type` | `local`, `supabase` |
| **Chunk Processing** | `video_aligner.alignment.enable_chunk_processing` | `true`, `false` |
| **Processing Type** | `integrated_processor.processing_type` | `use_offset`, `full_frames` |
| **Duration Limit** | `integrated_processor.limit_processing_duration` | `true`, `false` |
| **Run Detection** | `integrated_processor.run_detection` | `true`, `false` |
| **Unified Videos** | `integrated_processor.unified_videos` | `true`, `false` |
| **Stack Direction** | `integrated_processor.unified_video_config.stack_direction` | `vertical`, `horizontal` |

---

---

## Practical Usage Examples

### Example 1: Quick Test (30 seconds, database enabled)

**Goal**: Test the pipeline quickly with a short duration

```bash
# Command
python src/integrated_video_processor.py --max-duration 30

# What happens:
# 1. Checks database for existing alignment
# 2. Processes only first 30 seconds of each video
# 3. Runs all detection models
# 4. Creates annotated videos
# 5. Creates unified video
# 6. Saves all data to database
```

**Config** (`src/config/config_v1.yaml`):
```yaml
integrated_processor:
  processing_type: use_offset
  unified_videos: true
  detector_config:
    database:
      enabled: true
```

---

### Example 2: Full Analysis (Complete Videos)

**Goal**: Process entire videos from start to finish

```bash
# Command
python src/integrated_video_processor.py --no-duration-limit

# What happens:
# 1. Checks database for alignment (uses earliest_start)
# 2. Processes entire videos (full_frames mode)
# 3. Runs detection on all frames
# 4. Generates unified video with complete timeline
```

**Config**:
```yaml
integrated_processor:
  processing_type: full_frames  # Process entire videos
  limit_processing_duration: false
  unified_videos: true
```

---

### Example 3: Alignment Only (No Detection)

**Goal**: Just align videos and save to database

```bash
# Command
python src/integrated_video_processor.py --skip-detection

# What happens:
# 1. Scans videos and performs audio alignment
# 2. Saves alignment data to database
# 3. Creates aligned video files (if save_output_videos=true)
# 4. Skips all detection processing
# 5. No unified video creation
```

---

### Example 4: Detection Without Database

**Goal**: Run detection and create videos, but don't save to database

```bash
# Command
python src/integrated_video_processor.py

# Config
```yaml
integrated_processor:
  detector_config:
    database:
      enabled: false  # Disable database storage
    video:
      save_output_video: true
      preserve_audio: true
```

---

### Example 5: Process Different Video Set

**Goal**: Process a different set of videos without editing config

```bash
# Process vid_shot2 instead of vid_shot1
python src/integrated_video_processor.py \
  --alignment-dir /path/to/vid_shot2/original_video

# Uses all other settings from config
```

---

### Example 6: Chunk Video Processing

**Goal**: Process multi-chunk videos (cam_1_1.mp4, cam_1_2.mp4, cam_1_3.mp4)

```bash
# Command
python src/integrated_video_processor.py

# Config
```yaml
video_aligner:
  alignment:
    enable_chunk_processing: true  # Enable chunk support
  alignment_directory: /path/to/chunked_videos

# Directory structure:
# /path/to/chunked_videos/
#   ├── cam_1_1.mp4
#   ├── cam_1_2.mp4
#   ├── cam_1_3.mp4
#   ├── cam_2_1.mp4
#   ├── cam_2_2.mp4
#   └── cam_2_3.mp4

# What happens:
# 1. Groups chunks by camera (cam_1, cam_2)
# 2. Aligns each chunk to reference timeline
# 3. Merges chunks into continuous videos
# 4. Runs detection on merged videos
```

---

### Example 7: Horizontal Unified Video

**Goal**: Create side-by-side multi-camera view

```bash
# Command
python src/integrated_video_processor.py

# Config
```yaml
integrated_processor:
  unified_videos: true
  unified_video_config:
    stack_direction: horizontal  # Side-by-side
    add_camera_labels: true

# Output:
# ┌─────────────┬─────────────┬─────────────┐
# │   cam_1     │   cam_2     │   cam_3     │
# └─────────────┴─────────────┴─────────────┘
```

---

### Example 8: Custom Configuration File

**Goal**: Use a separate config for different processing scenarios

```bash
# Create custom config
cp src/config/config_v1.yaml my_test_config.yaml

# Edit my_test_config.yaml:
# - Set different detection models
# - Adjust thresholds
# - Change output directories

# Run with custom config
python src/integrated_video_processor.py --config my_test_config.yaml
```

---

## Real-World Use Cases

### Use Case 1: Music Performance Analysis

```bash
# Setup
- 3 cameras filming pianist from different angles
- Videos: cam_1.mp4, cam_2.mp4, cam_3.mp4
- Goal: Detect bad posture and analyze performance

# Command
python src/integrated_video_processor.py --no-duration-limit

# Config highlights:
```yaml
detection:
  pose_model: mediapipe
  hand_model: mediapipe
  transcript_model: none  # No speech in music

bad_gestures:
  detect_low_wrists: true
  detect_hunched_back: true
  detect_turtle_neck: true

integrated_processor:
  processing_type: full_frames
  unified_videos: true
  unified_video_config:
    stack_direction: vertical
```

# Output:
# - Individual annotated videos showing posture issues
# - Unified video with all 3 camera views
# - Database records of bad gestures per frame
```

---

### Use Case 2: Quick Test Before Long Processing

```bash
# Test with first 10 seconds
python src/integrated_video_processor.py --max-duration 10

# Verify:
# 1. Alignment is correct
# 2. Detection models working
# 3. Output videos look good

# If satisfied, run full processing
python src/integrated_video_processor.py --no-duration-limit
```

---

### Use Case 3: Re-process Detection Without Re-alignment

```bash
# First run: Alignment + Detection
python src/integrated_video_processor.py

# Alignment data now cached in database

# Later: Change detection settings and re-run
# (Uses cached alignment, no re-alignment needed)
python src/integrated_video_processor.py

# The system automatically:
# 1. Finds existing alignment in database
# 2. Skips alignment step
# 3. Runs detection with new settings
```

---

## Component Details

### 1. Video Aligner (`src/video_aligner/shape_based_aligner_multi.py`)

**Purpose**: Synchronize multi-camera videos using audio pattern matching

**Key Functions**:
- `scan_and_group_chunk_videos()` - Groups videos by camera prefix
- `determine_reference_camera_group_by_audio_pattern()` - Selects reference
- `align_chunks_to_reference_timeline()` - Calculates alignment offsets
- `combine_chunk_videos_timeline_based()` - Merges chunk videos

**Algorithm**:
1. Extract audio from first chunk of each camera
2. Calculate RMS energy profiles
3. Use cross-correlation to find time offsets
4. Select reference camera (earliest or latest start)
5. Align all cameras to reference timeline

---

### 2. Detection Processor (`src/detect_v2_3d.py`)

**Purpose**: Multi-modal detection and analysis

**Models Supported**:
- **Pose**: MediaPipe Pose (3D world landmarks), YOLO11 Pose
- **Hands**: MediaPipe Hands (3D world landmarks), YOLO11 Hands
- **Face**: YOLO Face Detection + MediaPipe Face Mesh
- **Emotion**: DeepFace, GhostFaceNet
- **Gestures**: 3D Bad Gesture Detector (turtle neck, low wrists, hunched back, pinky up)
- **Transcription**: OpenAI Whisper (with database caching)

**Key Features**:
- 3D world landmarks for pose/hand detection
- Bad gesture detection using 3D geometry
- Transcript validation and reuse from database
- Frame-level database storage
- Audio preservation in output videos

---

### 3. Database Integration (`src/database/setup.py`)

**Tables**:

**`chunk_video_alignment_offsets`** - Multi-chunk alignment data
```sql
- source (video session name)
- chunk_filename (e.g., cam_1_2.mp4)
- camera_prefix (e.g., cam_1)
- chunk_order (sequence number)
- start_time_offset (timeline position in seconds)
- chunk_duration (duration in seconds)
- method_type (earliest_start or latest_start)
```

**`musician_frame_analysis`** - Detection results per frame
```sql
- session_id, frame_number, video_file
- original_time, synced_time
- left_hand_landmarks, right_hand_landmarks
- pose_landmarks, facemesh_landmarks
- emotions (JSON)
- bad_gestures (JSON)
- transcript_segment_id (links to transcript_video)
```

**`transcript_video`** - Transcription segments
```sql
- video_file, session_id, segment_id
- start_time, end_time, duration
- text, word_count, words_json
- language, model_size
```

---

## Troubleshooting

### Issue: "No alignment data found in database"

**Solution**: Run alignment analysis first
```bash
# This will analyze and save to database
python src/integrated_video_processor.py --skip-detection
```

---

### Issue: FFmpeg not found

**Solution**: Install FFmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

---

### Issue: Database connection failed

**Solution**: Check database configuration
```yaml
database:
  enabled: true
  use_database_type: local  # or 'supabase'

  local:
    host: localhost
    port: 5432
    name: musician-tracking
    user: your_username
    password: ''  # Empty string if no password
```

---

### Issue: Transcription taking too long

**Solution**: Use smaller model or disable
```yaml
detection:
  transcript_model: whisper
  transcript_settings:
    model_size: tiny  # Fastest (use tiny, base, small, medium, large)

# Or disable completely:
detection:
  transcript_model: none
```

---

### Issue: Out of memory during processing

**Solution**: Enable duration limiting or skip frames
```yaml
integrated_processor:
  limit_processing_duration: true
  max_processing_duration: 60  # Process only 60s

video:
  skip_frames: 1  # Process every other frame
```

---

## Performance Tips

1. **Use duration limiting for testing**
   ```bash
   --max-duration 10  # Test with 10s first
   ```

2. **Disable heavy models during testing**
   ```yaml
   detection:
     emotion_model: none
     transcript_model: none
   ```

3. **Use database caching**
   - Alignment data cached automatically
   - Transcripts cached and reused
   - No re-processing needed

4. **Process videos in batches**
   ```bash
   # Process each video set separately
   python src/integrated_video_processor.py --alignment-dir vid_shot1
   python src/integrated_video_processor.py --alignment-dir vid_shot2
   ```

---

## References

**Main Files**:
- Integrated Processor: `src/integrated_video_processor.py`
- Video Aligner: `src/video_aligner/shape_based_aligner_multi.py`
- Detection System: `src/detect_v2_3d.py`
- Configuration: `src/config/config_v1.yaml`
- Database Setup: `src/database/setup.py`

**Related Documentation**:
- Detection Logic: `DETECTION_LOGIC_QUICK_REFERENCE.txt`
- Bad Gestures: `src/bad_gesture/README.md`
- Emotion Detection: `src/docs/EMOTION_DETECTION_README.md`
