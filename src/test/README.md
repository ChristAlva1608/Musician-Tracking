# Test Suite Documentation

Comprehensive test suite for the Musician Tracking System.

## Directory Structure

```
test/
├── database/              # Database-related tests
│   ├── check_table_structure.py
│   └── test_transcript_database.py
│
├── model/                 # Model tests organized by type
│   ├── emotion/          # Emotion detection models
│   │   ├── test_emotion.py
│   │   ├── test_deepface_webcam.py
│   │   ├── test_fer_only.py
│   │   ├── test_ghostfacenetsv2_emotion.py
│   │   ├── test_mediapipe_emotion.py
│   │   └── test_mobilenetv2_emotion.py
│   │
│   ├── face/             # Face detection and mesh models
│   │   ├── test_face.py
│   │   ├── test_facemesh_mediapipe.py
│   │   └── test_facemesh_yolo_mediapipe.py
│   │
│   └── hand/             # Hand detection models
│       ├── test_hand.py
│       ├── test_hand_two_stage.py
│       └── test_hand_yolo11.py
│
├── pose/                  # Pose detection tests
│   ├── test_pose.py
│   └── test_pose_hand_regions.py
│
├── transcript/            # Transcript processing tests
│   └── test_transcript.py
│
├── video/                 # Video processing tests
│   ├── test_unified_video.py
│   └── test_video_stacking.py
│
└── integration/           # Integration tests
    └── (Future tests)
```

## Test Categories

### Database Tests (`database/`)
Tests for database connectivity, schema, and data operations.

**Files:**
- `check_table_structure.py` - Validates database table structure
- `test_transcript_database.py` - Tests transcript data storage and retrieval

**Usage:**
```bash
# Test transcript database
python src/test/database/test_transcript_database.py

# Query specific video
python src/test/database/test_transcript_database.py --video "cam_1.mp4"

# Check table structure
python src/test/database/check_table_structure.py
```

### Model Tests (`model/`)
Tests for various detection models organized by type.

#### Emotion Detection (`model/emotion/`)
**Files:**
- `test_emotion.py` - Generic emotion model tests
- `test_deepface_webcam.py` - DeepFace emotion detection with webcam
- `test_fer_only.py` - FER (Facial Expression Recognition) model
- `test_ghostfacenetsv2_emotion.py` - GhostFaceNet v2 emotion detection
- `test_mediapipe_emotion.py` - MediaPipe-based emotion detection
- `test_mobilenetv2_emotion.py` - MobileNetV2 emotion detection

**Usage:**
```bash
python src/test/model/emotion/test_emotion.py
python src/test/model/emotion/test_fer_only.py
python src/test/model/emotion/test_deepface_webcam.py
```

#### Face Detection (`model/face/`)
**Files:**
- `test_face.py` - Generic face detection tests
- `test_facemesh_mediapipe.py` - MediaPipe face mesh detection
- `test_facemesh_yolo_mediapipe.py` - YOLO + MediaPipe combined approach

**Usage:**
```bash
python src/test/model/face/test_face.py
python src/test/model/face/test_facemesh_mediapipe.py
python src/test/model/face/test_facemesh_yolo_mediapipe.py
```

#### Hand Detection (`model/hand/`)
**Files:**
- `test_hand.py` - Generic hand detection tests
- `test_hand_yolo11.py` - YOLO11 hand detection
- `test_hand_two_stage.py` - Two-stage hand detection (YOLO + MediaPipe)

**Usage:**
```bash
python src/test/model/hand/test_hand.py
python src/test/model/hand/test_hand_yolo11.py
python src/test/model/hand/test_hand_two_stage.py
```

### Pose Detection Tests (`pose/`)
Tests for pose estimation and body landmark detection.

**Files:**
- `test_pose.py` - Generic pose detection tests
- `test_pose_hand_regions.py` - Combined pose and hand region detection

**Usage:**
```bash
python src/test/pose/test_pose.py
python src/test/pose/test_pose_hand_regions.py
```

### Video Processing Tests (`video/`)
Tests for video processing, stacking, and unified output creation.

**Files:**
- `test_unified_video.py` - Unified video creation from multiple cameras
- `test_video_stacking.py` - Video stacking functionality

**Usage:**
```bash
python src/test/video/test_unified_video.py
python src/test/video/test_video_stacking.py
```

### Transcript Tests (`transcript/`)
Tests for transcript processing and speech recognition.

**Files:**
- `test_transcript.py` - Transcript processing and Whisper integration tests

**Usage:**
```bash
python src/test/transcript/test_transcript.py
```

### Integration Tests (`integration/`)
End-to-end integration tests combining multiple components.

**Files:**
- (Future integration tests)

## Running Tests

### Run All Tests in a Category
```bash
# All database tests
python -m pytest src/test/database/

# All model tests
python -m pytest src/test/model/

# All emotion model tests
python -m pytest src/test/model/emotion/

# All hand model tests
python -m pytest src/test/model/hand/

# All pose tests
python -m pytest src/test/pose/

# All video tests
python -m pytest src/test/video/

# All transcript tests
python -m pytest src/test/transcript/
```

### Run Specific Test
```bash
python src/test/database/test_transcript_database.py
python src/test/model/emotion/test_fer_only.py
python src/test/model/hand/test_hand_two_stage.py
python src/test/pose/test_pose.py
python src/test/transcript/test_transcript.py
```

### Run with pytest
```bash
# Run all tests
python -m pytest src/test/

# Run with verbose output
python -m pytest src/test/ -v

# Run specific test file
python -m pytest src/test/database/test_transcript_database.py
```

## Adding New Tests

### 1. Choose the Appropriate Directory
Place your test in the directory that matches its primary function:
- Database operations → `database/`
- Model testing → `model/emotion/`, `model/face/`, `model/hand/`
- Pose detection → `pose/`
- Video processing → `video/`
- Transcript processing → `transcript/`
- End-to-end workflows → `integration/`

### 2. Follow Naming Convention
- Test files: `test_*.py`
- Test functions: `def test_*():`
- Test classes: `class Test*:`

### 3. Example Test Structure
```python
#!/usr/bin/env python3
"""
Description of what this test does
"""

import sys
from pathlib import Path

# Add project root to path
parent_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(parent_dir))

# Your imports here
from src.models.hand.mediapipe import MediaPipeHandDetector

def test_hand_detection():
    """Test basic hand detection"""
    detector = MediaPipeHandDetector()
    # Your test code here
    assert detector is not None

if __name__ == "__main__":
    test_hand_detection()
    print("✅ All tests passed")
```

## Test Guidelines

1. **Keep tests focused** - Each test should test one specific functionality
2. **Use descriptive names** - Test names should clearly indicate what they test
3. **Add documentation** - Include docstrings explaining the test purpose
4. **Make tests independent** - Tests should not depend on each other
5. **Clean up resources** - Release webcam, close files, cleanup temp data
6. **Use meaningful assertions** - Check specific conditions, not just "no errors"

## Common Test Patterns

### Testing Database Operations
```python
def test_database_query():
    db = DatabaseManager()
    result = db.query_something()
    assert result is not None
    assert len(result) > 0
    db.close()
```

### Testing Model Detection
```python
def test_model_detection():
    detector = SomeDetector()
    frame = cv2.imread('test_image.jpg')
    results = detector.detect(frame)
    assert results is not None
    assert len(results) > 0
```

### Testing Video Processing
```python
def test_video_processing():
    processor = VideoProcessor()
    output = processor.process('input.mp4')
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0
```

## Troubleshooting

See `TROUBLESHOOTING.md` for common test issues and solutions.

## Quick Reference

| Test Type | Location | Example |
|-----------|----------|---------|
| Database | `database/` | `python src/test/database/test_transcript_database.py` |
| Emotion Models | `model/emotion/` | `python src/test/model/emotion/test_fer_only.py` |
| Face Models | `model/face/` | `python src/test/model/face/test_facemesh_mediapipe.py` |
| Hand Models | `model/hand/` | `python src/test/model/hand/test_hand_two_stage.py` |
| Pose Detection | `pose/` | `python src/test/pose/test_pose.py` |
| Transcripts | `transcript/` | `python src/test/transcript/test_transcript.py` |
| Video Processing | `video/` | `python src/test/video/test_unified_video.py` |
| Integration | `integration/` | (Future tests) |
