# Test Directory Organization

This document describes the reorganized test structure for the Musician Tracking project.

## Previous Structure (Before Reorganization)
```
scripts/test_script/          # Scattered test files
├── test_deepface_webcam.py
├── test_facemesh_mediapipe.py
├── test_fer_only.py
└── ... (other implementation tests)

test/                         # Mixed model and other tests
├── test_hand.py
├── test_pose.py
├── test_face.py
├── test_emotion.py
└── test_transcript.py
```

## New Structure (After Reorganization)
```
test/                         # All test files centralized
├── test_model/              # Model-specific test files
│   ├── test_emotion.py      # Emotion detection model tests
│   ├── test_face.py         # Face detection model tests
│   ├── test_hand.py         # Hand detection model tests
│   ├── test_pose.py         # Pose detection model tests
│   └── test_transcript.py   # Transcript model tests
│
├── check_table_structure.py          # Database structure tests
├── test_deepface_webcam.py           # DeepFace webcam implementation tests
├── test_facemesh_mediapipe.py        # MediaPipe FaceMesh implementation tests
├── test_facemesh_yolo_mediapipe.py   # YOLO+MediaPipe FaceMesh tests
├── test_fer_only.py                  # FER emotion detection implementation tests
├── test_ghostfacenetsv2_emotion.py   # GhostFaceNet emotion implementation tests
├── test_hand_yolo11.py               # YOLO hand detection implementation tests
├── test_mediapipe_emotion.py         # MediaPipe emotion implementation tests
├── test_mobilenetv2_emotion.py       # MobileNetV2 emotion implementation tests
├── test_unified_video.py             # Unified video creation tests
└── test_video_stacking.py            # Video stacking functionality tests
```

## Changes Made

### 1. Directory Structure
- **Created**: `test/test_model/` for model-specific tests
- **Moved**: Model tests from `test/` to `test/test_model/`
- **Moved**: Implementation tests from `scripts/test_script/` to `test/`
- **Removed**: Empty `scripts/test_script/` directory

### 2. Import Path Updates
- **Files moved from `scripts/test_script/` to `test/`**:
  - Updated path from 3 levels up to 1 level up
  - Changed: `os.path.dirname(os.path.dirname(os.path.dirname()))`
  - To: `os.path.dirname(os.path.dirname())`

- **Files moved from `test/` to `test/test_model/`**:
  - Import paths already correct (2 levels up)
  - No changes needed

### 3. Documentation Updates
- Updated `CLAUDE.md` project structure
- Updated testing commands section
- Updated notes about test organization

## Test Categories

### Model Tests (`test/test_model/`)
- **Purpose**: Test core model functionality
- **Scope**: Abstract model interfaces and implementations
- **Examples**: Hand detection, pose detection, face detection, emotion recognition, transcript generation

### Implementation Tests (`test/`)
- **Purpose**: Test specific implementation details
- **Scope**: Concrete implementations using specific libraries/frameworks
- **Examples**: MediaPipe implementations, YOLO implementations, DeepFace implementations

### System Tests (`test/`)
- **Purpose**: Test system-level functionality
- **Scope**: Integration testing, video processing workflows
- **Examples**: Unified video creation, video stacking, database structure

## Running Tests

### All Tests
```bash
python -m pytest test/
```

### Model-Specific Tests
```bash
python test/test_model/test_hand.py
python test/test_model/test_pose.py
python test/test_model/test_face.py
python test/test_model/test_emotion.py
python test/test_model/test_transcript.py
```

### Implementation Tests
```bash
python test/test_mediapipe_emotion.py
python test/test_facemesh_mediapipe.py
python test/test_hand_yolo11.py
```

### System Tests
```bash
python test/test_unified_video.py
python test/test_video_stacking.py
```

## Benefits of New Structure

1. **Clear Separation**: Model tests vs implementation tests vs system tests
2. **Centralized Location**: All tests in one place (`test/`)
3. **Organized Hierarchy**: Related tests grouped together
4. **Easier Navigation**: Clear naming and directory structure
5. **Scalable**: Easy to add new test categories