# Test Directory Structure

## Current Organization

```
test/
│
├── __init__.py                          # Test package initialization
├── README.md                            # Main documentation
├── STRUCTURE.md                         # This file
├── TROUBLESHOOTING.md                   # Common issues and solutions
│
├── database/                            # Database tests
│   ├── __init__.py
│   ├── check_table_structure.py        # Validate database schema
│   └── test_transcript_database.py     # Test transcript storage/retrieval
│
├── model/                               # Model tests by category
│   │
│   ├── emotion/                         # Emotion detection models
│   │   ├── __init__.py
│   │   ├── test_emotion.py             # Generic emotion tests
│   │   ├── test_deepface_webcam.py     # DeepFace model
│   │   ├── test_fer_only.py            # FER model
│   │   ├── test_ghostfacenetsv2_emotion.py  # GhostFaceNet v2
│   │   ├── test_mediapipe_emotion.py   # MediaPipe emotion
│   │   └── test_mobilenetv2_emotion.py # MobileNetV2
│   │
│   ├── face/                            # Face detection models
│   │   ├── __init__.py
│   │   ├── test_face.py                # Generic face tests
│   │   ├── test_facemesh_mediapipe.py  # MediaPipe FaceMesh
│   │   └── test_facemesh_yolo_mediapipe.py  # YOLO + MediaPipe
│   │
│   └── hand/                            # Hand detection models
│       ├── __init__.py
│       ├── test_hand.py                # Generic hand tests
│       ├── test_hand_two_stage.py      # Two-stage detection
│       └── test_hand_yolo11.py         # YOLO11 detection
│
├── pose/                                # Pose detection tests
│   ├── __init__.py
│   ├── test_pose.py                    # Generic pose tests
│   └── test_pose_hand_regions.py       # Pose + hand regions
│
├── transcript/                          # Transcript processing tests
│   ├── __init__.py
│   └── test_transcript.py              # Whisper transcription
│
├── video/                               # Video processing tests
│   ├── __init__.py
│   ├── test_unified_video.py           # Multi-camera unified videos
│   └── test_video_stacking.py          # Video stacking
│
└── integration/                         # Integration tests
    └── __init__.py
    └── (Future end-to-end tests)
```

## Test Count by Category

- Database: 2 tests
- Model Tests: 13 tests
  - Emotion: 6 tests
  - Face: 3 tests
  - Hand: 3 tests
  - Pose: 2 tests (includes 1 generic)
- Transcript: 1 test
- Video: 2 tests
- Integration: 0 tests (planned)

**Total: 20 test files**

## Quick Commands

```bash
# Run all tests
python -m pytest src/test/

# Run by category
python -m pytest src/test/database/
python -m pytest src/test/model/
python -m pytest src/test/model/emotion/
python -m pytest src/test/pose/
python -m pytest src/test/transcript/
python -m pytest src/test/video/

# Run specific test
python src/test/database/test_transcript_database.py
python src/test/model/emotion/test_fer_only.py
python src/test/model/hand/test_hand_two_stage.py
```
