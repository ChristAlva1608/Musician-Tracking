# Musician-Tracking

A comprehensive computer vision system for real-time posture and gesture tracking of musicians, designed to detect and analyze performance techniques, bad postures, and provide detailed motion analysis.

## 🎯 Project Overview

This project provides an advanced tracking system specifically designed for musicians, offering:
- Multi-person pose detection and tracking with consistent ID assignment
- Hand gesture recognition with person-specific association
- Face detection and emotion analysis
- Bad posture detection (hunched back, turtle neck, low wrists)
- Multi-camera video analysis with layout change detection
- Real-time and batch video processing capabilities
- 3D world coordinate extraction for spatial analysis

The system uses state-of-the-art computer vision models including YOLO, MediaPipe, and custom detection algorithms to provide comprehensive musician performance analysis.

## 🚀 Core Features

### 1. **Multi-Person Tracking**
- Consistent person ID tracking across frames using centroid tracking
- Temporal history buffer for robust tracking through occlusions
- Support for multiple musicians in the same frame

### 2. **Hand Detection & Matching**
- MediaPipe Hands integration for 21-landmark hand detection
- Spatial matching of hands to specific people based on wrist proximity
- Multi-tier matching strategy with temporal fallback
- Preservation of unmatched hands for complete data capture

### 3. **Pose Analysis**
- YOLO11 and MediaPipe Pose support
- 33 pose landmarks with 3D world coordinates
- Bad gesture detection:
  - Low wrist position
  - Turtle neck posture
  - Hunched back detection
  - Finger pointing analysis

### 4. **Face & Emotion Detection**
- YOLO face detection with MediaPipe FaceMesh (478 landmarks)
- Multiple emotion detection models (FER, DeepFace, GhostFaceNet)
- Person-specific face association

### 5. **Video Processing**
- Real-time webcam processing
- Batch video file processing
- Multi-camera layout detection (grid vs fullscreen)
- Video alignment for synchronized multi-angle analysis
- 360° video conversion support

### 6. **Data Management**
- PostgreSQL database integration
- Batch data insertion with optimization
- Session-based tracking
- Comprehensive logging and analysis reports

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8 or higher
- PostgreSQL 12 or higher
- Node.js 16+ and npm (for web interface)
- CUDA-capable GPU (optional, for acceleration)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Musician-Tracking.git
cd Musician-Tracking
```

2. **Create and activate virtual environment**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Install additional model dependencies**
```bash
# MediaPipe
pip install mediapipe

# YOLO
pip install ultralytics

# Database
pip install psycopg2-binary sqlalchemy

# Video processing
pip install opencv-python-headless numpy

# Emotion detection (optional)
pip install fer deepface tensorflow
```

## 🗄️ PostgreSQL Installation

### macOS
```bash
# Using Homebrew
brew install postgresql
brew services start postgresql

# Create database
createdb musician_tracking
```

### Ubuntu/Debian
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres createdb musician_tracking
```

### Windows
1. Download PostgreSQL installer from https://www.postgresql.org/download/windows/
2. Run the installer and follow the setup wizard
3. Create database using pgAdmin or command line:
```bash
createdb musician_tracking
```

### Database Configuration
1. Create a `.env` file in the project root:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=musician_tracking
DB_USER=your_username
DB_PASSWORD=your_password
```

2. Initialize the database schema:
```bash
python src/database/database_setup_v2.py
```

## 📁 Folder Structure

```
Musician-Tracking/
├── src/
│   ├── detect_v2_2d.py          # 2D detection pipeline
│   ├── detect_v2_3d.py          # 3D detection pipeline with world coordinates
│   ├── integrated_video_processor.py  # Unified video processing
│   ├── config/
│   │   └── config_v1.yaml       # Configuration files
│   ├── database/
│   │   ├── database_setup_v2.py # PostgreSQL setup and management
│   │   └── migrations/          # Database migration scripts
│   ├── models/
│   │   ├── emotion/             # Emotion detection models
│   │   ├── facemesh/            # Face mesh detection
│   │   ├── hand/                # Hand detection models
│   │   └── pose/                # Pose detection models
│   ├── utils/
│   │   ├── hand_person_matcher.py          # Basic hand-person matching
│   │   ├── hand_person_matcher_enhanced.py # Enhanced matcher with temporal
│   │   ├── person_tracker_enhanced.py      # Multi-person tracking
│   │   └── bad_gesture_detector.py         # Bad posture detection
│   ├── screen_state_detector/
│   │   ├── screen_state_detector.py        # Multi-camera layout detection
│   │   └── ScreenStateDetector.tsx         # React component
│   ├── video_aligner/
│   │   └── shape_based_aligner_multi.py    # Multi-video alignment
│   ├── web/
│   │   ├── backend/             # FastAPI backend
│   │   │   ├── main.py
│   │   │   └── routers/
│   │   └── frontend/            # React frontend
│   │       ├── package.json
│   │       └── src/
│   └── analysis/
│       └── analysis_report/      # Generated analysis reports
├── checkpoints/                  # Model weights (YOLO)
├── video/                        # Sample videos for testing
├── scripts/
│   ├── run_2person_detection.sh # Multi-person detection script
│   └── profile_performance.py   # Performance profiling
├── test/                         # Unit and integration tests
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Development guidelines
└── README.md                     # This file
```

## 💻 Usage

### Basic Detection (Single Person)
```bash
# Run with webcam
python src/detect_v2_3d.py --webcam

# Run with video file
python src/detect_v2_3d.py --video path/to/video.mp4
```

### Multi-Person Detection (2D)
```bash
python src/detect_v2_2d.py --video path/to/video.mp4 --save-db
```

### Multi-Person Detection with 3D Coordinates
```bash
python src/detect_v2_3d.py --video path/to/video.mp4 --config src/config/config_v1.yaml
```

### Batch Processing Multiple Videos
```bash
./scripts/run_2person_detection.sh
```

### Screen State Detection (Multi-camera Analysis)
```bash
# Standalone processing
python src/screen_state_detector/screen_state_detector.py video/multicam.mp4 --preview

# Without preview
python src/screen_state_detector/screen_state_detector.py video/multicam.mp4
```

### Web Interface

#### Start Backend Server
```bash
cd src/web/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Start Frontend (in new terminal)
```bash
cd src/web/frontend
npm install  # First time only
npm start
```

Access the web interface at http://localhost:3000

### Configuration Options

Edit `src/config/config_v1.yaml` to customize:
- Model selection (YOLO/MediaPipe)
- Detection thresholds
- Database settings
- Bad gesture detection parameters
- Video processing options

### Command Line Arguments

```bash
python src/detect_v2_3d.py --help

Options:
  --video PATH         Path to input video file
  --webcam            Use webcam instead of video file
  --config PATH       Path to configuration file
  --save-db          Save results to database
  --no-display       Run without GUI display
  --output PATH      Save processed video to file
  --skip-frames N    Process every Nth frame
```

## 🔬 Advanced Features

### Custom Bad Gesture Detection
Modify thresholds in `src/utils/bad_gesture_detector.py`:
```python
LOW_WRIST_THRESHOLD = 0.15
TURTLE_NECK_THRESHOLD = 0.12
HUNCHBACK_ANGLE_THRESHOLD = 150
```

### Database Queries
```python
from src.database.database_setup_v2 import DatabaseManager

db = DatabaseManager()
# Get latest session
session_data = db.get_latest_session()
# Query specific person's data
person_data = db.get_person_landmarks(session_id, person_id)
```

### Emotion Detection Models
Enable different emotion models by modifying detection pipeline:
```python
# In detect_v2_3d.py
self.emotion_model = "fer"  # Options: "fer", "deepface", "ghostfacenet"
```

## 🧪 Testing

Run test suite:
```bash
# All tests
python -m pytest test/

# Specific module tests
python test/test_model/test_hand.py
python test/test_model/test_pose.py
python test/test_model/test_emotion.py

# Performance profiling
python scripts/profile_performance.py
```

## 📊 Performance

- **Real-time Processing**: 15-30 FPS on modern GPU
- **Batch Processing**: ~5-10 FPS with full pipeline
- **Database Operations**: Optimized batch insertion (100 frames/batch)
- **Memory Usage**: ~2-4GB RAM for typical session

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see below:

```
MIT License

Copyright (c) 2024 Musician-Tracking Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- MediaPipe team for pose and hand detection models
- Ultralytics for YOLO implementation
- OpenCV community for computer vision tools
- All contributors and testers

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---
*This project is under active development. Features and APIs may change.*