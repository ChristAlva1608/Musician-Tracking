# Musician-Detection-Project
This project helps detect a musician's performance and track abnormal gestures when he is playing

# Landmark Detection with Supabase

This project detects hand, face, and pose landmarks from video files and stores them in a Supabase database.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Supabase Setup

1. Create a Supabase project at [supabase.com](https://supabase.com)
2. Create a table called `landmarks` with the following structure:

```sql
CREATE TABLE landmarks (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    face_landmarks TEXT,
    hand_landmarks TEXT,
    pose_landmarks TEXT
);
```

3. Create a `.env` file in the project root with your Supabase credentials:

```env
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

### 3. Video File
Place your video file in the project directory and update the path in `detect.py`:
```python
cap = cv2.VideoCapture("your_video_file.mp4")
```

### 4. Run the Application
```bash
python detect.py
```

## Features

- **Hand Landmark Detection**: Detects 21 hand landmarks per hand
- **Face Landmark Detection**: Detects 468 face landmarks
- **Pose Landmark Detection**: Detects 33 pose landmarks
- **Real-time Visualization**: Shows landmarks on video frames
- **Database Storage**: Saves all landmarks to Supabase with timestamps
- **Frame Tracking**: Displays frame count and save status

## Database Schema

| Field | Type | Description |
|-------|------|-------------|
| id | BIGSERIAL | Auto-incrementing primary key |
| created_at | TIMESTAMP | When the record was created |
| face_landmarks | TEXT | JSON string of face landmark coordinates |
| hand_landmarks | TEXT | JSON string of hand landmark coordinates |
| pose_landmarks | TEXT | JSON string of pose landmark coordinates |

## Usage

1. The application will process each frame of your video
2. Landmarks are detected and drawn on the video display
3. All landmark data is saved to Supabase database
4. Press 'q' to quit early
5. Check the console for processing summary

## Querying Data

You can query the stored landmarks using Supabase's SQL interface:

```sql
-- Get all landmarks
SELECT * FROM landmarks;

-- Get landmarks with face detection
SELECT * FROM landmarks WHERE face_landmarks IS NOT NULL;

-- Get recent landmarks
SELECT * FROM landmarks ORDER BY created_at DESC LIMIT 10;
``` 