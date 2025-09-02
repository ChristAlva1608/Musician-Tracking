import os
from supabase import create_client, Client
from typing import Optional, List, Dict, Any
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MusicianDatabase:
    def __init__(self):
        """Initialize Supabase connection"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables."
            )
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Batch insert configuration
        self.batch_size = 50  # Number of frames to batch before inserting
        self.batch_data = []  # Buffer to store frame data
        self.batch_timeout = 5.0  # Maximum time to wait before forcing insert (seconds)
        self.last_batch_time = time.time()
        
        print(f"✅ Connected to Supabase: {self.supabase_url}")
        print(f"📦 Batch insert enabled: {self.batch_size} frames per batch, {self.batch_timeout}s timeout")
    
    def create_table(self) -> bool:
        """
        Create both musician_frame_analysis and video_alignment_offsets tables in Supabase
        Note: This uses raw SQL execution via Supabase's RPC function
        """
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS musician_frame_analysis (
            id BIGSERIAL PRIMARY KEY,
            
            -- Session metadata
            session_id VARCHAR(50) NOT NULL,
            video_file VARCHAR(255),
            frame_number INTEGER NOT NULL,
            original_time DECIMAL(10,3),  -- Original time in video (seconds)
            synced_time DECIMAL(10,3),    -- Synced time after alignment (seconds)
            fps REAL,
            
            -- Hand landmarks (JSON for flexibility)
            left_hand_landmarks JSONB,   -- [{x, y, z, confidence}, ...] 21 points
            right_hand_landmarks JSONB,  -- [{x, y, z, confidence}, ...] 21 points
            
            -- Pose landmarks (JSON)
            pose_landmarks JSONB,        -- [{x, y, z, confidence}, ...] 17-33 points
            
            -- Emotion scores (0.0 - 1.0)
            emotion_angry DECIMAL(4,3) DEFAULT 0.000,
            emotion_disgust DECIMAL(4,3) DEFAULT 0.000,
            emotion_fear DECIMAL(4,3) DEFAULT 0.000,
            emotion_happy DECIMAL(4,3) DEFAULT 0.000,
            emotion_sad DECIMAL(4,3) DEFAULT 0.000,
            emotion_surprise DECIMAL(4,3) DEFAULT 0.000,
            emotion_neutral DECIMAL(4,3) DEFAULT 0.000,
            
            -- Bad gesture detection flags
            flag_low_wrists BOOLEAN DEFAULT FALSE,
            flag_turtle_neck BOOLEAN DEFAULT FALSE,
            flag_hunched_back BOOLEAN DEFAULT FALSE,
            flag_fingers_pointing_up BOOLEAN DEFAULT FALSE,
            
            -- Analysis metadata
            processing_time_ms INTEGER,
            model_version VARCHAR(20),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_session_frame ON musician_frame_analysis(session_id, frame_number);
        CREATE INDEX IF NOT EXISTS idx_session_time ON musician_frame_analysis(session_id, synced_time);
        CREATE INDEX IF NOT EXISTS idx_bad_gestures ON musician_frame_analysis(flag_low_wrists, flag_turtle_neck, flag_hunched_back, flag_fingers_pointing_up);
        CREATE INDEX IF NOT EXISTS idx_created_at ON musician_frame_analysis(created_at);
        
        -- Video Alignment Offsets Table
        CREATE TABLE IF NOT EXISTS video_alignment_offsets (
            id BIGSERIAL PRIMARY KEY,
            video_name VARCHAR(255) NOT NULL UNIQUE,
            start_time_offset DECIMAL(10,3) NOT NULL DEFAULT 0.000,  -- offset in seconds
            matching_duration DECIMAL(10,3) NOT NULL DEFAULT 0.000,  -- duration of matching content in seconds
            reference_video VARCHAR(255),  -- which video was used as reference
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for video alignment table
        CREATE INDEX IF NOT EXISTS idx_video_name ON video_alignment_offsets(video_name);
        CREATE INDEX IF NOT EXISTS idx_reference_video ON video_alignment_offsets(reference_video);
        """
        
        try:
            # Execute the SQL using Supabase's rpc function
            # Note: You may need to create a PostgreSQL function in Supabase first
            result = self.supabase.rpc('exec_sql', {'sql': create_table_sql}).execute()
            print("✅ Tables 'musician_frame_analysis' and 'video_alignment_offsets' created successfully!")
            return True
        except Exception as e:
            # Fallback: Try using direct table creation (if you have sufficient permissions)
            print(f"❌ RPC method failed: {e}")
            print("💡 Please create the tables manually in Supabase SQL Editor using the provided SQL.")
            return False
    
    def add_frame_to_batch(self, 
                          session_id: str,
                          frame_number: int,
                          video_file: Optional[str] = None,
                          original_time: Optional[float] = None,
                          synced_time: Optional[float] = None,
                          left_hand_landmarks: Optional[list] = None,
                          right_hand_landmarks: Optional[list] = None,
                          pose_landmarks: Optional[list] = None,
                          facemesh_landmarks: Optional[list] = None,
                          emotions: Optional[dict] = None,
                          bad_gestures: Optional[dict] = None,
                          processing_time_ms: Optional[int] = None,
                          hand_processing_time_ms: Optional[int] = None,
                          pose_processing_time_ms: Optional[int] = None,
                          facemesh_processing_time_ms: Optional[int] = None,
                          emotion_processing_time_ms: Optional[int] = None,
                          hand_model: Optional[str] = None,
                          pose_model: Optional[str] = None,
                          facemesh_model: Optional[str] = None,
                          emotion_model: Optional[str] = None) -> bool:
        """
        Add frame data to batch buffer for later bulk insert
        
        Returns True if batch was automatically flushed, False if just added to buffer
        """
        # Prepare data for insertion
        data = {
            'session_id': session_id,
            'frame_number': frame_number,
            'video_file': video_file,
            'original_time': original_time,
            'synced_time': synced_time,
            'left_hand_landmarks': left_hand_landmarks,
            'right_hand_landmarks': right_hand_landmarks,
            'pose_landmarks': pose_landmarks,
            'facemesh_landmarks': facemesh_landmarks,
            'processing_time_ms': processing_time_ms,
            'hand_processing_time_ms': hand_processing_time_ms or 0,
            'pose_processing_time_ms': pose_processing_time_ms or 0,
            'facemesh_processing_time_ms': facemesh_processing_time_ms or 0,
            'emotion_processing_time_ms': emotion_processing_time_ms or 0,
            'hand_model': hand_model,
            'pose_model': pose_model,
            'facemesh_model': facemesh_model,
            'emotion_model': emotion_model
        }
        
        # Add emotion scores
        if emotions:
            data.update({
                'emotion_angry': emotions.get('angry', 0.0),
                'emotion_disgust': emotions.get('disgust', 0.0),
                'emotion_fear': emotions.get('fear', 0.0),
                'emotion_happy': emotions.get('happy', 0.0),
                'emotion_sad': emotions.get('sad', 0.0),
                'emotion_surprise': emotions.get('surprise', 0.0),
                'emotion_neutral': emotions.get('neutral', 0.0)
            })
        
        # Add bad gesture flags
        if bad_gestures:
            data.update({
                'flag_low_wrists': bad_gestures.get('low_wrists', False),
                'flag_turtle_neck': bad_gestures.get('turtle_neck', False),
                'flag_hunched_back': bad_gestures.get('hunched_back', False),
                'flag_fingers_pointing_up': bad_gestures.get('fingers_pointing_up', False)
            })
        
        # Add to batch
        self.batch_data.append(data)
        
        # Check if we should flush the batch
        current_time = time.time()
        should_flush = (
            len(self.batch_data) >= self.batch_size or 
            (current_time - self.last_batch_time) >= self.batch_timeout
        )
        
        if should_flush:
            return self.flush_batch()
        
        return False
    
    def flush_batch(self) -> bool:
        """
        Force flush all batched data to database
        
        Returns True if successful, False otherwise
        """
        if not self.batch_data:
            return True
        
        try:
            batch_start_time = time.time()
            
            # Insert all batched data at once
            response = self.supabase.table('musician_frame_analysis').insert(self.batch_data).execute()
            
            batch_duration = (time.time() - batch_start_time) * 1000
            frames_inserted = len(self.batch_data)
            avg_time_per_frame = batch_duration / frames_inserted if frames_inserted > 0 else 0
            
            print(f"📦 Batch insert: {frames_inserted} frames in {batch_duration:.0f}ms ({avg_time_per_frame:.1f}ms/frame)")
            
            # Clear batch and reset timer
            self.batch_data.clear()
            self.last_batch_time = time.time()
            
            return True
            
        except Exception as e:
            print(f"❌ Batch insert failed: {e}")
            # Keep the batch data for retry
            return False
    
    def close(self):
        """
        Close database connection and flush any remaining batch data
        """
        if self.batch_data:
            print(f"📦 Flushing remaining {len(self.batch_data)} frames before closing...")
            self.flush_batch()
        print("📪 Database connection closed")
    
    def insert_frame_data(self, 
                         session_id: str,
                         frame_number: int,
                         video_file: Optional[str] = None,
                         original_time: Optional[float] = None,
                         synced_time: Optional[float] = None,
                         left_hand_landmarks: Optional[list] = None,
                         right_hand_landmarks: Optional[list] = None,
                         pose_landmarks: Optional[list] = None,
                         facemesh_landmarks: Optional[list] = None,
                         emotions: Optional[dict] = None,
                         bad_gestures: Optional[dict] = None,
                         processing_time_ms: Optional[int] = None,
                         hand_processing_time_ms: Optional[int] = None,
                         pose_processing_time_ms: Optional[int] = None,
                         facemesh_processing_time_ms: Optional[int] = None,
                         emotion_processing_time_ms: Optional[int] = None,
                         hand_model: Optional[str] = None,
                         pose_model: Optional[str] = None,
                         facemesh_model: Optional[str] = None,
                         emotion_model: Optional[str] = None) -> bool:
        """
        Insert frame analysis data into the database
        
        Args:
            session_id: Unique identifier for the session
            frame_number: Frame number in the video
            video_file: Path to the video file
            original_time: Original time in video (seconds)
            synced_time: Synced time after alignment (seconds)
            left_hand_landmarks: List of left hand landmark points
            right_hand_landmarks: List of right hand landmark points
            pose_landmarks: List of pose landmark points
            facemesh_landmarks: List of face mesh landmark points
            emotions: Dictionary of emotion scores
            bad_gestures: Dictionary of bad gesture flags
            processing_time_ms: Total time taken to process this frame
            hand_processing_time_ms: Time taken for hand detection
            pose_processing_time_ms: Time taken for pose detection
            facemesh_processing_time_ms: Time taken for face mesh detection
            emotion_processing_time_ms: Time taken for emotion detection
            hand_model: Hand detection model used (e.g., "mediapipe")
            pose_model: Pose detection model used (e.g., "yolo", "mediapipe")
            facemesh_model: Face mesh model used (e.g., "mediapipe", "yolo")
            emotion_model: Emotion detection model used (e.g., "deepface", "fer")
        """
        
        # Prepare data for insertion
        data = {
            'session_id': session_id,
            'frame_number': frame_number,
            'video_file': video_file,
            'original_time': original_time,
            'synced_time': synced_time,
            'left_hand_landmarks': left_hand_landmarks,
            'right_hand_landmarks': right_hand_landmarks,
            'pose_landmarks': pose_landmarks,
            'facemesh_landmarks': facemesh_landmarks,
            'processing_time_ms': processing_time_ms,
            'hand_processing_time_ms': hand_processing_time_ms or 0,
            'pose_processing_time_ms': pose_processing_time_ms or 0,
            'facemesh_processing_time_ms': facemesh_processing_time_ms or 0,
            'emotion_processing_time_ms': emotion_processing_time_ms or 0,
            'hand_model': hand_model,
            'pose_model': pose_model,
            'facemesh_model': facemesh_model,
            'emotion_model': emotion_model
        }
        
        # Add emotion scores
        if emotions:
            data.update({
                'emotion_angry': emotions.get('angry', 0.0),
                'emotion_disgust': emotions.get('disgust', 0.0),
                'emotion_fear': emotions.get('fear', 0.0),
                'emotion_happy': emotions.get('happy', 0.0),
                'emotion_sad': emotions.get('sad', 0.0),
                'emotion_surprise': emotions.get('surprise', 0.0),
                'emotion_neutral': emotions.get('neutral', 0.0)
            })
        
        # Add bad gesture flags
        if bad_gestures:
            data.update({
                'flag_low_wrists': bad_gestures.get('low_wrists', False),
                'flag_turtle_neck': bad_gestures.get('turtle_neck', False),
                'flag_hunched_back': bad_gestures.get('hunched_back', False),
                'flag_fingers_pointing_up': bad_gestures.get('fingers_pointing_up', False)
            })
        
        try:
            result = self.supabase.table('musician_frame_analysis').insert(data).execute()
            return True
        except Exception as e:
            print(f"❌ Error inserting frame data: {e}")
            return False
    
    def get_session_data(self, session_id: str, limit: Optional[int] = None):
        """
        Retrieve all frame data for a specific session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of records to retrieve
        """
        try:
            query = self.supabase.table('musician_frame_analysis').select('*').eq('session_id', session_id).order('frame_number')
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            return result.data
        except Exception as e:
            print(f"❌ Error retrieving session data: {e}")
            return None
    
    def get_bad_gesture_summary(self, session_id: str):
        """
        Get summary of bad gestures for a session
        """
        try:
            result = self.supabase.table('musician_frame_analysis').select(
                'flag_low_wrists, flag_turtle_neck, flag_hunched_back, flag_fingers_pointing_up'
            ).eq('session_id', session_id).execute()
            
            if result.data:
                summary = {
                    'low_wrists': sum(1 for row in result.data if row['flag_low_wrists']),
                    'turtle_neck': sum(1 for row in result.data if row['flag_turtle_neck']),
                    'hunched_back': sum(1 for row in result.data if row['flag_hunched_back']),
                    'fingers_pointing_up': sum(1 for row in result.data if row['flag_fingers_pointing_up']),
                    'total_frames': len(result.data)
                }
                return summary
            return None
        except Exception as e:
            print(f"❌ Error getting bad gesture summary: {e}")
            return None
    
    def get_emotion_summary(self, session_id: str):
        """
        Get average emotion scores for a session
        """
        try:
            result = self.supabase.table('musician_frame_analysis').select(
                'emotion_angry, emotion_disgust, emotion_fear, emotion_happy, emotion_sad, emotion_surprise, emotion_neutral'
            ).eq('session_id', session_id).execute()
            
            if result.data:
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                summary = {}
                
                for emotion in emotions:
                    col_name = f'emotion_{emotion}'
                    values = [row[col_name] for row in result.data if row[col_name] is not None]
                    summary[emotion] = sum(values) / len(values) if values else 0.0
                
                return summary
            return None
        except Exception as e:
            print(f"❌ Error getting emotion summary: {e}")
            return None

    # Video Alignment Offset Methods
    def insert_video_alignment(self, video_name: str, start_time_offset: float, 
                              matching_duration: float = 0.0, reference_video: str = None) -> bool:
        """
        Insert or update video alignment offset data
        
        Args:
            video_name: Name of the video file
            start_time_offset: Start time offset in seconds
            matching_duration: Duration of matching content in seconds (default 0.0)
            reference_video: Name of the reference video used for alignment
        """
        data = {
            'video_name': video_name,
            'start_time_offset': start_time_offset,
            'matching_duration': matching_duration,
            'reference_video': reference_video,
            'updated_at': 'NOW()'
        }
        
        try:
            # Try to upsert (insert or update if exists)
            result = self.supabase.table('video_alignment_offsets').upsert(
                data, on_conflict='video_name'
            ).execute()
            print(f"✅ Video alignment data saved for {video_name}")
            return True
        except Exception as e:
            print(f"❌ Error inserting video alignment data: {e}")
            return False
    
    def get_video_alignment(self, video_name: str) -> dict:
        """
        Get alignment data for a specific video
        
        Args:
            video_name: Name of the video file
            
        Returns:
            Dictionary containing alignment data or None if not found
        """
        try:
            result = self.supabase.table('video_alignment_offsets').select('*').eq(
                'video_name', video_name
            ).execute()
            
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"❌ Error retrieving video alignment data: {e}")
            return None
    
    def get_all_video_alignments(self) -> list:
        """
        Get all video alignment data
        
        Returns:
            List of dictionaries containing alignment data for all videos
        """
        try:
            result = self.supabase.table('video_alignment_offsets').select('*').order('video_name').execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"❌ Error retrieving all video alignment data: {e}")
            return []
    
    def determine_reference_video(self, video_list: list) -> str:
        """
        Automatically determine which video should be the reference based on alignment data
        The reference video is typically the one with both start_time_offset and end_time_offset as 0
        
        Args:
            video_list: List of video names to check
            
        Returns:
            Name of the reference video, or the first video if none is found
        """
        try:
            for video_name in video_list:
                alignment_data = self.get_video_alignment(video_name)
                if (alignment_data and 
                    alignment_data.get('start_time_offset', 0) == 0 and 
                    alignment_data.get('end_time_offset', 0) == 0):
                    return video_name
            
            # If no reference video found, return the first one
            return video_list[0] if video_list else None
        except Exception as e:
            print(f"❌ Error determining reference video: {e}")
            return video_list[0] if video_list else None
    
    def delete_video_alignment(self, video_name: str) -> bool:
        """
        Delete alignment data for a specific video
        
        Args:
            video_name: Name of the video file
        """
        try:
            result = self.supabase.table('video_alignment_offsets').delete().eq(
                'video_name', video_name
            ).execute()
            print(f"✅ Deleted alignment data for {video_name}")
            return True
        except Exception as e:
            print(f"❌ Error deleting video alignment data: {e}")
            return False

def setup_environment():
    """
    Setup environment variables for Supabase connection
    """
    print("🔧 Setting up Supabase connection...")
    print("\nPlease set the following environment variables:")
    print("1. SUPABASE_URL - Your Supabase project URL")
    print("2. SUPABASE_ANON_KEY - Your Supabase anon/public key")
    print("\nYou can set them in your shell:")
    print("export SUPABASE_URL='https://your-project-id.supabase.co'")
    print("export SUPABASE_ANON_KEY='your-anon-key-here'")
    print("\nOr create a .env file with:")
    print("SUPABASE_URL=https://your-project-id.supabase.co")
    print("SUPABASE_ANON_KEY=your-anon-key-here")

def test_video_alignments(db):
    """
    Test video alignment functionality
    """
    print("\n🔧 Testing video alignment functionality...")
    
    # Test inserting video alignment data
    test_videos = [
        ("cam_1.mp4", 12.494, 75.0, "cam_3.mp4"),
        ("cam_2.mp4", 11.045, 75.0, "cam_3.mp4"),
        ("cam_3.mp4", 0.0, 75.0, None)  # Reference video
    ]
    
    for video_name, start_offset, matching_dur, ref_video in test_videos:
        success = db.insert_video_alignment(video_name, start_offset, matching_dur, ref_video)
        if success:
            print(f"✅ Inserted alignment data for {video_name}")
        else:
            print(f"❌ Failed to insert alignment data for {video_name}")
    
    # Test retrieving video alignment data
    for video_name, _, _, _ in test_videos:
        data = db.get_video_alignment(video_name)
        if data:
            print(f"✅ Retrieved data for {video_name}: start={data['start_time_offset']}s")
        else:
            print(f"❌ Failed to retrieve data for {video_name}")
    
    # Test determining reference video
    video_names = [video[0] for video in test_videos]
    reference_video = db.determine_reference_video(video_names)
    print(f"✅ Determined reference video: {reference_video}")
    
    # Test getting all alignments
    all_alignments = db.get_all_video_alignments()
    print(f"✅ Retrieved {len(all_alignments)} total video alignment records")

def test_connection():
    """
    Test the database connection and operations
    """
    try:
        # Initialize database
        db = MusicianDatabase()
        
        # Create table
        db.create_table()
        
        # Test data insertion
        test_session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = db.insert_frame_data(
            session_id=test_session_id,
            frame_number=1,
            video_file="test_video.mp4",
            left_hand_landmarks=[{"x": 0.5, "y": 0.3, "z": 0.1, "confidence": 0.9}],
            pose_landmarks=[{"x": 0.4, "y": 0.2, "z": 0.05, "confidence": 0.95}],
            emotions={"happy": 0.8, "neutral": 0.2},
            bad_gestures={"low_wrists": True},
            processing_time_ms=50,
            hand_model="mediapipe",
            pose_model="yolo",
            emotion_model="deepface"
        )
        
        if success:
            print("✅ Test data inserted successfully!")
            
            # Test data retrieval
            data = db.get_session_data(test_session_id)
            if data:
                print(f"✅ Retrieved {len(data)} records")
                
                # Test video alignment functionality
                test_video_alignments(db)
                print("🧪 Test completed successfully!")
            else:
                print("❌ Failed to retrieve test data")
        else:
            print("❌ Failed to insert test data")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        setup_environment()

if __name__ == "__main__":
    # Check if environment variables are set
    if not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_ANON_KEY'):
        setup_environment()
    else:
        test_connection() 