import os
from typing import Optional, List, Dict, Any
import json
import time
from datetime import datetime
import yaml
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, BigInteger, SmallInteger, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from supabase import create_client, Client

# Load config from YAML
def load_config():
    """Load configuration from config_v1.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'config_v1.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

Base = declarative_base()

# SQLAlchemy Models
class MusicianFrameAnalysis(Base):
    __tablename__ = 'musician_frame_analysis'

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Session metadata
    session_id = Column(String(50), nullable=False, index=True)
    video_file = Column(String(255))
    frame_number = Column(Integer, nullable=False)
    person_id = Column(Integer, nullable=False, default=0, index=True)  # Person identifier (0=left, 1=right, etc.)
    original_time = Column(DECIMAL(10, 3))
    synced_time = Column(DECIMAL(10, 3))

    # Hand landmarks (JSON)
    left_hand_landmarks = Column(JSONB)
    right_hand_landmarks = Column(JSONB)

    # Pose landmarks (JSON)
    pose_landmarks = Column(JSONB)

    # Facemesh landmarks (JSON)
    facemesh_landmarks = Column(JSONB)

    # Bad gesture flags
    flag_low_wrists = Column(Boolean, default=False)
    flag_turtle_neck = Column(Boolean, default=False)
    flag_hunched_back = Column(Boolean, default=False)
    flag_fingers_pointing_up = Column(Boolean, default=False)

    # Processing times
    processing_time_ms = Column(DECIMAL(10, 3))
    hand_processing_time_ms = Column(DECIMAL(10, 3), default=0)
    pose_processing_time_ms = Column(DECIMAL(10, 3), default=0)
    facemesh_processing_time_ms = Column(DECIMAL(10, 3), default=0)
    bad_gesture_processing_time_ms = Column(DECIMAL(10, 3), default=0)

    # Model versions
    hand_model = Column(String(50))
    pose_model = Column(String(50))
    facemesh_model = Column(String(50))

    # Transcript reference
    transcript_segment_id = Column(Integer)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())


class VideoAlignmentOffsets(Base):
    __tablename__ = 'video_alignment_offsets'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(Text, nullable=False, unique=True, index=True)
    camera_type = Column(SmallInteger, index=True)
    start_time_offset = Column(DECIMAL(10, 3), nullable=False, default=0.000)
    matching_duration = Column(DECIMAL(10, 3), nullable=False, default=0.000)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())


class ChunkVideoAlignmentOffsets(Base):
    __tablename__ = 'chunk_video_alignment_offsets'

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Source and session info
    source = Column(String(100), nullable=False, index=True)
    session_id = Column(String(50), index=True)

    # Chunk identification
    chunk_filename = Column(String(255), nullable=False, index=True)
    camera_prefix = Column(String(50), nullable=False)
    chunk_order = Column(Integer, nullable=False)

    # Timeline positioning
    start_time_offset = Column(DECIMAL(10, 3), nullable=False, default=0.000)
    chunk_duration = Column(DECIMAL(10, 3), nullable=False, default=0.000)

    # Reference information
    reference_camera_prefix = Column(String(50))
    reference_chunk_filename = Column(String(255))

    # Processing metadata
    correlation_score = Column(DECIMAL(6, 4))

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())


class TranscriptVideo(Base):
    __tablename__ = 'transcript_video'

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Video metadata
    video_file = Column(String(255), nullable=False, index=True)
    session_id = Column(String(50), index=True)

    # Transcript segment data
    segment_id = Column(Integer, nullable=False)
    start_time = Column(DECIMAL(10, 3), nullable=False)
    end_time = Column(DECIMAL(10, 3), nullable=False)
    duration = Column(DECIMAL(8, 3), nullable=False)

    # Transcript text
    text = Column(Text, nullable=False)
    word_count = Column(Integer, default=0)

    # Confidence and quality metrics
    avg_logprob = Column(DECIMAL(8, 4))
    no_speech_prob = Column(DECIMAL(6, 4))

    # Processing metadata
    language = Column(String(10))
    model_size = Column(String(20))
    chunk_duration = Column(DECIMAL(6, 2))

    # Word-level timestamps
    words_json = Column(JSONB)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())


class ProcessingJobs(Base):
    __tablename__ = 'processing_jobs'

    job_id = Column(String(100), primary_key=True)
    type = Column(String(50), nullable=False, index=True)
    status = Column(String(50), nullable=False, default='queued', index=True)
    progress = Column(Integer, default=0)
    message = Column(Text)
    request_data = Column(JSONB)
    output_files = Column(JSONB)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    processing_time_seconds = Column(DECIMAL(10, 2))


class DatabaseManager:
    """Unified database manager that can use either local PostgreSQL or Supabase"""

    def __init__(self, use_local: Optional[bool] = None, config: Optional[dict] = None):
        """
        Initialize database connection

        Args:
            use_local: If True, use local PostgreSQL. If False, use Supabase.
                      If None, read from config file
            config: Configuration dictionary. If None, will load from config_v1.yaml
        """
        # Load config if not provided
        if config is None:
            config = CONFIG

        # Determine which database to use
        if use_local is None:
            db_type = config.get('database', {}).get('use_database_type', 'local').lower()
            self.use_local = (db_type == 'local')
        else:
            self.use_local = use_local

        self.config = config

        if self.use_local:
            self._init_local_postgres()
        else:
            self._init_supabase()

        # Batch insert configuration
        self.batch_size = 50
        self.batch_data = []
        self.batch_timeout = 5.0
        self.last_batch_time = time.time()

        print(f"📦 Batch insert enabled: {self.batch_size} frames per batch, {self.batch_timeout}s timeout")

    def _init_local_postgres(self):
        """Initialize local PostgreSQL connection using SQLAlchemy"""
        db_config = self.config.get('database', {}).get('local', {})
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        database = db_config.get('name', 'musician_tracking')
        user = db_config.get('user', 'postgres')
        password = db_config.get('password', '')

        # Create connection string (handle empty password)
        if password:
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            connection_string = f"postgresql://{user}@{host}:{port}/{database}"

        # Create engine and session
        self.engine = create_engine(connection_string, pool_size=10, max_overflow=20)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)

        # Store connection type
        self.connection_type = "local_postgres"
        self.supabase = None

        print(f"✅ Connected to Local PostgreSQL: {host}:{port}/{database}")

    def _init_supabase(self):
        """Initialize Supabase connection"""
        supabase_config = self.config.get('database', {}).get('supabase', {})
        self.supabase_url = supabase_config.get('url')
        self.supabase_key = supabase_config.get('anon_key')

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing Supabase credentials. Please set database.supabase.url and database.supabase.anon_key in config_v1.yaml"
            )

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.connection_type = "supabase"
        self.engine = None
        self.SessionLocal = None

        print(f"✅ Connected to Supabase: {self.supabase_url}")

    def get_session(self) -> Optional[Session]:
        """Get a database session (only for local PostgreSQL)"""
        if self.use_local:
            return self.SessionLocal()
        return None

    def add_frame_to_batch(self,
                          session_id: str,
                          frame_number: int,
                          person_id: int = 0,  # Person identifier for multi-person videos
                          video_file: Optional[str] = None,
                          original_time: Optional[float] = None,
                          synced_time: Optional[float] = None,
                          left_hand_landmarks: Optional[list] = None,
                          right_hand_landmarks: Optional[list] = None,
                          pose_landmarks: Optional[list] = None,
                          facemesh_landmarks: Optional[list] = None,
                          bad_gestures: Optional[dict] = None,
                          processing_time_ms: Optional[float] = None,
                          hand_processing_time_ms: Optional[float] = None,
                          pose_processing_time_ms: Optional[float] = None,
                          facemesh_processing_time_ms: Optional[float] = None,
                          bad_gesture_processing_time_ms: Optional[float] = None,
                          hand_model: Optional[str] = None,
                          pose_model: Optional[str] = None,
                          facemesh_model: Optional[str] = None,
                          transcript_segment_id: Optional[int] = None) -> bool:
        """Add frame data to batch buffer for later bulk insert"""

        # ✅ FIX: Validate all input data types to prevent .get() errors
        # Ensure bad_gestures is a dict
        if bad_gestures is not None and not isinstance(bad_gestures, dict):
            print(f"⚠️ Warning: bad_gestures is {type(bad_gestures)}, expected dict. Converting to empty dict.")
            bad_gestures = {}

        # Ensure all landmark data are lists (not dicts or other types)
        def ensure_list_or_none(value, name):
            """Ensure value is a list or None, not a dict or other type"""
            if value is None:
                return None
            if isinstance(value, list):
                return value
            # If it's a dict, it might be a single landmark - wrap it in a list
            if isinstance(value, dict):
                print(f"⚠️ Warning: {name} is dict, expected list. Wrapping in list.")
                return [value]
            # For any other type, convert to None
            print(f"⚠️ Warning: {name} is {type(value)}, expected list. Converting to None.")
            return None

        left_hand_landmarks = ensure_list_or_none(left_hand_landmarks, 'left_hand_landmarks')
        right_hand_landmarks = ensure_list_or_none(right_hand_landmarks, 'right_hand_landmarks')
        pose_landmarks = ensure_list_or_none(pose_landmarks, 'pose_landmarks')
        facemesh_landmarks = ensure_list_or_none(facemesh_landmarks, 'facemesh_landmarks')

        # Prepare data
        data = {
            'session_id': session_id,
            'frame_number': frame_number,
            'person_id': person_id,  # Add person identifier
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
            'bad_gesture_processing_time_ms': bad_gesture_processing_time_ms or 0,
            'hand_model': hand_model,
            'pose_model': pose_model,
            'facemesh_model': facemesh_model,
            'transcript_segment_id': transcript_segment_id
        }

        # Add bad gesture flags with safe access
        if bad_gestures and isinstance(bad_gestures, dict):
            data.update({
                'flag_low_wrists': bool(bad_gestures.get('low_wrists', False)),
                'flag_turtle_neck': bool(bad_gestures.get('turtle_neck', False)),
                'flag_hunched_back': bool(bad_gestures.get('hunched_back', False)),
                'flag_fingers_pointing_up': bool(bad_gestures.get('fingers_pointing_up', False))
            })
        else:
            # Default all flags to False if bad_gestures is None or invalid
            data.update({
                'flag_low_wrists': False,
                'flag_turtle_neck': False,
                'flag_hunched_back': False,
                'flag_fingers_pointing_up': False
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
        """Force flush all batched data to database"""
        if not self.batch_data:
            return True

        try:
            batch_start_time = time.time()

            if self.use_local:
                # Use SQLAlchemy for local PostgreSQL
                session = self.get_session()
                try:
                    for data in self.batch_data:
                        frame = MusicianFrameAnalysis(**data)
                        session.add(frame)
                    session.commit()
                finally:
                    session.close()
            else:
                # Use Supabase client
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
            return False

    def get_session_data(self, session_id: str, limit: Optional[int] = None):
        """Retrieve all frame data for a specific session"""
        try:
            if self.use_local:
                session = self.get_session()
                try:
                    query = session.query(MusicianFrameAnalysis).filter(
                        MusicianFrameAnalysis.session_id == session_id
                    ).order_by(MusicianFrameAnalysis.frame_number)

                    if limit:
                        query = query.limit(limit)

                    result = query.all()
                    # Convert to dict for compatibility
                    return [row.__dict__ for row in result]
                finally:
                    session.close()
            else:
                query = self.supabase.table('musician_frame_analysis').select('*').eq('session_id', session_id).order('frame_number')

                if limit:
                    query = query.limit(limit)

                result = query.execute()
                return result.data
        except Exception as e:
            print(f"❌ Error retrieving session data: {e}")
            return None

    def get_bad_gesture_summary(self, session_id: str):
        """Get summary of bad gestures for a session"""
        try:
            data = self.get_session_data(session_id)

            if data:
                summary = {
                    'low_wrists': sum(1 for row in data if row.get('flag_low_wrists')),
                    'turtle_neck': sum(1 for row in data if row.get('flag_turtle_neck')),
                    'hunched_back': sum(1 for row in data if row.get('flag_hunched_back')),
                    'fingers_pointing_up': sum(1 for row in data if row.get('flag_fingers_pointing_up')),
                    'total_frames': len(data)
                }
                return summary
            return None
        except Exception as e:
            print(f"❌ Error getting bad gesture summary: {e}")
            return None


    def insert_video_alignment(self, source: str, start_time_offset: float,
                              matching_duration: float = 0.0, camera_type: int = None) -> bool:
        """Insert or update video alignment offset data"""
        try:
            if self.use_local:
                session = self.get_session()
                try:
                    # Check if exists
                    existing = session.query(VideoAlignmentOffsets).filter(
                        VideoAlignmentOffsets.source == source
                    ).first()

                    if existing:
                        # Update existing
                        existing.start_time_offset = start_time_offset
                        existing.matching_duration = matching_duration
                        existing.camera_type = camera_type
                        existing.updated_at = func.now()
                    else:
                        # Insert new
                        alignment = VideoAlignmentOffsets(
                            source=source,
                            start_time_offset=start_time_offset,
                            matching_duration=matching_duration,
                            camera_type=camera_type
                        )
                        session.add(alignment)

                    session.commit()
                    print(f"✅ Video alignment data saved for {source}")
                    return True
                finally:
                    session.close()
            else:
                # Use Supabase
                data = {
                    'source': source,
                    'start_time_offset': start_time_offset,
                    'matching_duration': matching_duration,
                    'camera_type': camera_type,
                    'updated_at': 'NOW()'
                }
                result = self.supabase.table('video_alignment_offsets').upsert(data).execute()
                print(f"✅ Video alignment data saved for {source}")
                return True
        except Exception as e:
            print(f"❌ Error inserting video alignment data: {e}")
            return False

    def get_video_alignment(self, source: str) -> dict:
        """Get alignment data for a specific video source"""
        try:
            if self.use_local:
                session = self.get_session()
                try:
                    result = session.query(VideoAlignmentOffsets).filter(
                        VideoAlignmentOffsets.source == source
                    ).first()

                    if result:
                        return {
                            'source': result.source,
                            'start_time_offset': float(result.start_time_offset),
                            'matching_duration': float(result.matching_duration),
                            'camera_type': result.camera_type
                        }
                    return None
                finally:
                    session.close()
            else:
                result = self.supabase.table('video_alignment_offsets').select('*').eq(
                    'source', source
                ).execute()

                if result.data:
                    return result.data[0]
                return None
        except Exception as e:
            print(f"❌ Error retrieving video alignment data: {e}")
            return None

    def close(self):
        """Close database connection and flush any remaining batch data"""
        if self.batch_data:
            print(f"📦 Flushing remaining {len(self.batch_data)} frames before closing...")
            self.flush_batch()

        if self.use_local and self.engine:
            self.engine.dispose()

        print("📪 Database connection closed")


# Convenience classes for backwards compatibility
class MusicianDatabase(DatabaseManager):
    """Legacy class name for backwards compatibility"""
    def __init__(self, table_name: str = 'musician_frame_analysis'):
        super().__init__()
        self.table_name = table_name
        print(f"📋 Using table: {self.table_name}")

    # Add any legacy methods that need to be preserved
    def insert_frame_data(self, **kwargs):
        """Legacy method for single frame insertion"""
        return self.add_frame_to_batch(**kwargs)


def test_connection():
    """Test the database connection"""
    print("\n🧪 Testing Database Connection...")
    print(f"📍 Database Type: {CONFIG.get('database', {}).get('use_database_type', 'local')}")

    try:
        # Test with config setting
        db = DatabaseManager()

        # Test data insertion
        test_session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        success = db.add_frame_to_batch(
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

        # Force flush to test insertion
        db.flush_batch()

        # Test data retrieval
        data = db.get_session_data(test_session_id, limit=1)
        if data:
            print(f"✅ Successfully inserted and retrieved test data")
            print(f"✅ Using {db.connection_type} database")

            # Test video alignment
            db.insert_video_alignment("test_cam_1.mp4", 12.5, 60.0, 1)
            alignment = db.get_video_alignment("test_cam_1.mp4")
            if alignment:
                print(f"✅ Video alignment functionality working")

            print("\n🎉 All tests passed!")
        else:
            print("❌ Failed to retrieve test data")

        db.close()

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        print("\n📋 Please check your config_v1.yaml file has the correct settings:")
        print("   - For local PostgreSQL: database.use_database_type: local")
        print("   - For Supabase: database.use_database_type: supabase")


if __name__ == "__main__":
    test_connection()