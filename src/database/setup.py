import os
from supabase import create_client, Client
from typing import Optional, List, Dict, Any
import json
import time
from datetime import datetime
import yaml
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, DECIMAL, SmallInteger, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

# Load config from YAML
def load_config():
    """Load configuration from config_v1.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'config_v1.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# SQLAlchemy Base for ORM models
Base = declarative_base()

# SQLAlchemy models for local PostgreSQL
class VideoAlignmentOffset(Base):
    """Model for video_alignment_offsets table"""
    __tablename__ = 'video_alignment_offsets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(Text, nullable=False, unique=True)
    camera_type = Column(SmallInteger)
    start_time_offset = Column(DECIMAL(10, 3), nullable=False, default=0.000)
    matching_duration = Column(DECIMAL(10, 3), nullable=False, default=0.000)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class ChunkVideoAlignmentOffset(Base):
    """Model for chunk_video_alignment_offsets table"""
    __tablename__ = 'chunk_video_alignment_offsets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(VARCHAR(100), nullable=False)
    session_id = Column(VARCHAR(50))
    chunk_filename = Column(VARCHAR(255), nullable=False)
    camera_prefix = Column(VARCHAR(50), nullable=False)
    chunk_order = Column(Integer, nullable=False)
    start_time_offset = Column(DECIMAL(10, 3), nullable=False, default=0.000)
    chunk_duration = Column(DECIMAL(10, 3), nullable=False, default=0.000)
    reference_camera_prefix = Column(VARCHAR(50))
    reference_chunk_filename = Column(VARCHAR(255))
    correlation_score = Column(DECIMAL(6, 4))
    method_type = Column(Text, default='earliest_start')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class MusicianDatabase:
    def __init__(self, table_name: str = 'musician_frame_analysis', config: Optional[dict] = None):
        """Initialize Supabase connection

        Args:
            table_name: Name of the table to use for frame analysis data
            config: Configuration dictionary. If None, will load from config_v1.yaml
        """
        # Load config if not provided
        if config is None:
            config = CONFIG

        supabase_config = config.get('database', {}).get('supabase', {})
        self.supabase_url = supabase_config.get('url')
        self.supabase_key = supabase_config.get('anon_key')

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing Supabase credentials. Please set database.supabase.url and database.supabase.anon_key in config_v1.yaml"
            )
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.table_name = table_name  # Store the table name
        
        # Batch insert configuration
        self.batch_size = 50  # Number of frames to batch before inserting
        self.batch_data = []  # Buffer to store frame data
        self.batch_timeout = 5.0  # Maximum time to wait before forcing insert (seconds)
        self.last_batch_time = time.time()
        
        print(f"✅ Connected to Supabase: {self.supabase_url}")
        print(f"📋 Using table: {self.table_name}")
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
            'bad_gesture_processing_time_ms': bad_gesture_processing_time_ms or 0,
            'hand_model': hand_model,
            'pose_model': pose_model,
            'facemesh_model': facemesh_model,
            'transcript_segment_id': transcript_segment_id
        }

        # Add bad gesture flags (ensure Python bool type for JSON serialization)
        if bad_gestures:
            data.update({
                'flag_low_wrists': bool(bad_gestures.get('low_wrists', False)),
                'flag_turtle_neck': bool(bad_gestures.get('turtle_neck', False)),
                'flag_hunched_back': bool(bad_gestures.get('hunched_back', False)),
                'flag_fingers_pointing_up': bool(bad_gestures.get('fingers_pointing_up', False))
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
            response = self.supabase.table(self.table_name).insert(self.batch_data).execute()
            
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
                         bad_gestures: Optional[dict] = None,
                         processing_time_ms: Optional[int] = None,
                         hand_processing_time_ms: Optional[int] = None,
                         pose_processing_time_ms: Optional[int] = None,
                         facemesh_processing_time_ms: Optional[int] = None,
                         hand_model: Optional[str] = None,
                         pose_model: Optional[str] = None,
                         facemesh_model: Optional[str] = None) -> bool:
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
            bad_gestures: Dictionary of bad gesture flags
            processing_time_ms: Total time taken to process this frame
            hand_processing_time_ms: Time taken for hand detection
            pose_processing_time_ms: Time taken for pose detection
            facemesh_processing_time_ms: Time taken for face mesh detection
            hand_model: Hand detection model used (e.g., "mediapipe")
            pose_model: Pose detection model used (e.g., "yolo", "mediapipe")
            facemesh_model: Face mesh model used (e.g., "mediapipe", "yolo")
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
            'hand_model': hand_model,
            'pose_model': pose_model,
            'facemesh_model': facemesh_model
        }

        # Add bad gesture flags (ensure Python bool type for JSON serialization)
        if bad_gestures:
            data.update({
                'flag_low_wrists': bool(bad_gestures.get('low_wrists', False)),
                'flag_turtle_neck': bool(bad_gestures.get('turtle_neck', False)),
                'flag_hunched_back': bool(bad_gestures.get('hunched_back', False)),
                'flag_fingers_pointing_up': bool(bad_gestures.get('fingers_pointing_up', False))
            })
        
        try:
            result = self.supabase.table(self.table_name).insert(data).execute()
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
            query = self.supabase.table(self.table_name).select('*').eq('session_id', session_id).order('frame_number')
            
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
            result = self.supabase.table(self.table_name).select(
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



class VideoAlignmentDatabase:
    def __init__(self, use_local: Optional[bool] = None, config: Optional[dict] = None):
        """Initialize database connection for video alignment operations

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

        print(f"✅ VideoAlignmentDatabase connected to Local PostgreSQL: {host}:{port}/{database}")

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

        print(f"✅ VideoAlignmentDatabase connected to Supabase: {self.supabase_url}")

    def get_session(self) -> Optional[Session]:
        """Get a database session (only for local PostgreSQL)"""
        if self.use_local:
            return self.SessionLocal()
        return None
    
    def create_table(self) -> bool:
        """Create video_alignment_offsets table in Supabase with updated schema"""
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS video_alignment_offsets (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL UNIQUE,
            camera_type SMALLINT,
            start_time_offset DECIMAL(10,3) NOT NULL DEFAULT 0.000,  -- offset in seconds
            matching_duration DECIMAL(10,3) NOT NULL DEFAULT 0.000,  -- duration of matching content in seconds
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for video alignment table
        CREATE INDEX IF NOT EXISTS idx_video_source ON video_alignment_offsets(source);
        CREATE INDEX IF NOT EXISTS idx_camera_type ON video_alignment_offsets(camera_type);
        """
        
        try:
            # Execute the SQL using Supabase's rpc function
            result = self.supabase.rpc('exec_sql', {'sql': create_table_sql}).execute()
            print("✅ Table 'video_alignment_offsets' created successfully with updated schema!")
            return True
        except Exception as e:
            print(f"❌ RPC method failed: {e}")
            print("💡 Please create the table manually in Supabase SQL Editor using the provided SQL.")
            return False
    
    def insert_video_alignment(self, source: str, start_time_offset: float,
                              matching_duration: float = 0.0, camera_type: int = None) -> bool:
        """
        Insert or update video alignment offset data

        Args:
            source: Source video file name or identifier
            start_time_offset: Start time offset in seconds
            matching_duration: Duration of matching content in seconds (default 0.0)
            camera_type: Camera type identifier (int8)
        """
        try:
            if self.use_local:
                session = self.get_session()
                try:
                    # Check if record exists
                    existing = session.query(VideoAlignmentOffset).filter_by(source=source).first()

                    if existing:
                        # Update existing record
                        existing.start_time_offset = start_time_offset
                        existing.matching_duration = matching_duration
                        existing.camera_type = camera_type
                        existing.updated_at = datetime.now()
                    else:
                        # Insert new record
                        new_record = VideoAlignmentOffset(
                            source=source,
                            start_time_offset=start_time_offset,
                            matching_duration=matching_duration,
                            camera_type=camera_type
                        )
                        session.add(new_record)

                    session.commit()
                    print(f"✅ Video alignment data saved for {source}")
                    return True
                finally:
                    session.close()
            else:
                # Supabase implementation
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
        """
        Get alignment data for a specific video source

        Args:
            source: Source video file name or identifier

        Returns:
            Dictionary containing alignment data or None if not found
        """
        try:
            if self.use_local:
                session = self.get_session()
                try:
                    record = session.query(VideoAlignmentOffset).filter_by(source=source).first()
                    if record:
                        return {
                            'id': record.id,
                            'source': record.source,
                            'camera_type': record.camera_type,
                            'start_time_offset': float(record.start_time_offset),
                            'matching_duration': float(record.matching_duration),
                            'created_at': record.created_at.isoformat() if record.created_at else None,
                            'updated_at': record.updated_at.isoformat() if record.updated_at else None
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

    def get_all_video_alignments(self) -> list:
        """
        Get all video alignment data

        Returns:
            List of dictionaries containing alignment data for all videos
        """
        try:
            if self.use_local:
                session = self.get_session()
                try:
                    records = session.query(VideoAlignmentOffset).order_by(VideoAlignmentOffset.source).all()
                    return [{
                        'id': record.id,
                        'source': record.source,
                        'camera_type': record.camera_type,
                        'start_time_offset': float(record.start_time_offset),
                        'matching_duration': float(record.matching_duration),
                        'created_at': record.created_at.isoformat() if record.created_at else None,
                        'updated_at': record.updated_at.isoformat() if record.updated_at else None
                    } for record in records]
                finally:
                    session.close()
            else:
                result = self.supabase.table('video_alignment_offsets').select('*').order('source').execute()
                return result.data if result.data else []
        except Exception as e:
            print(f"❌ Error retrieving all video alignment data: {e}")
            return []
    
    def get_alignments_by_camera_type(self, camera_type: int) -> list:
        """
        Get all video alignment data for a specific camera type

        Args:
            camera_type: Camera type identifier

        Returns:
            List of dictionaries containing alignment data for the specified camera type
        """
        try:
            if self.use_local:
                # Local PostgreSQL implementation
                session = self.get_session()
                try:
                    records = session.query(VideoAlignmentOffset).filter(
                        VideoAlignmentOffset.camera_type == camera_type
                    ).order_by(VideoAlignmentOffset.source).all()

                    return [{
                        'id': record.id,
                        'source': record.source,
                        'camera_type': record.camera_type,
                        'start_time_offset': float(record.start_time_offset),
                        'matching_duration': float(record.matching_duration),
                        'created_at': record.created_at.isoformat() if record.created_at else None,
                        'updated_at': record.updated_at.isoformat() if record.updated_at else None
                    } for record in records]
                finally:
                    session.close()
            else:
                # Supabase implementation
                result = self.supabase.table('video_alignment_offsets').select('*').eq(
                    'camera_type', camera_type
                ).order('source').execute()
                return result.data if result.data else []
        except Exception as e:
            print(f"❌ Error retrieving video alignment data for camera type {camera_type}: {e}")
            return []
    
    def determine_reference_video(self, video_list: list) -> str:
        """
        Automatically determine which video should be the reference based on alignment data
        The reference video is typically the one with start_time_offset as 0
        
        Args:
            video_list: List of video source names to check
            
        Returns:
            Name of the reference video, or the first video if none is found
        """
        try:
            for source in video_list:
                alignment_data = self.get_video_alignment(source)
                if alignment_data and alignment_data.get('start_time_offset', 0) == 0:
                    return source
            
            # If no reference video found, return the first one
            return video_list[0] if video_list else None
        except Exception as e:
            print(f"❌ Error determining reference video: {e}")
            return video_list[0] if video_list else None
    
    def delete_video_alignment(self, source: str) -> bool:
        """
        Delete alignment data for a specific video source

        Args:
            source: Source video file name or identifier
        """
        try:
            if self.use_local:
                # Local PostgreSQL implementation
                session = self.get_session()
                try:
                    session.query(VideoAlignmentOffset).filter(
                        VideoAlignmentOffset.source == source
                    ).delete()
                    session.commit()
                    print(f"✅ Deleted alignment data for {source}")
                    return True
                except Exception as e:
                    session.rollback()
                    raise e
                finally:
                    session.close()
            else:
                # Supabase implementation
                result = self.supabase.table('video_alignment_offsets').delete().eq(
                    'source', source
                ).execute()
                print(f"✅ Deleted alignment data for {source}")
                return True
        except Exception as e:
            print(f"❌ Error deleting video alignment data: {e}")
            return False


class ChunkVideoAlignmentDatabase:
    def __init__(self, use_local: Optional[bool] = None, config: Optional[dict] = None):
        """Initialize database connection for chunk video alignment operations

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

        print(f"✅ ChunkVideoAlignmentDatabase connected to Local PostgreSQL: {host}:{port}/{database}")

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

        print(f"✅ ChunkVideoAlignmentDatabase connected to Supabase: {self.supabase_url}")

    def get_session(self) -> Optional[Session]:
        """Get a database session (only for local PostgreSQL)"""
        if self.use_local:
            return self.SessionLocal()
        return None

    def create_table(self) -> bool:
        """Create chunk_video_alignment_offsets table in Supabase"""

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS chunk_video_alignment_offsets (
            id BIGSERIAL PRIMARY KEY,

            -- Source and session info
            source VARCHAR(100) NOT NULL,              -- e.g., 'vid_shot1'
            session_id VARCHAR(50),                    -- optional session identifier

            -- Chunk identification
            chunk_filename VARCHAR(255) NOT NULL,      -- e.g., 'cam_1_2.mp4'
            camera_prefix VARCHAR(50) NOT NULL,        -- e.g., 'cam_1', 'cam_2'
            chunk_order INTEGER NOT NULL,              -- chunk number within camera group (0, 1, 2, etc.)

            -- Timeline positioning
            start_time_offset DECIMAL(10,3) NOT NULL DEFAULT 0.000,  -- absolute timeline position in seconds
            chunk_duration DECIMAL(10,3) NOT NULL DEFAULT 0.000,     -- duration of this chunk in seconds

            -- Reference information
            reference_camera_prefix VARCHAR(50),       -- which camera group was used as reference
            reference_chunk_filename VARCHAR(255),     -- which specific chunk was used as reference

            -- Processing metadata
            correlation_score DECIMAL(6,4),            -- audio pattern matching correlation score
            method_type TEXT DEFAULT 'earliest_start',  -- alignment method: 'earliest_start' or 'latest_start'

            -- Timestamps
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Create indexes for chunk_video_alignment_offsets table
        CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunk_video_alignment_offsets(source);
        CREATE INDEX IF NOT EXISTS idx_chunk_session ON chunk_video_alignment_offsets(session_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_filename ON chunk_video_alignment_offsets(chunk_filename);
        CREATE INDEX IF NOT EXISTS idx_chunk_camera ON chunk_video_alignment_offsets(camera_prefix, chunk_order);
        CREATE INDEX IF NOT EXISTS idx_chunk_timeline ON chunk_video_alignment_offsets(source, start_time_offset);
        CREATE INDEX IF NOT EXISTS idx_chunk_reference ON chunk_video_alignment_offsets(reference_camera_prefix);
        CREATE INDEX IF NOT EXISTS idx_chunk_created_at ON chunk_video_alignment_offsets(created_at);
        CREATE INDEX IF NOT EXISTS idx_chunk_method_type ON chunk_video_alignment_offsets(method_type);

        -- Composite index for efficient queries
        CREATE INDEX IF NOT EXISTS idx_chunk_source_camera_order ON chunk_video_alignment_offsets(source, camera_prefix, chunk_order);
        CREATE INDEX IF NOT EXISTS idx_chunk_source_method ON chunk_video_alignment_offsets(source, method_type);
        """

        try:
            # Execute the SQL using Supabase's rpc function
            result = self.supabase.rpc('exec_sql', {'sql': create_table_sql}).execute()
            print("✅ Table 'chunk_video_alignment_offsets' created successfully!")
            return True
        except Exception as e:
            print(f"❌ RPC method failed: {e}")
            print("💡 Please create the table manually in Supabase SQL Editor using the provided SQL.")
            return False

    def insert_chunk_alignment(self,
                             source: str,
                             chunk_filename: str,
                             camera_prefix: str,
                             chunk_order: int,
                             start_time_offset: float,
                             chunk_duration: float,
                             reference_camera_prefix: str = None,
                             reference_chunk_filename: str = None,
                             correlation_score: float = None,
                             session_id: str = None,
                             method_type: str = 'earliest_start') -> bool:
        """
        Insert chunk alignment data

        Args:
            source: Video session identifier (e.g., 'vid_shot1')
            chunk_filename: Name of the chunk file (e.g., 'cam_1_2.mp4')
            camera_prefix: Camera group identifier (e.g., 'cam_1')
            chunk_order: Chunk number within camera group (0, 1, 2, etc.)
            start_time_offset: Absolute timeline position in seconds
            chunk_duration: Duration of this chunk in seconds
            reference_camera_prefix: Reference camera group used
            reference_chunk_filename: Reference chunk used
            correlation_score: Audio pattern matching correlation score
            session_id: Optional session identifier
            method_type: Alignment method used ('earliest_start' or 'latest_start')
        """
        try:
            if self.use_local:
                session = self.get_session()
                try:
                    # Insert new record
                    new_record = ChunkVideoAlignmentOffset(
                        source=source,
                        session_id=session_id,
                        chunk_filename=chunk_filename,
                        camera_prefix=camera_prefix,
                        chunk_order=chunk_order,
                        start_time_offset=start_time_offset,
                        chunk_duration=chunk_duration,
                        reference_camera_prefix=reference_camera_prefix,
                        reference_chunk_filename=reference_chunk_filename,
                        correlation_score=correlation_score,
                        method_type=method_type
                    )
                    session.add(new_record)
                    session.commit()
                    return True
                finally:
                    session.close()
            else:
                # Supabase implementation
                data = {
                    'source': source,
                    'session_id': session_id,
                    'chunk_filename': chunk_filename,
                    'camera_prefix': camera_prefix,
                    'chunk_order': chunk_order,
                    'start_time_offset': start_time_offset,
                    'chunk_duration': chunk_duration,
                    'reference_camera_prefix': reference_camera_prefix,
                    'reference_chunk_filename': reference_chunk_filename,
                    'correlation_score': correlation_score,
                    'method_type': method_type,
                    'updated_at': 'NOW()'
                }
                result = self.supabase.table('chunk_video_alignment_offsets').insert(data).execute()
                return True
        except Exception as e:
            print(f"❌ Error inserting chunk alignment data for {chunk_filename}: {e}")
            return False

    def get_chunk_alignments_by_source_and_method(self, source: str, method_type: str = 'latest_start') -> list:
        """
        Get chunk alignment data for a specific source and method type

        Args:
            source: Source identifier (e.g., 'vid_shot1')
            method_type: Alignment method ('earliest_start' or 'latest_start')

        Returns:
            List of dictionaries containing chunk alignment data
        """
        try:
            if self.use_local:
                # Local PostgreSQL implementation
                session = self.get_session()
                try:
                    records = session.query(ChunkVideoAlignmentOffset).filter(
                        ChunkVideoAlignmentOffset.source == source,
                        ChunkVideoAlignmentOffset.method_type == method_type
                    ).order_by(ChunkVideoAlignmentOffset.start_time_offset).all()

                    # Convert to dict format
                    return [{
                        'id': r.id,
                        'source': r.source,
                        'session_id': r.session_id,
                        'chunk_filename': r.chunk_filename,
                        'camera_prefix': r.camera_prefix,
                        'chunk_order': r.chunk_order,
                        'start_time_offset': r.start_time_offset,
                        'chunk_duration': r.chunk_duration,
                        'reference_camera_prefix': r.reference_camera_prefix,
                        'reference_chunk_filename': r.reference_chunk_filename,
                        'correlation_score': r.correlation_score,
                        'method_type': r.method_type,
                        'created_at': r.created_at,
                        'updated_at': r.updated_at
                    } for r in records]
                finally:
                    session.close()
            else:
                # Supabase implementation
                result = self.supabase.table('chunk_video_alignment_offsets').select('*').eq(
                    'source', source
                ).eq(
                    'method_type', method_type
                ).order('start_time_offset').execute()

                return result.data if result.data else []
        except Exception as e:
            print(f"❌ Error retrieving chunk alignment data for source {source} with method {method_type}: {e}")
            return []

    def get_chunk_alignments_by_source(self, source: str) -> list:
        """
        Get all chunk alignment data for a specific source (latest session only)

        Args:
            source: Source identifier (e.g., 'vid_shot1')

        Returns:
            List of dictionaries containing chunk alignment data from the latest session
        """
        try:
            if self.use_local:
                session = self.get_session()
                try:
                    # First, get the latest session_id for this source
                    latest_record = session.query(ChunkVideoAlignmentOffset).filter_by(
                        source=source
                    ).order_by(ChunkVideoAlignmentOffset.created_at.desc()).first()

                    if not latest_record:
                        return []

                    latest_session_id = latest_record.session_id
                    print(f"🔍 Found latest session for source '{source}': {latest_session_id}")

                    # Get all records for this source and latest session
                    records = session.query(ChunkVideoAlignmentOffset).filter_by(
                        source=source,
                        session_id=latest_session_id
                    ).order_by(ChunkVideoAlignmentOffset.start_time_offset).all()

                    return [{
                        'id': record.id,
                        'source': record.source,
                        'session_id': record.session_id,
                        'chunk_filename': record.chunk_filename,
                        'camera_prefix': record.camera_prefix,
                        'chunk_order': record.chunk_order,
                        'start_time_offset': float(record.start_time_offset),
                        'chunk_duration': float(record.chunk_duration),
                        'reference_camera_prefix': record.reference_camera_prefix,
                        'reference_chunk_filename': record.reference_chunk_filename,
                        'correlation_score': float(record.correlation_score) if record.correlation_score else None,
                        'method_type': record.method_type,
                        'created_at': record.created_at.isoformat() if record.created_at else None,
                        'updated_at': record.updated_at.isoformat() if record.updated_at else None
                    } for record in records]
                finally:
                    session.close()
            else:
                # First, get the latest session_id for this source
                latest_session_result = self.supabase.table('chunk_video_alignment_offsets').select('session_id, created_at').eq(
                    'source', source
                ).order('created_at', desc=True).limit(1).execute()

                if not latest_session_result.data:
                    return []

                latest_session_id = latest_session_result.data[0]['session_id']
                print(f"🔍 Found latest session for source '{source}': {latest_session_id}")

                # Get all records for this source and latest session
                result = self.supabase.table('chunk_video_alignment_offsets').select('*').eq(
                    'source', source
                ).eq('session_id', latest_session_id).order('start_time_offset').execute()

                return result.data if result.data else []
        except Exception as e:
            print(f"❌ Error retrieving chunk alignment data for source {source}: {e}")
            return []

    def get_chunk_alignments_by_camera(self, source: str, camera_prefix: str) -> list:
        """
        Get all chunk alignment data for a specific camera group

        Args:
            source: Source identifier (e.g., 'vid_shot1')
            camera_prefix: Camera group identifier (e.g., 'cam_1')

        Returns:
            List of dictionaries containing chunk alignment data for the camera group
        """
        try:
            if self.use_local:
                # Local PostgreSQL implementation
                session = self.get_session()
                try:
                    records = session.query(ChunkVideoAlignmentOffset).filter(
                        ChunkVideoAlignmentOffset.source == source,
                        ChunkVideoAlignmentOffset.camera_prefix == camera_prefix
                    ).order_by(ChunkVideoAlignmentOffset.chunk_order).all()

                    # Convert to dict format
                    return [{
                        'id': r.id,
                        'source': r.source,
                        'session_id': r.session_id,
                        'chunk_filename': r.chunk_filename,
                        'camera_prefix': r.camera_prefix,
                        'chunk_order': r.chunk_order,
                        'start_time_offset': r.start_time_offset,
                        'chunk_duration': r.chunk_duration,
                        'reference_camera_prefix': r.reference_camera_prefix,
                        'reference_chunk_filename': r.reference_chunk_filename,
                        'correlation_score': r.correlation_score,
                        'method_type': r.method_type,
                        'created_at': r.created_at,
                        'updated_at': r.updated_at
                    } for r in records]
                finally:
                    session.close()
            else:
                # Supabase implementation
                result = self.supabase.table('chunk_video_alignment_offsets').select('*').eq(
                    'source', source
                ).eq(
                    'camera_prefix', camera_prefix
                ).order('chunk_order').execute()
                return result.data if result.data else []
        except Exception as e:
            print(f"❌ Error retrieving chunk alignment data for {source}/{camera_prefix}: {e}")
            return []

    def delete_chunk_alignments_by_source(self, source: str) -> bool:
        """
        Delete all chunk alignment data for a specific source

        Args:
            source: Source identifier (e.g., 'vid_shot1')
        """
        try:
            if self.use_local:
                # Local PostgreSQL implementation
                session = self.get_session()
                try:
                    session.query(ChunkVideoAlignmentOffset).filter(
                        ChunkVideoAlignmentOffset.source == source
                    ).delete()
                    session.commit()
                    print(f"✅ Deleted chunk alignment data for source {source}")
                    return True
                except Exception as e:
                    session.rollback()
                    raise e
                finally:
                    session.close()
            else:
                # Supabase implementation
                result = self.supabase.table('chunk_video_alignment_offsets').delete().eq(
                    'source', source
                ).execute()
                print(f"✅ Deleted chunk alignment data for source {source}")
                return True
        except Exception as e:
            print(f"❌ Error deleting chunk alignment data for source {source}: {e}")
            return False

    def get_all_sources(self) -> List[str]:
        """
        Get all unique source identifiers from chunk video alignment data

        Returns:
            List of unique source identifiers
        """
        try:
            if self.use_local:
                # Local PostgreSQL implementation
                session = self.get_session()
                try:
                    # Get distinct sources using SQLAlchemy
                    records = session.query(ChunkVideoAlignmentOffset.source).distinct().all()

                    # Extract unique sources
                    sources = [r.source for r in records if r.source]
                    sources.sort()  # Sort alphabetically

                    print(f"✅ Found {len(sources)} unique sources")
                    return sources
                finally:
                    session.close()
            else:
                # Supabase implementation
                # Get distinct sources from the table
                result = self.supabase.table('chunk_video_alignment_offsets').select('source').execute()

                if not result.data:
                    return []

                # Extract unique sources
                sources = list(set([item['source'] for item in result.data if item.get('source')]))
                sources.sort()  # Sort alphabetically

                print(f"✅ Found {len(sources)} unique sources")
                return sources
        except Exception as e:
            print(f"❌ Error retrieving sources: {e}")
            return []


def setup_environment():
    """
    Setup Supabase connection configuration
    """
    print("🔧 Setting up Supabase connection...")
    print("\nPlease set the following in config_v1.yaml:")
    print("database:")
    print("  use_database_type: supabase")
    print("  supabase:")
    print("    url: 'https://your-project-id.supabase.co'")
    print("    anon_key: 'your-anon-key-here'")

def test_video_alignments():
    """
    Test video alignment functionality
    """
    print("\n🔧 Testing video alignment functionality...")
    
    try:
        # Initialize video alignment database
        video_db = VideoAlignmentDatabase()
        video_db.create_table()
        
        # Test inserting video alignment data with updated schema
        test_videos = [
            ("cam_1.mp4", 12.494, 75.0, 1),  # camera_type 1
            ("cam_2.mp4", 11.045, 75.0, 2),  # camera_type 2
            ("cam_3.mp4", 0.0, 75.0, 3)      # camera_type 3 (reference)
        ]
        
        for source, start_offset, matching_dur, camera_type in test_videos:
            success = video_db.insert_video_alignment(source, start_offset, matching_dur, camera_type)
            if success:
                print(f"✅ Inserted alignment data for {source}")
            else:
                print(f"❌ Failed to insert alignment data for {source}")
        
        # Test retrieving video alignment data
        for source, _, _, _ in test_videos:
            data = video_db.get_video_alignment(source)
            if data:
                print(f"✅ Retrieved data for {source}: start={data['start_time_offset']}s, camera_type={data['camera_type']}")
            else:
                print(f"❌ Failed to retrieve data for {source}")
        
        # Test determining reference video
        video_sources = [video[0] for video in test_videos]
        reference_video = video_db.determine_reference_video(video_sources)
        print(f"✅ Determined reference video: {reference_video}")
        
        # Test getting all alignments
        all_alignments = video_db.get_all_video_alignments()
        print(f"✅ Retrieved {len(all_alignments)} total video alignment records")
        
        # Test getting alignments by camera type
        camera_type_1_alignments = video_db.get_alignments_by_camera_type(1)
        print(f"✅ Retrieved {len(camera_type_1_alignments)} alignments for camera type 1")
        
    except Exception as e:
        print(f"❌ Video alignment test failed: {e}")

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
            bad_gestures={"low_wrists": True},
            processing_time_ms=50,
            hand_model="mediapipe",
            pose_model="yolo"
        )
        
        if success:
            print("✅ Test data inserted successfully!")
            
            # Test data retrieval
            data = db.get_session_data(test_session_id)
            if data:
                print(f"✅ Retrieved {len(data)} records")
                
                # Test video alignment functionality with new class
                test_video_alignments()
                print("🧪 Test completed successfully!")
            else:
                print("❌ Failed to retrieve test data")
        else:
            print("❌ Failed to insert test data")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        setup_environment()

if __name__ == "__main__":
    # Check if config has Supabase credentials set
    supabase_config = CONFIG.get('database', {}).get('supabase', {})
    if not supabase_config.get('url') or not supabase_config.get('anon_key'):
        setup_environment()
    else:
        test_connection() 