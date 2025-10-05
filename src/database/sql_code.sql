-- Combined SQL Schema for Musician Tracking Database
-- Run this in Supabase SQL Editor

-- =====================================================================
-- MUSICIAN FRAME ANALYSIS TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS musician_frame_analysis (
    id BIGSERIAL PRIMARY KEY,
    
    -- Session metadata
    session_id VARCHAR(50) NOT NULL,
    video_file VARCHAR(255),
    frame_number INTEGER NOT NULL,
    original_time DECIMAL(10,3),  -- Original time in video (seconds)
    synced_time DECIMAL(10,3),    -- Synced time after alignment (seconds)
    
    -- Hand landmarks (JSON for flexibility)
    left_hand_landmarks JSONB,   -- [{x, y, z, confidence}, ...] 21 points
    right_hand_landmarks JSONB,  -- [{x, y, z, confidence}, ...] 21 points
    
    -- Pose landmarks (JSON)
    pose_landmarks JSONB,        -- [{x, y, z, confidence}, ...] 17-33 points
    
    -- Face mesh landmarks (JSON)
    facemesh_landmarks JSONB,    -- [{x, y, z, confidence}, ...] 468 points
    
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
    
    -- Model information
    hand_model VARCHAR(50),      -- e.g., 'mediapipe', 'yolo'
    pose_model VARCHAR(50),      -- e.g., 'mediapipe', 'yolo'
    facemesh_model VARCHAR(50),  -- e.g., 'mediapipe', 'yolo+mediapipe', 'none'
    emotion_model VARCHAR(50),   -- e.g., 'deepface', 'ghostfacenet', 'none'
    
    -- Analysis metadata  
    processing_time_ms DECIMAL(10,3),                    -- Total processing time
    hand_processing_time_ms DECIMAL(10,3) DEFAULT 0,     -- Time spent on hand detection
    pose_processing_time_ms DECIMAL(10,3) DEFAULT 0,     -- Time spent on pose detection
    facemesh_processing_time_ms DECIMAL(10,3) DEFAULT 0, -- Time spent on face mesh detection
    emotion_processing_time_ms DECIMAL(10,3) DEFAULT 0,  -- Time spent on emotion detection
    bad_gesture_processing_time_ms DECIMAL(10,3) DEFAULT 0,  -- Time spent on bad gesture detection
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_session_frame ON musician_frame_analysis(session_id, frame_number);
CREATE INDEX IF NOT EXISTS idx_session_time ON musician_frame_analysis(session_id, synced_time);
CREATE INDEX IF NOT EXISTS idx_synced_time ON musician_frame_analysis(synced_time);
CREATE INDEX IF NOT EXISTS idx_bad_gestures ON musician_frame_analysis(flag_low_wrists, flag_turtle_neck, flag_hunched_back, flag_fingers_pointing_up);
CREATE INDEX IF NOT EXISTS idx_created_at ON musician_frame_analysis(created_at);
CREATE INDEX IF NOT EXISTS idx_models ON musician_frame_analysis(hand_model, pose_model, emotion_model);
CREATE INDEX IF NOT EXISTS idx_transcript_segment ON musician_frame_analysis(transcript_segment_id);

-- =====================================================================
-- VIDEO ALIGNMENT OFFSETS TABLE
-- =====================================================================
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

-- =====================================================================
-- MIGRATION COMMANDS (if updating existing tables)
-- =====================================================================

-- Add new columns to existing musician_frame_analysis table
ALTER TABLE musician_frame_analysis 
ADD COLUMN IF NOT EXISTS original_time DECIMAL(10,3),
ADD COLUMN IF NOT EXISTS synced_time DECIMAL(10,3),
ADD COLUMN IF NOT EXISTS hand_model VARCHAR(50),
ADD COLUMN IF NOT EXISTS pose_model VARCHAR(50),
ADD COLUMN IF NOT EXISTS facemesh_model VARCHAR(50),
ADD COLUMN IF NOT EXISTS emotion_model VARCHAR(50),
ADD COLUMN IF NOT EXISTS hand_processing_time_ms INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS pose_processing_time_ms INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS facemesh_processing_time_ms INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS emotion_processing_time_ms INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS facemesh_landmarks JSONB,
ADD COLUMN IF NOT EXISTS transcript_segment_id INTEGER;

-- Remove old model_version column if it exists
ALTER TABLE musician_frame_analysis 
DROP COLUMN IF EXISTS model_version;

-- Add matching_duration column to video_alignment_offsets if needed
ALTER TABLE video_alignment_offsets 
ADD COLUMN IF NOT EXISTS matching_duration DECIMAL(10,3) NOT NULL DEFAULT 0.000;

-- Drop old end_time_offset column if it exists
ALTER TABLE video_alignment_offsets 
DROP COLUMN IF EXISTS end_time_offset;

-- =====================================================================
-- VERIFICATION QUERIES
-- =====================================================================

-- Verify musician_frame_analysis table structure
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'musician_frame_analysis'
ORDER BY ordinal_position;

-- Verify video_alignment_offsets table structure  
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'video_alignment_offsets'
ORDER BY ordinal_position;

-- Optional: Create RLS (Row Level Security) policies
-- ALTER TABLE musician_frame_analysis ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE video_alignment_offsets ENABLE ROW LEVEL SECURITY;

-- Example policies for musician_frame_analysis
-- CREATE POLICY "Users can view own sessions" ON musician_frame_analysis
--     FOR SELECT USING (auth.uid()::text = session_id OR session_id LIKE 'public_%');

-- CREATE POLICY "Users can insert own sessions" ON musician_frame_analysis
--     FOR INSERT WITH CHECK (auth.uid()::text = session_id OR session_id LIKE 'public_%');

-- Grant necessary permissions (adjust based on your needs)
-- GRANT ALL ON musician_frame_analysis TO authenticated;
-- GRANT ALL ON musician_frame_analysis TO anon;
-- GRANT ALL ON video_alignment_offsets TO authenticated;
-- GRANT ALL ON video_alignment_offsets TO anon;

-- =====================================================================
-- TRANSCRIPT VIDEO TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS transcript_video (
    id BIGSERIAL PRIMARY KEY,
    
    -- Video metadata
    video_file VARCHAR(255) NOT NULL,
    session_id VARCHAR(50),
    
    -- Transcript segment data
    segment_id INTEGER NOT NULL,  -- Whisper segment index within the video
    start_time DECIMAL(10,3) NOT NULL,  -- Start time in seconds
    end_time DECIMAL(10,3) NOT NULL,    -- End time in seconds
    duration DECIMAL(8,3) NOT NULL,     -- Duration of this segment
    
    -- Transcript text
    text TEXT NOT NULL,              -- Full transcript text for this segment
    word_count INTEGER DEFAULT 0,   -- Number of words in this segment
    
    -- Confidence and quality metrics
    avg_logprob DECIMAL(8,4),       -- Average log probability from Whisper
    no_speech_prob DECIMAL(6,4),    -- No speech probability from Whisper
    
    -- Processing metadata
    language VARCHAR(10),            -- Detected/specified language
    model_size VARCHAR(20),          -- Whisper model used (tiny, base, small, etc.)
    chunk_duration DECIMAL(6,2),    -- Audio chunk duration used for processing
    
    -- Word-level timestamps (optional, for advanced features)
    words_json JSONB,               -- [{"word": "hello", "start": 1.0, "end": 1.2, "probability": 0.95}, ...]
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for transcript_video table
CREATE INDEX IF NOT EXISTS idx_transcript_video_file ON transcript_video(video_file);
CREATE INDEX IF NOT EXISTS idx_transcript_session ON transcript_video(session_id);
CREATE INDEX IF NOT EXISTS idx_transcript_time_range ON transcript_video(video_file, start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_transcript_start_time ON transcript_video(start_time);
CREATE INDEX IF NOT EXISTS idx_transcript_segment ON transcript_video(video_file, segment_id);
CREATE INDEX IF NOT EXISTS idx_transcript_created_at ON transcript_video(created_at);

-- =====================================================================
-- ALTERNATIVE: Add transcript_text column to existing musician_frame_analysis table
-- =====================================================================
-- Option 1: Add transcript column to existing table (simpler approach)
ALTER TABLE musician_frame_analysis 
ADD COLUMN IF NOT EXISTS transcript_text TEXT;

-- Create index for transcript search
CREATE INDEX IF NOT EXISTS idx_transcript_text ON musician_frame_analysis 
USING gin(to_tsvector('english', transcript_text));

-- =====================================================================
-- CHUNK VIDEO ALIGNMENT OFFSETS TABLE
-- =====================================================================
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

-- Composite index for efficient queries
CREATE INDEX IF NOT EXISTS idx_chunk_source_camera_order ON chunk_video_alignment_offsets(source, camera_prefix, chunk_order);

-- =====================================================================
-- PROCESSING JOBS TABLE
-- =====================================================================
CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id VARCHAR(100) PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    message TEXT,
    request_data JSONB,
    output_files JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    processing_time_seconds DECIMAL(10,2)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_job_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_job_type ON processing_jobs(type);
CREATE INDEX IF NOT EXISTS idx_job_created_at ON processing_jobs(created_at);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE processing_jobs ENABLE ROW LEVEL SECURITY;

-- Create a policy to allow all operations (adjust based on your needs)
CREATE POLICY "Allow all operations on processing_jobs" ON processing_jobs
    FOR ALL USING (true) WITH CHECK (true);

-- Add comment to table
COMMENT ON TABLE processing_jobs IS 'Tracks video processing jobs and their status';

-- Add comments to columns
COMMENT ON COLUMN processing_jobs.job_id IS 'Primary key - Unique job identifier (UUID)';
COMMENT ON COLUMN processing_jobs.type IS 'Job type: single_video or folder_processing';
COMMENT ON COLUMN processing_jobs.status IS 'Job status: queued, running, completed, failed, cancelled';
COMMENT ON COLUMN processing_jobs.progress IS 'Progress percentage (0-100)';
COMMENT ON COLUMN processing_jobs.message IS 'Current status message';
COMMENT ON COLUMN processing_jobs.request_data IS 'Original request parameters as JSON';
COMMENT ON COLUMN processing_jobs.output_files IS 'Array of output file paths as JSON';
COMMENT ON COLUMN processing_jobs.error_message IS 'Error details if job failed';
COMMENT ON COLUMN processing_jobs.created_at IS 'Job creation timestamp';
COMMENT ON COLUMN processing_jobs.updated_at IS 'Last update timestamp';
COMMENT ON COLUMN processing_jobs.completed_at IS 'Job completion timestamp';
COMMENT ON COLUMN processing_jobs.processing_time_seconds IS 'Total processing time in seconds';

-- =====================================================================
-- DELETE QUERIES
-- =====================================================================

-- Delete all records with specific video file
DELETE FROM musician_frame_analysis
WHERE video_file = 'vid_shot1_cam_1.mp4';

-- Delete transcript records for specific video
DELETE FROM transcript_video
WHERE video_file = 'vid_shot1_cam_1.mp4';

-- Delete chunk alignment records for specific source
DELETE FROM chunk_video_alignment_offsets
WHERE source = 'vid_shot1';