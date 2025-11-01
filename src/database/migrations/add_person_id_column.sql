-- Migration: Add person_id column to musician_frame_analysis table
-- Purpose: Track which person (0, 1, etc.) each record belongs to in multi-person videos
-- Date: 2025-01-28

-- =====================================================================
-- ADD PERSON_ID COLUMN
-- =====================================================================

-- Add person_id column (nullable initially for backward compatibility)
ALTER TABLE musician_frame_analysis
ADD COLUMN IF NOT EXISTS person_id INTEGER DEFAULT 0;

-- Add comment to explain the column
COMMENT ON COLUMN musician_frame_analysis.person_id IS 'Person identifier in multi-person videos (0=left person, 1=right person, etc.). Sorted left-to-right by position.';

-- Create index for efficient queries by person
CREATE INDEX IF NOT EXISTS idx_person_id ON musician_frame_analysis(person_id);

-- Create composite index for session + person queries
CREATE INDEX IF NOT EXISTS idx_session_person ON musician_frame_analysis(session_id, person_id, frame_number);

-- =====================================================================
-- UPDATE EXISTING RECORDS
-- =====================================================================

-- Set person_id = 0 for all existing records (assuming single-person videos)
UPDATE musician_frame_analysis
SET person_id = 0
WHERE person_id IS NULL;

-- Now make the column NOT NULL
ALTER TABLE musician_frame_analysis
ALTER COLUMN person_id SET NOT NULL;

-- =====================================================================
-- VERIFICATION
-- =====================================================================

-- Verify the column was added
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'musician_frame_analysis'
AND column_name = 'person_id';

-- Check index creation
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'musician_frame_analysis'
AND indexname LIKE '%person%';

-- Sample query to verify data
SELECT
    session_id,
    person_id,
    frame_number,
    COUNT(*) as record_count
FROM musician_frame_analysis
GROUP BY session_id, person_id, frame_number
ORDER BY session_id, person_id, frame_number
LIMIT 10;
