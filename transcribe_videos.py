#!/usr/bin/env python3
"""
Standalone Audio Transcription Script
Transcribes video files and saves results to database

Usage:
    python3 transcribe_videos.py --video-dir "/path/to/videos" --participant-id "P05"
    python3 transcribe_videos.py --video-file "/path/to/video.mp4" --participant-id "P05"
"""

import argparse
import os
import sys
import glob
import whisper
import torch
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_batch

# mlx-community repos per openai-whisper size name (Apple-GPU path)
MLX_MODELS = {
    'tiny': 'mlx-community/whisper-tiny-mlx',
    'base': 'mlx-community/whisper-base-mlx',
    'small': 'mlx-community/whisper-small-mlx',
    'medium': 'mlx-community/whisper-medium-mlx',
    'large': 'mlx-community/whisper-large-v3-mlx',
    'large-v3': 'mlx-community/whisper-large-v3-mlx',
}
ARGS_MODEL_SIZE = 'large-v3'

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'musician-tracking',
    'user': 'ngothanhnhan',
    'password': ''
}

def transcribe_video(video_path, model, language='en', chunk_duration=30):
    """
    Transcribe a single video file

    Args:
        video_path: Path to video file
        model: Whisper model instance
        language: Language code (default: 'en')
        chunk_duration: Not used in this simple version

    Returns:
        List of transcript segments with timestamps
    """
    print(f"🎤 Transcribing: {os.path.basename(video_path)}")

    try:
        if model == 'mlx':
            # Apple-GPU path (M-series): openai-whisper can't run on MPS
            # (float64/sparse ops), but mlx-whisper uses the GPU natively
            import mlx_whisper
            result = mlx_whisper.transcribe(
                video_path,
                path_or_hf_repo=MLX_MODELS.get(ARGS_MODEL_SIZE,
                                               'mlx-community/whisper-large-v3-mlx'),
                language=language,
                verbose=False,
                word_timestamps=True
            )
        else:
            # Transcribe using Whisper (CPU)
            result = model.transcribe(
                video_path,
                language=language,
                verbose=False,
                word_timestamps=True
            )

        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'confidence': segment.get('confidence', 0.0)
            })

        print(f"   ✅ Found {len(segments)} transcript segments")
        return segments

    except Exception as e:
        print(f"   ❌ Error transcribing: {e}")
        return []

def save_to_database(conn, video_file, participant_id, segments, session_id):
    """Save transcript segments to database"""

    if not segments:
        print("   ⚠️  No segments to save")
        return 0

    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcript_video (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(50) NOT NULL,
            video_file VARCHAR(255) NOT NULL,
            participant_id VARCHAR(10),
            segment_number INTEGER NOT NULL,
            start_time DECIMAL(10,3) NOT NULL,
            end_time DECIMAL(10,3) NOT NULL,
            text TEXT NOT NULL,
            confidence DECIMAL(4,3),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()

    # Prepare data for batch insert
    insert_data = []
    for i, segment in enumerate(segments, 1):
        insert_data.append((
            session_id,
            video_file,
            participant_id,
            i,
            segment['start'],
            segment['end'],
            segment['text'],
            segment.get('confidence', 0.0)
        ))

    # Batch insert
    insert_query = """
        INSERT INTO transcript_video
        (session_id, video_file, participant_id, segment_number, start_time, end_time, text, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    execute_batch(cursor, insert_query, insert_data, page_size=100)
    conn.commit()
    cursor.close()

    print(f"   ✅ Saved {len(segments)} segments to database")
    return len(segments)

def main():
    parser = argparse.ArgumentParser(description='Transcribe video files using Whisper')
    parser.add_argument('--video-dir', help='Directory containing video files')
    parser.add_argument('--video-file', help='Single video file to transcribe')
    parser.add_argument('--participant-id', required=True, help='Participant ID (e.g., P05)')
    parser.add_argument('--model-size', default='tiny', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: tiny)')
    parser.add_argument('--language', default='en', help='Language code (default: en)')

    args = parser.parse_args()

    # Validate input
    if not args.video_dir and not args.video_file:
        print("❌ Error: Must specify either --video-dir or --video-file")
        sys.exit(1)

    print()
    print("=" * 70)
    print("STANDALONE AUDIO TRANSCRIPTION")
    print("=" * 70)
    print(f"Participant ID: {args.participant_id}")
    print(f"Whisper Model: {args.model_size}")
    print(f"Language: {args.language}")
    print()

    # Load Whisper model — prefer mlx-whisper (Apple-GPU) when installed;
    # openai-whisper falls back to CPU on M-series (MPS breaks on float64)
    print(f"📥 Loading Whisper model '{args.model_size}'...")
    global ARGS_MODEL_SIZE
    ARGS_MODEL_SIZE = args.model_size
    try:
        import mlx_whisper  # noqa: F401
        model = 'mlx'
        print("   Using device: Apple GPU via mlx-whisper")
    except ImportError:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}"
              + ("  (tip: pip install mlx-whisper for Apple-GPU speed)"
                 if device == "cpu" else ""))
        try:
            model = whisper.load_model(args.model_size, device=device)
            print(f"   ✅ Model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            sys.exit(1)

    print()

    # Get list of video files
    if args.video_file:
        video_files = [args.video_file]
    else:
        video_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
        video_files.sort()

    if not video_files:
        print("❌ No video files found")
        sys.exit(1)

    print(f"📹 Found {len(video_files)} video file(s) to transcribe")
    print()

    # Connect to database
    print("📡 Connecting to database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("   ✅ Database connected")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        print("   Transcripts will be printed but not saved to database")
        conn = None

    print()

    # Session ID
    session_id = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Process each video
    total_segments = 0
    for i, video_path in enumerate(video_files, 1):
        print("-" * 70)
        print(f"[{i}/{len(video_files)}] {os.path.basename(video_path)}")
        print("-" * 70)

        # Transcribe
        segments = transcribe_video(video_path, model, args.language)

        # Save to database
        if conn and segments:
            saved = save_to_database(
                conn,
                os.path.basename(video_path),
                args.participant_id,
                segments,
                session_id
            )
            total_segments += saved

        # Print preview
        if segments:
            print()
            print("   📝 Transcript preview (first 3 segments):")
            for seg in segments[:3]:
                print(f"      [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
            if len(segments) > 3:
                print(f"      ... and {len(segments) - 3} more segments")

        print()

    # Summary
    print("=" * 70)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 70)
    print(f"✅ Processed {len(video_files)} video file(s)")
    print(f"✅ Total segments: {total_segments}")
    print(f"📊 Session ID: {session_id}")

    if conn:
        print()
        print("💾 Data saved to database table: transcript_video")
        print()
        print("To export to CSV:")
        print("   python3 export_to_csv.py")
        conn.close()

    print()

if __name__ == '__main__':
    main()
