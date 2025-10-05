#!/usr/bin/env python3
"""
Test script to check transcript_video table records
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.database.database_setup_v2 import DatabaseManager, TranscriptVideo


def test_transcript_table():
    """Test reading from transcript_video table"""

    print("=" * 80)
    print("TESTING TRANSCRIPT_VIDEO TABLE")
    print("=" * 80)

    try:
        # Initialize database manager
        print("\n📊 Initializing database connection...")
        db = DatabaseManager()

        print(f"✅ Connected to database: {db.connection_type}")
        print(f"   Using local database: {db.use_local}")

        # Query transcript_video table
        print("\n🔍 Querying transcript_video table...")

        if db.use_local:
            # Local PostgreSQL - use SQLAlchemy
            session = db.get_session()
            try:
                # Get all records
                all_records = session.query(TranscriptVideo).all()
                total_count = len(all_records)

                print(f"\n📊 Total records in transcript_video: {total_count}")

                if total_count == 0:
                    print("⚠️  No records found in transcript_video table")
                    print("\nPossible reasons:")
                    print("  1. Database is not enabled in config (database.enabled: false)")
                    print("  2. Transcript processing hasn't been run yet")
                    print("  3. Transcript storage is disabled (database.store_transcript_video: false)")
                else:
                    # Show unique video files
                    unique_videos = session.query(TranscriptVideo.video_file).distinct().all()
                    print(f"\n📹 Unique video files: {len(unique_videos)}")
                    for video in unique_videos:
                        video_name = video[0]
                        video_count = session.query(TranscriptVideo).filter(
                            TranscriptVideo.video_file == video_name
                        ).count()
                        print(f"   • {video_name}: {video_count} segments")

                    # Show latest 5 records
                    print(f"\n📝 Latest 5 transcript segments:")
                    latest_records = session.query(TranscriptVideo).order_by(
                        TranscriptVideo.created_at.desc()
                    ).limit(5).all()

                    for record in latest_records:
                        print(f"\n   ID: {record.id}")
                        print(f"   Video: {record.video_file}")
                        print(f"   Session: {record.session_id}")
                        print(f"   Time: {record.start_time:.2f}s - {record.end_time:.2f}s ({record.duration:.2f}s)")
                        print(f"   Text: {record.text[:100]}..." if len(record.text) > 100 else f"   Text: {record.text}")
                        print(f"   Words: {record.word_count}")
                        print(f"   Language: {record.language}")
                        print(f"   Model: {record.model_size}")

            finally:
                session.close()
        else:
            # Supabase
            result = db.supabase.table('transcript_video').select('*').execute()
            total_count = len(result.data) if result.data else 0

            print(f"\n📊 Total records in transcript_video: {total_count}")

            if total_count == 0:
                print("⚠️  No records found in transcript_video table")
            else:
                # Show unique video files
                unique_videos = set(r['video_file'] for r in result.data)
                print(f"\n📹 Unique video files: {len(unique_videos)}")
                for video_name in unique_videos:
                    video_count = sum(1 for r in result.data if r['video_file'] == video_name)
                    print(f"   • {video_name}: {video_count} segments")

                # Show latest 5 records
                print(f"\n📝 Latest 5 transcript segments:")
                sorted_data = sorted(result.data, key=lambda x: x.get('created_at', ''), reverse=True)[:5]

                for record in sorted_data:
                    print(f"\n   ID: {record['id']}")
                    print(f"   Video: {record['video_file']}")
                    print(f"   Session: {record['session_id']}")
                    print(f"   Time: {record['start_time']:.2f}s - {record['end_time']:.2f}s ({record['duration']:.2f}s)")
                    text = record['text']
                    print(f"   Text: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")
                    print(f"   Words: {record['word_count']}")
                    print(f"   Language: {record['language']}")
                    print(f"   Model: {record['model_size']}")

        print("\n" + "=" * 80)
        print("✅ Test completed successfully")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error testing transcript table: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_query_by_video(video_file: str):
    """Test querying transcript segments for a specific video"""

    print("\n" + "=" * 80)
    print(f"QUERYING TRANSCRIPTS FOR: {video_file}")
    print("=" * 80)

    try:
        db = DatabaseManager()

        if db.use_local:
            session = db.get_session()
            try:
                records = session.query(TranscriptVideo).filter(
                    TranscriptVideo.video_file == video_file
                ).order_by(TranscriptVideo.start_time).all()

                print(f"\n📊 Found {len(records)} transcript segments for {video_file}")

                for record in records:
                    print(f"\n   [{record.start_time:.2f}s - {record.end_time:.2f}s]: {record.text}")

            finally:
                session.close()
        else:
            result = db.supabase.table('transcript_video').select('*').eq(
                'video_file', video_file
            ).order('start_time').execute()

            records = result.data if result.data else []
            print(f"\n📊 Found {len(records)} transcript segments for {video_file}")

            for record in records:
                print(f"\n   [{record['start_time']:.2f}s - {record['end_time']:.2f}s]: {record['text']}")

        return records

    except Exception as e:
        print(f"\n❌ Error querying transcript for video: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test transcript_video table')
    parser.add_argument('--video', type=str, help='Query transcripts for specific video file')

    args = parser.parse_args()

    # Run main test
    success = test_transcript_table()

    # If video specified, query for that video
    if args.video and success:
        test_query_by_video(args.video)
