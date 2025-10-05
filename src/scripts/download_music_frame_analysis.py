#!/usr/bin/env python3
"""
Download all data from musician_frame_analysis table in Supabase to CSV
"""

import os
import sys
import csv
import json
import time
from datetime import datetime
import yaml
from pathlib import Path
from supabase import create_client, Client
from typing import Optional, List, Dict, Any

# Load config from YAML
config_path = Path(__file__).parent.parent / 'config' / 'config_v1.yaml'
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

class MusicFrameAnalysisExporter:
    def __init__(self):
        """Initialize Supabase connection"""
        supabase_config = CONFIG.get('database', {}).get('supabase', {})
        self.supabase_url = supabase_config.get('url')
        self.supabase_key = supabase_config.get('anon_key')

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing Supabase credentials. Please set database.supabase.url and database.supabase.anon_key in config_v1.yaml"
            )

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        print(f"✅ Connected to Supabase")

    def fetch_all_data(self, table_name: str = 'musician_frame_analysis', batch_size: int = 300, max_retries: int = 5) -> List[Dict]:
        """
        Fetch all data from the table in batches with retry logic and timeout handling

        Args:
            table_name: Name of the table to fetch from
            batch_size: Number of records to fetch per batch (default 300 to avoid timeouts)
            max_retries: Maximum number of retries for failed fetches

        Returns:
            List of all records
        """
        all_data = []
        offset = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        batch_number = 0

        print(f"📥 Fetching data from '{table_name}' table...")
        print(f"   Batch size: {batch_size} records")
        print(f"   Max retries per batch: {max_retries}")
        print(f"   This may take a while for large datasets...\n")

        # First, get total count if possible
        try:
            count_result = self.supabase.table(table_name).select('id', count='exact').execute()
            total_count = count_result.count
            if total_count:
                print(f"   Total records in table: {total_count:,}")
                estimated_batches = (total_count + batch_size - 1) // batch_size
                print(f"   Estimated batches: {estimated_batches}\n")
        except:
            print("   Could not determine total count\n")
            total_count = None

        while True:
            retry_count = 0
            batch_fetched = False
            batch_number += 1

            while retry_count < max_retries and not batch_fetched:
                try:
                    # Fetch a batch of data with ordering by id for consistency
                    result = self.supabase.table(table_name).select('*').order('id').range(
                        offset,
                        offset + batch_size - 1
                    ).execute()

                    if not result.data:
                        # No more data to fetch
                        if offset == 0:
                            print(f"  No data found in table")
                        else:
                            print(f"\n  ✓ Reached end of data")
                        return all_data

                    all_data.extend(result.data)
                    fetched = len(result.data)

                    # Progress indicator
                    progress = f"  Batch {batch_number:4d}: Fetched {fetched:4d} records | Total: {len(all_data):,}"
                    if total_count:
                        percentage = (len(all_data) / total_count) * 100
                        progress += f" ({percentage:.1f}%)"
                    print(progress)

                    # Reset consecutive error counter on success
                    consecutive_errors = 0
                    batch_fetched = True

                    # If we got fewer records than batch_size, we've reached the end
                    if fetched < batch_size:
                        print(f"\n  ✓ Reached end of data (last batch had {fetched} records)")
                        return all_data

                    offset += batch_size

                    # Small delay to avoid rate limiting
                    time.sleep(0.05)

                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)

                    if 'timeout' in error_msg.lower() or '57014' in error_msg:
                        print(f"\n  ⚠️  Timeout at offset {offset} (attempt {retry_count}/{max_retries})")

                        if retry_count < max_retries:
                            # Try with smaller batch size
                            temp_batch_size = max(50, batch_size // 3)
                            print(f"      Retrying with smaller batch size: {temp_batch_size}")

                            try:
                                # Attempt with smaller batch
                                result = self.supabase.table(table_name).select('*').order('id').range(
                                    offset,
                                    offset + temp_batch_size - 1
                                ).execute()

                                if result.data:
                                    all_data.extend(result.data)
                                    fetched = len(result.data)
                                    print(f"      ✓ Success with smaller batch: fetched {fetched} records")
                                    offset += fetched
                                    batch_fetched = True
                                    consecutive_errors = 0
                                    continue
                            except Exception as retry_error:
                                print(f"      Failed with smaller batch: {retry_error}")

                            # Wait before retry with exponential backoff
                            wait_time = min(retry_count * 2, 10)
                            print(f"      Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                    else:
                        print(f"\n  ❌ Error at offset {offset} (attempt {retry_count}/{max_retries}): {e}")

                        if retry_count < max_retries:
                            wait_time = min(retry_count, 5)
                            print(f"      Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)

            if not batch_fetched:
                consecutive_errors += 1
                print(f"\n  ⚠️  Failed to fetch batch at offset {offset} after {max_retries} attempts")
                print(f"      Consecutive errors: {consecutive_errors}/{max_consecutive_errors}")

                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n  ❌ Too many consecutive errors. Stopping fetch.")
                    print(f"     Successfully fetched {len(all_data):,} records before stopping")
                    break

                # Try to skip this problematic batch
                print(f"      Skipping batch and continuing...")
                offset += batch_size
                time.sleep(2)

        return all_data

    def export_to_csv(self, data: List[Dict], output_file: str = 'music_frame_analysis.csv'):
        """
        Export data to CSV file

        Args:
            data: List of records to export
            output_file: Path to output CSV file
        """
        if not data:
            print("⚠️ No data to export")
            return

        # Get all unique column names from all records
        all_columns = set()
        for record in data:
            all_columns.update(record.keys())

        # Sort columns for consistent output
        columns = sorted(list(all_columns))

        # Move 'id' to the beginning if it exists
        if 'id' in columns:
            columns.remove('id')
            columns.insert(0, 'id')

        # Move session and frame info to the beginning
        priority_cols = ['session_id', 'video_file', 'frame_number', 'original_time', 'synced_time']
        for col in reversed(priority_cols):
            if col in columns:
                columns.remove(col)
                columns.insert(1 if 'id' in columns else 0, col)

        print(f"📝 Writing {len(data)} records to '{output_file}'...")
        print(f"   Columns: {len(columns)}")

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            for i, record in enumerate(data):
                # Convert JSON/JSONB fields to string representation
                processed_record = {}
                for col in columns:
                    value = record.get(col)

                    # Handle JSON/list/dict types
                    if isinstance(value, (list, dict)):
                        processed_record[col] = json.dumps(value, ensure_ascii=False)
                    elif value is None:
                        processed_record[col] = ''
                    else:
                        processed_record[col] = value

                writer.writerow(processed_record)

                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"   Written {i + 1} records...")

        print(f"✅ Successfully exported data to '{output_file}'")

        # Print file size
        file_size = os.path.getsize(output_file)
        if file_size > 1024 * 1024:
            print(f"   File size: {file_size / (1024 * 1024):.2f} MB")
        else:
            print(f"   File size: {file_size / 1024:.2f} KB")

def main():
    """Main function with support for large datasets"""
    import argparse

    parser = argparse.ArgumentParser(description='Download musician_frame_analysis data from Supabase')
    parser.add_argument('--batch-size', type=int, default=300,
                       help='Number of records to fetch per batch (default: 300)')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Maximum retries per batch (default: 5)')
    parser.add_argument('--table', type=str, default='musician_frame_analysis',
                       help='Table name to fetch from (default: musician_frame_analysis)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV filename (default: auto-generated with timestamp)')

    args = parser.parse_args()

    try:
        # Initialize exporter
        print("🚀 Initializing database connection...")
        exporter = MusicFrameAnalysisExporter()

        # Fetch all data with custom parameters
        print(f"\n🎯 Fetching from table: '{args.table}'")
        data = exporter.fetch_all_data(
            table_name=args.table,
            batch_size=args.batch_size,
            max_retries=args.max_retries
        )

        if data:
            # Determine output filename
            if args.output:
                output_file = args.output
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'{args.table}_{timestamp}.csv'

            # Export to CSV
            exporter.export_to_csv(data, output_file)

            # Also create a copy without timestamp for convenience
            if not args.output:
                simple_filename = f'{args.table}.csv'
                print(f"\n📄 Creating convenience copy: '{simple_filename}'")
                exporter.export_to_csv(data, simple_filename)

            # Print summary statistics
            print("\n📊 Data Summary:")
            print(f"   Total records: {len(data):,}")

            # Count unique sessions
            sessions = set(record.get('session_id') for record in data if record.get('session_id'))
            if sessions:
                print(f"   Unique sessions: {len(sessions)}")

            # Count unique video files
            videos = set(record.get('video_file') for record in data if record.get('video_file'))
            if videos:
                print(f"   Unique video files: {len(videos)}")

            # Frame number range
            frame_numbers = [record.get('frame_number') for record in data if record.get('frame_number') is not None]
            if frame_numbers:
                print(f"   Frame range: {min(frame_numbers):,} - {max(frame_numbers):,}")

            # Date range
            dates = [record.get('created_at') for record in data if record.get('created_at')]
            if dates:
                dates.sort()
                print(f"   Date range: {dates[0]} to {dates[-1]}")

            print("\n✅ Export complete!")
            print(f"   Output file: {output_file}")

        else:
            print("⚠️ No data found in the table")

    except KeyboardInterrupt:
        print("\n\n⚠️  Export interrupted by user")
        print("   Partial data may have been saved")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()