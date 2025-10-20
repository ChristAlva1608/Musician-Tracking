#!/usr/bin/env python3
"""
2D Camera Video Renaming Tool

Handles iPhone and GoPro videos by renaming them to the chunk### format.
Can optionally transcode to standardize format and resolution.

Usage:
    python camera_2d_renamer.py --participant P03 --mode rename
    python camera_2d_renamer.py --participant P03 --mode transcode
"""

import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import re
import json


class Camera2DRenamer:
    """
    Processes 2D camera videos (iPhone, GoPro) for pose detection workflow.
    """

    def __init__(self, data_root: str, output_root: str):
        """
        Initialize renamer.

        Args:
            data_root: Path to participant data
            output_root: Path to output processed videos
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.log_file = self.output_root / "2d_camera_log.csv"

    def scan_2d_cameras(self, participant_id: str, date_folder: str = None) -> dict:
        """
        Scan participant folder for 2D camera videos.

        Returns:
            Dictionary with iPhone and GoPro videos sorted by timestamp
        """
        participant_folder = self.data_root / participant_id

        if not participant_folder.exists():
            print(f"Error: Participant folder not found: {participant_folder}")
            return {}

        # Find date folder if not specified
        if date_folder is None:
            date_folders = sorted([d for d in participant_folder.iterdir() if d.is_dir()])
            if not date_folders:
                print(f"Error: No date folders found in {participant_folder}")
                return {}
            date_folder = date_folders[0].name

        data_folder = participant_folder / date_folder

        videos = {
            'iPhone': [],
            'GoPro': []
        }

        # Scan all camera folders
        for camera_folder in sorted(data_folder.iterdir()):
            if not camera_folder.is_dir():
                continue

            camera_type = None
            if 'iPhone' in camera_folder.name or 'iphone' in camera_folder.name:
                camera_type = 'iPhone'
            elif 'GoPro' in camera_folder.name or 'gopro' in camera_folder.name.lower():
                camera_type = 'GoPro'

            if camera_type is None:
                continue

            # Find video files with their creation times
            for video_file in camera_folder.glob('*'):
                if video_file.suffix.lower() in ['.mp4', '.mov']:
                    # Get file creation timestamp
                    timestamp = video_file.stat().st_birthtime

                    videos[camera_type].append({
                        'path': video_file,
                        'timestamp': timestamp,
                        'name': video_file.name
                    })

        # Sort by timestamp
        for camera_type in videos:
            videos[camera_type] = sorted(videos[camera_type], key=lambda x: x['timestamp'])

        return videos

    def rename_videos(self, participant_id: str, date_folder: str = None,
                     start_chunk: int = 1, mode: str = 'copy'):
        """
        Rename 2D camera videos to chunk### format.

        Args:
            participant_id: Participant ID (e.g., P03)
            date_folder: Date folder name
            start_chunk: Starting chunk number (default: 1)
            mode: 'copy' (copy files) or 'transcode' (re-encode)
        """
        print(f"\n{'='*60}")
        print(f"📹 Processing 2D Cameras for {participant_id}")
        print(f"{'='*60}")

        videos = self.scan_2d_cameras(participant_id, date_folder)

        log_entries = []

        # Determine camera numbers (continue from 360° cameras)
        # Assuming Cam1-3 for 360° (X3, X4, X5), start 2D from Cam4
        camera_mapping = {
            'iPhone': 4,
            'GoPro': 5
        }

        for camera_type in ['iPhone', 'GoPro']:
            if not videos[camera_type]:
                continue

            cam_num = camera_mapping[camera_type]
            print(f"\n📱 {camera_type} (Cam{cam_num}) - {len(videos[camera_type])} videos")

            for idx, video_info in enumerate(videos[camera_type], start=start_chunk):
                input_path = video_info['path']
                chunk_num = f"{idx:03d}"

                # Output path
                output_dir = self.output_root / participant_id / date_folder / f"Cam{cam_num}"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{participant_id}_Cam{cam_num}_chunk{chunk_num}.mp4"

                print(f"   Chunk {chunk_num}: {input_path.name}")
                print(f"   → {output_file.name}")

                success = False
                if mode == 'copy':
                    # Simple copy
                    try:
                        shutil.copy2(input_path, output_file)
                        success = True
                        print(f"   ✅ Copied")
                    except Exception as e:
                        print(f"   ❌ Copy failed: {e}")
                elif mode == 'transcode':
                    # Transcode to standardize format
                    success = self._transcode_video(input_path, output_file)

                # Log
                log_entries.append({
                    'timestamp': datetime.now().isoformat(),
                    'participant': participant_id,
                    'camera': camera_type,
                    'cam_number': cam_num,
                    'chunk': chunk_num,
                    'input': str(input_path),
                    'output': str(output_file),
                    'mode': mode,
                    'success': success
                })

        # Save log
        self._save_log(log_entries)
        print(f"\n✅ 2D camera processing complete! Log: {self.log_file}")

    def _transcode_video(self, input_path: Path, output_path: Path) -> bool:
        """
        Transcode video to standardized format.

        Settings:
        - Resolution: 1920x1080 (or original if smaller)
        - Codec: H.264
        - CRF: 23 (good quality)
        """
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ Transcoded")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Transcode failed: {e}")
            return False

    def _save_log(self, entries: list):
        """Save processing log to CSV."""
        import csv

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.log_file.exists()

        with open(self.log_file, 'a', newline='') as f:
            fieldnames = ['timestamp', 'participant', 'camera', 'cam_number',
                         'chunk', 'input', 'output', 'mode', 'success']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerows(entries)


def main():
    parser = argparse.ArgumentParser(description='Process 2D camera videos (iPhone, GoPro)')
    parser.add_argument('--mode', choices=['rename', 'transcode'], required=True,
                       help='rename: copy files with new names, transcode: re-encode to standard format')
    parser.add_argument('--participant', required=True,
                       help='Participant ID (e.g., P03)')
    parser.add_argument('--date-folder', help='Date folder name (auto-detected if not provided)')
    parser.add_argument('--start-chunk', type=int, default=1,
                       help='Starting chunk number (default: 1)')
    parser.add_argument('--data-root', default='/Volumes/X10 Pro 1/Phan Dissertation Data',
                       help='Root path to participant data')
    parser.add_argument('--output-root', default='./Processed_Videos',
                       help='Root path for output processed videos')

    args = parser.parse_args()

    renamer = Camera2DRenamer(args.data_root, args.output_root)

    # Map mode to internal format
    process_mode = 'copy' if args.mode == 'rename' else 'transcode'

    renamer.rename_videos(args.participant, args.date_folder, args.start_chunk, process_mode)


if __name__ == '__main__':
    main()
