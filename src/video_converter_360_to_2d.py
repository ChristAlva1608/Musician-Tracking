#!/usr/bin/env python3
"""
360° Video to 2D Perspective Conversion Tool

This tool helps convert 360° equirectangular videos to 2D perspective format
optimized for pose detection with MediaPipe.

Features:
- Interactive angle selection with video preview
- Batch processing with saved camera angles
- Progress tracking and logging
- Handles both 360° cameras (X3, X4, X5) and 2D cameras (iPhone, GoPro)
- Automatic chunk renaming for 2D cameras

Usage:
    python video_converter_360_to_2d.py --mode interactive --participant P03
    python video_converter_360_to_2d.py --mode batch --participant P03
    python video_converter_360_to_2d.py --mode rename-2d --participant P03
"""

import argparse
import json
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
import re
import cv2
import numpy as np

class VideoConverter360to2D:
    """
    Converts 360° videos to 2D perspective format for pose detection.
    """

    def __init__(self, data_root: str, output_root: str):
        """
        Initialize converter.

        Args:
            data_root: Path to participant data (e.g., "/Volumes/X10 Pro 1/Phan Dissertation Data")
            output_root: Path to output processed videos
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.angles_file = self.output_root / "camera_angles.json"
        self.log_file = self.output_root / "conversion_log.csv"

        # Conversion settings optimized for pose detection
        self.default_settings = {
            "projection": "perspective",  # flat projection from equirectangular
            "fov_h": 120,  # Horizontal field of view (degrees)
            "fov_v": 90,   # Vertical field of view (degrees)
            "width": 1920,
            "height": 1080,
            "crf": 23,  # Quality (lower = better, 18-28 reasonable range)
            "preset": "medium"  # Encoding speed vs compression
        }

    def load_camera_angles(self) -> dict:
        """Load saved camera angle settings."""
        if self.angles_file.exists():
            with open(self.angles_file, 'r') as f:
                return json.load(f)
        return {}

    def save_camera_angles(self, angles: dict):
        """Save camera angle settings."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        with open(self.angles_file, 'w') as f:
            json.dump(angles, indent=2, fp=f)

    def get_video_info(self, video_path: Path) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)

            # Extract relevant info
            video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
            if not video_stream:
                return None

            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                'duration': float(info['format'].get('duration', 0)),
                'codec': video_stream.get('codec_name', 'unknown')
            }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None

    def is_360_video(self, video_path: Path) -> bool:
        """
        Determine if video is 360° based on filename and metadata.

        360° cameras: Insta360 X3, X4, X5 (typically .insv or .mp4 with 2:1 aspect ratio)
        2D cameras: iPhone, GoPro (standard aspect ratios)
        """
        filename = video_path.name.lower()

        # Check file extension
        if filename.endswith('.insv'):
            return True

        # Check aspect ratio for .mp4 files
        info = self.get_video_info(video_path)
        if info and info['width'] > 0 and info['height'] > 0:
            aspect_ratio = info['width'] / info['height']
            # 360° equirectangular is typically 2:1 ratio
            if abs(aspect_ratio - 2.0) < 0.1:
                return True

        return False

    def preview_360_video(self, video_path: Path, yaw: int = 0, pitch: int = 0, roll: int = 0):
        """
        Show interactive preview of 360° video with current viewing angle.

        Args:
            video_path: Path to 360° video
            yaw: Horizontal angle (0-360°)
            pitch: Vertical angle (-90 to 90°)
            roll: Rotation angle (typically 0)
        """
        print(f"\n📹 Previewing: {video_path.name}")
        print(f"Current angle - Yaw: {yaw}°, Pitch: {pitch}°, Roll: {roll}°")
        print("\nControls:")
        print("  Arrow keys: Adjust yaw (left/right) and pitch (up/down)")
        print("  Q/E: Adjust roll")
        print("  +/-: Adjust FOV")
        print("  SPACE: Accept current angle")
        print("  ESC: Cancel")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None

        # Get middle frame for preview
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

        current_yaw = yaw
        current_pitch = pitch
        current_roll = roll
        current_fov = self.default_settings['fov_h']

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                continue

            # Convert 360° equirectangular to perspective view
            h, w = frame.shape[:2]

            # Create perspective projection using cv2 remap
            # This is a simplified version; FFmpeg does better job
            preview_frame = self._simple_equirect_to_perspective(
                frame, current_yaw, current_pitch, current_fov
            )

            # Add angle info overlay
            cv2.putText(preview_frame, f"Yaw: {current_yaw} Pitch: {current_pitch} FOV: {current_fov}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("360° Preview (Arrow keys to adjust, SPACE to accept, ESC to cancel)", preview_frame)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == ord(' '):  # SPACE
                cap.release()
                cv2.destroyAllWindows()
                return {'yaw': current_yaw, 'pitch': current_pitch, 'roll': current_roll, 'fov': current_fov}
            elif key == 81:  # Left arrow
                current_yaw = (current_yaw - 5) % 360
            elif key == 83:  # Right arrow
                current_yaw = (current_yaw + 5) % 360
            elif key == 82:  # Up arrow
                current_pitch = max(-90, current_pitch + 5)
            elif key == 84:  # Down arrow
                current_pitch = min(90, current_pitch - 5)
            elif key == ord('q'):
                current_roll = (current_roll - 5) % 360
            elif key == ord('e'):
                current_roll = (current_roll + 5) % 360
            elif key == ord('+') or key == ord('='):
                current_fov = min(180, current_fov + 5)
            elif key == ord('-') or key == ord('_'):
                current_fov = max(30, current_fov - 5)

        cap.release()
        cv2.destroyAllWindows()
        return None

    def _simple_equirect_to_perspective(self, equirect_img, yaw, pitch, fov):
        """
        Simple equirectangular to perspective conversion for preview.
        For actual conversion, FFmpeg is used which has better quality.
        """
        h, w = equirect_img.shape[:2]

        # Output size for preview (smaller for performance)
        out_h, out_w = 720, 1280

        # Convert angles to radians
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        fov_rad = np.deg2rad(fov)

        # Create output coordinate grid
        y_coords, x_coords = np.mgrid[0:out_h, 0:out_w]

        # Normalize to [-1, 1]
        x_norm = (x_coords - out_w / 2) / (out_w / 2)
        y_norm = (y_coords - out_h / 2) / (out_h / 2)

        # Apply FOV
        x_norm *= np.tan(fov_rad / 2)
        y_norm *= np.tan(fov_rad / 2) * (out_h / out_w)

        # Convert to 3D rays
        z = np.ones_like(x_norm)
        x = x_norm
        y = y_norm

        # Normalize rays
        norm = np.sqrt(x**2 + y**2 + z**2)
        x /= norm
        y /= norm
        z /= norm

        # Apply pitch rotation
        y_rot = y * np.cos(pitch_rad) - z * np.sin(pitch_rad)
        z_rot = y * np.sin(pitch_rad) + z * np.cos(pitch_rad)

        # Apply yaw rotation
        x_final = x * np.cos(yaw_rad) + z_rot * np.sin(yaw_rad)
        z_final = -x * np.sin(yaw_rad) + z_rot * np.cos(yaw_rad)

        # Convert to equirectangular coordinates
        theta = np.arctan2(x_final, z_final)
        phi = np.arcsin(np.clip(y_rot, -1, 1))

        # Map to pixel coordinates
        equirect_x = ((theta / np.pi + 1) * w / 2).astype(np.float32)
        equirect_y = ((phi / (np.pi / 2) + 1) * h / 2).astype(np.float32)

        # Remap
        result = cv2.remap(equirect_img, equirect_x, equirect_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        return result

    def convert_360_to_2d(self, input_video: Path, output_video: Path,
                         yaw: int = 0, pitch: int = 0, roll: int = 0,
                         fov: int = None) -> bool:
        """
        Convert 360° video to 2D perspective using FFmpeg.

        Args:
            input_video: Input 360° video path
            output_video: Output 2D video path
            yaw: Horizontal viewing angle (0-360°)
            pitch: Vertical viewing angle (-90 to 90°)
            roll: Rotation angle (typically 0)
            fov: Field of view (if None, uses default)

        Returns:
            True if successful, False otherwise
        """
        if fov is None:
            fov = self.default_settings['fov_h']

        output_video.parent.mkdir(parents=True, exist_ok=True)

        # FFmpeg v360 filter for 360° to perspective conversion
        # e = equirectangular input
        # flat = perspective output
        vf_filter = (
            f"v360=e:flat:"
            f"iv_fov=360:ih_fov=180:"  # Input FOV (full 360°)
            f"yaw={yaw}:pitch={pitch}:roll={roll}:"  # Viewing angle
            f"w={self.default_settings['width']}:"
            f"h={self.default_settings['height']}:"
            f"interp=linear"  # Interpolation method
        )

        cmd = [
            'ffmpeg', '-i', str(input_video),
            '-vf', vf_filter,
            '-c:v', 'libx264',
            '-preset', self.default_settings['preset'],
            '-crf', str(self.default_settings['crf']),
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-y',  # Overwrite output
            str(output_video)
        ]

        print(f"\n🎬 Converting: {input_video.name}")
        print(f"   → {output_video.name}")
        print(f"   Angle: yaw={yaw}°, pitch={pitch}°, fov={fov}°")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ Conversion complete!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Conversion failed: {e}")
            print(f"   stderr: {e.stderr}")
            return False

    def scan_participant_videos(self, participant_id: str, date_folder: str = None) -> dict:
        """
        Scan participant folder for videos.

        Returns:
            Dictionary with camera types and their video files
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
            'X3': [],
            'X4': [],
            'X5': [],
            'iPhone': [],
            'GoPro': [],
            'Unknown': []
        }

        # Scan all camera folders
        for camera_folder in sorted(data_folder.iterdir()):
            if not camera_folder.is_dir():
                continue

            camera_type = None
            if 'X3' in camera_folder.name:
                camera_type = 'X3'
            elif 'X4' in camera_folder.name:
                camera_type = 'X4'
            elif 'X5' in camera_folder.name:
                camera_type = 'X5'
            elif 'iPhone' in camera_folder.name or 'iphone' in camera_folder.name:
                camera_type = 'iPhone'
            elif 'GoPro' in camera_folder.name or 'gopro' in camera_folder.name.lower():
                camera_type = 'GoPro'
            else:
                camera_type = 'Unknown'

            # Find video files
            for video_file in sorted(camera_folder.glob('*')):
                if video_file.suffix.lower() in ['.mp4', '.insv', '.mov']:
                    videos[camera_type].append(video_file)

        return videos

    def interactive_angle_selection(self, participant_id: str, date_folder: str = None):
        """
        Interactive mode: Select viewing angles for each camera.
        """
        print(f"\n{'='*60}")
        print(f"🎯 Interactive Angle Selection for {participant_id}")
        print(f"{'='*60}")

        videos = self.scan_participant_videos(participant_id, date_folder)
        angles = self.load_camera_angles()

        if participant_id not in angles:
            angles[participant_id] = {}

        # Process only 360° cameras
        for camera_type in ['X3', 'X4', 'X5']:
            if not videos[camera_type]:
                continue

            print(f"\n📹 {camera_type} Camera - {len(videos[camera_type])} videos found")

            # Use first video for angle selection
            sample_video = videos[camera_type][0]

            # Check if angle already saved
            camera_key = f"{participant_id}_{camera_type}"
            if camera_key in angles.get(participant_id, {}):
                print(f"   Existing angle: {angles[participant_id][camera_key]}")
                response = input("   Use existing angle? (y/n): ").strip().lower()
                if response == 'y':
                    continue

            # Preview and select angle
            print(f"\n   Opening preview for: {sample_video.name}")
            selected_angle = self.preview_360_video(sample_video)

            if selected_angle:
                angles[participant_id][camera_key] = selected_angle
                print(f"   ✅ Saved angle for {camera_type}: {selected_angle}")
            else:
                print(f"   ⏭️  Skipped {camera_type}")

        self.save_camera_angles(angles)
        print(f"\n✅ Angles saved to: {self.angles_file}")

    def batch_convert(self, participant_id: str, date_folder: str = None):
        """
        Batch mode: Convert all 360° videos using saved angles.
        """
        print(f"\n{'='*60}")
        print(f"🚀 Batch Conversion for {participant_id}")
        print(f"{'='*60}")

        videos = self.scan_participant_videos(participant_id, date_folder)
        angles = self.load_camera_angles()

        if participant_id not in angles:
            print(f"❌ No saved angles for {participant_id}. Run interactive mode first.")
            return

        # Prepare log
        log_entries = []

        # Process 360° cameras
        for camera_type in ['X3', 'X4', 'X5']:
            if not videos[camera_type]:
                continue

            camera_key = f"{participant_id}_{camera_type}"
            if camera_key not in angles[participant_id]:
                print(f"⚠️  No angle saved for {camera_type}, skipping...")
                continue

            angle = angles[participant_id][camera_key]
            print(f"\n📹 Processing {camera_type} - {len(videos[camera_type])} videos")

            for idx, video_file in enumerate(videos[camera_type], 1):
                # Extract chunk number from filename
                chunk_match = re.search(r'(\d{3,4})', video_file.stem)
                chunk_num = chunk_match.group(1) if chunk_match else f"{idx:03d}"

                # Output path
                output_dir = self.output_root / participant_id / date_folder / f"Cam{['X3', 'X4', 'X5'].index(camera_type) + 1}"
                output_file = output_dir / f"{participant_id}_Cam{['X3', 'X4', 'X5'].index(camera_type) + 1}_chunk{chunk_num}.mp4"

                # Convert
                success = self.convert_360_to_2d(
                    video_file, output_file,
                    yaw=angle['yaw'],
                    pitch=angle['pitch'],
                    roll=angle.get('roll', 0),
                    fov=angle.get('fov', self.default_settings['fov_h'])
                )

                # Log
                log_entries.append({
                    'timestamp': datetime.now().isoformat(),
                    'participant': participant_id,
                    'camera': camera_type,
                    'chunk': chunk_num,
                    'input': str(video_file),
                    'output': str(output_file),
                    'success': success
                })

        # Save log
        self._save_conversion_log(log_entries)
        print(f"\n✅ Batch conversion complete! Log: {self.log_file}")

    def _save_conversion_log(self, entries: list):
        """Save conversion log to CSV."""
        import csv

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.log_file.exists()

        with open(self.log_file, 'a', newline='') as f:
            fieldnames = ['timestamp', 'participant', 'camera', 'chunk', 'input', 'output', 'success']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerows(entries)


def main():
    parser = argparse.ArgumentParser(description='Convert 360° videos to 2D perspective for pose detection')
    parser.add_argument('--mode', choices=['interactive', 'batch', 'scan'], required=True,
                       help='Operation mode')
    parser.add_argument('--participant', required=True,
                       help='Participant ID (e.g., P03)')
    parser.add_argument('--date-folder', help='Date folder name (auto-detected if not provided)')
    parser.add_argument('--data-root', default='/Volumes/X10 Pro 1/Phan Dissertation Data',
                       help='Root path to participant data')
    parser.add_argument('--output-root', default='./Processed_Videos',
                       help='Root path for output processed videos')

    args = parser.parse_args()

    converter = VideoConverter360to2D(args.data_root, args.output_root)

    if args.mode == 'scan':
        videos = converter.scan_participant_videos(args.participant, args.date_folder)
        print(f"\n📊 Scan Results for {args.participant}:")
        for camera_type, video_list in videos.items():
            if video_list:
                print(f"\n{camera_type}: {len(video_list)} videos")
                for v in video_list[:3]:  # Show first 3
                    print(f"  - {v.name}")
                if len(video_list) > 3:
                    print(f"  ... and {len(video_list) - 3} more")

    elif args.mode == 'interactive':
        converter.interactive_angle_selection(args.participant, args.date_folder)

    elif args.mode == 'batch':
        converter.batch_convert(args.participant, args.date_folder)


if __name__ == '__main__':
    main()
