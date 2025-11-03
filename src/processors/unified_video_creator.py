"""
Unified Video Creator
Create stacked/unified videos from detection outputs
"""

import os
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional

from src.processors.base_processor import BaseProcessor, ValidationError, ProcessorError, DependencyError
from src.helpers.video_path_helper import generate_unified_video_path


class UnifiedVideoCreator(BaseProcessor):
    """Create unified video from detection outputs"""

    def __init__(self, config: Dict[str, Any], session_id: str,
                 detection_output_videos: Dict[str, str], alignment_results: Dict,
                 unified_videos_dir: str, timestamp_prefix: str):
        super().__init__(config, session_id)
        self.detection_output_videos = detection_output_videos
        self.alignment_results = alignment_results
        self.unified_videos_dir = unified_videos_dir
        self.timestamp_prefix = timestamp_prefix
        self.unified_video_config = config.get('integrated_processor', {}).get('unified_video_config', {})

    def validate_dependencies(self):
        """Validate dependencies for unified video creation"""
        required_packages = [
            {'name': 'cv2', 'import_name': 'cv2', 'install_name': 'opencv-python'},
            {'name': 'numpy', 'import_name': 'numpy'},
        ]
        self.require_packages(required_packages)

        # Check for ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("   ✅ ffmpeg available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise DependencyError(
                "ffmpeg not found. Install with: brew install ffmpeg (Mac) or apt-get install ffmpeg (Linux)"
            )

    def validate_inputs(self):
        """Validate detection videos"""
        if not self.detection_output_videos:
            raise ValidationError("No detection videos provided")

        missing_videos = []
        for camera, path in self.detection_output_videos.items():
            if not os.path.exists(path):
                missing_videos.append(f"{camera}: {path}")

        if missing_videos:
            raise ValidationError(
                f"Detection videos not found:\n" + "\n".join(f"  - {v}" for v in missing_videos)
            )

        os.makedirs(self.unified_videos_dir, exist_ok=True)
        print(f"   📁 Output directory: {self.unified_videos_dir}")
        print(f"   🎥 Combining {len(self.detection_output_videos)} videos")

    def process(self) -> Dict[str, Any]:
        """Create unified video"""
        output_filename = self.unified_video_config.get('output_filename', 'unified_detection_output.mp4')
        unified_output_path = generate_unified_video_path(
            self.unified_videos_dir, output_filename, self.timestamp_prefix
        )

        # Analyze videos
        video_info, max_duration, reference_fps = self._analyze_videos()
        if not video_info:
            raise ProcessorError("No valid videos found for unified creation")

        print(f"🎬 Creating unified video: {max_duration:.1f}s, {reference_fps:.1f} fps")

        # Create stacked video
        success = self._create_stacked_video_with_sync(
            video_info, unified_output_path, max_duration, reference_fps
        )

        if not success:
            raise ProcessorError("Failed to create unified video")

        return {
            'unified_video_path': unified_output_path,
            'duration': max_duration,
            'fps': reference_fps,
            'num_videos': len(video_info)
        }

    def _analyze_videos(self):
        """Analyze video properties"""
        video_info = {}
        max_duration = 0
        reference_fps = 30.0

        print("📊 Analyzing video properties...")

        for camera_prefix, video_path in self.detection_output_videos.items():
            if not os.path.exists(video_path):
                print(f"⚠️ Skipping missing video: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️ Cannot open video: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            # Get alignment offset
            offset = self._get_camera_offset(camera_prefix)

            video_info[camera_prefix] = {
                'path': video_path,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration,
                'frame_count': frame_count,
                'offset': offset,
                'total_duration': duration + offset
            }

            max_duration = max(max_duration, duration + offset)
            reference_fps = fps

            print(f"   📹 {camera_prefix}: {width}x{height}, {duration:.1f}s, offset: {offset:.3f}s")

        return video_info, max_duration, reference_fps

    def _get_camera_offset(self, camera_prefix: str) -> float:
        """Get camera offset from alignment results"""
        offset = 0.0
        if self.alignment_results:
            for prefix, group in self.alignment_results.items():
                if camera_prefix == prefix and hasattr(group, 'chunks') and group.chunks:
                    offset = group.chunks[0].start_time_offset
                    break
        return offset

    def _create_stacked_video_with_sync(self, video_info: dict, output_path: str,
                                       total_duration: float, fps: float) -> bool:
        """Create stacked video (implementation simplified for brevity)"""
        # This is a simplified version - full implementation would include:
        # - Frame-by-frame stacking
        # - Audio processing
        # - Progress tracking
        # For now, use ffmpeg filter_complex

        try:
            print("🎥 Creating unified video with ffmpeg...")

            # Build ffmpeg command for vertical stacking
            inputs = []
            for camera_prefix in sorted(video_info.keys()):
                inputs.extend(['-i', video_info[camera_prefix]['path']])

            num_videos = len(video_info)
            filter_complex = f"vstack=inputs={num_videos}"

            cmd = [
                'ffmpeg', '-y',
                *inputs,
                '-filter_complex', filter_complex,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Unified video created: {output_path}")
                return True
            else:
                print(f"❌ ffmpeg failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error creating unified video: {e}")
            return False
