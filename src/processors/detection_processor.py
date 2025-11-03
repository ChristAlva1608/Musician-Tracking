"""
Detection Processor
Runs pose, hand, face, and transcript detection on aligned videos
"""

import os
import sys
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from src.processors.base_processor import BaseProcessor, DependencyError, ValidationError, ProcessorError
from src.helpers.video_path_helper import (
    extract_source_name,
    generate_detection_video_path,
    generate_temp_detection_video_path
)


class DetectionProcessor(BaseProcessor):
    """
    Processor for running detection on aligned videos
    Handles pose, hand, face, and transcript detection
    """

    def __init__(self, config: Dict[str, Any], session_id: str, output_videos: Dict[str, str],
                 alignment_results: Dict = None, alignment_directory: str = None):
        """
        Initialize detection processor

        Args:
            config: Full configuration dictionary
            session_id: Unique session identifier
            output_videos: Dictionary of {camera_prefix: video_path} to process
            alignment_results: Optional alignment data for offset calculation
            alignment_directory: Source alignment directory for naming
        """
        super().__init__(config, session_id)
        self.output_videos = output_videos
        self.alignment_results = alignment_results or {}
        self.alignment_directory = alignment_directory or ""

        # Extract configuration
        self.integrated_config = config.get('integrated_processor', {})
        self.detection_videos_dir = self.integrated_config.get(
            'detection_videos_dir', 'src/output/annotated_detection_videos'
        )
        self.processing_type = self.integrated_config.get('processing_type', 'use_offset')
        self.limit_processing_duration = self.integrated_config.get('limit_processing_duration', False)
        self.max_processing_duration = self.integrated_config.get('max_processing_duration', 60.0)
        self.timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Results tracking
        self.detection_output_videos = {}

    def validate_dependencies(self):
        """Validate detection dependencies"""
        required_packages = [
            {'name': 'cv2', 'import_name': 'cv2', 'install_name': 'opencv-python'},
            {'name': 'numpy', 'import_name': 'numpy'},
            {'name': 'mediapipe', 'import_name': 'mediapipe'},
            {'name': 'ultralytics', 'import_name': 'ultralytics'},
        ]

        self.require_packages(required_packages)

        # Check optional packages
        optional_packages = [
            {'name': 'whisper', 'message': 'Transcript features will be disabled'},
        ]

        for pkg in optional_packages:
            if not self.check_package(pkg['name']):
                print(f"   ⚠️  {pkg['name']} not available - {pkg['message']}")

    def validate_inputs(self):
        """Validate input videos and configuration"""
        if not self.output_videos:
            raise ValidationError("No output videos provided for detection")

        # Check that all video files exist
        missing_videos = []
        for camera_prefix, video_path in self.output_videos.items():
            if not os.path.exists(video_path):
                missing_videos.append(f"{camera_prefix}: {video_path}")

        if missing_videos:
            raise ValidationError(
                f"Video files not found:\n" + "\n".join(f"  - {v}" for v in missing_videos)
            )

        # Create output directory
        os.makedirs(self.detection_videos_dir, exist_ok=True)
        print(f"   📁 Output directory: {self.detection_videos_dir}")
        print(f"   🎬 Processing {len(self.output_videos)} videos")

    def _apply_config_overrides(self, target_config: dict, overrides: dict) -> dict:
        """Apply configuration overrides with deep merge"""
        for key, value in overrides.items():
            if key in target_config and isinstance(target_config[key], dict) and isinstance(value, dict):
                target_config[key] = self._apply_config_overrides(target_config[key], value)
            else:
                target_config[key] = value
        return target_config

    def process(self) -> Dict[str, Any]:
        """Run detection on all aligned videos"""
        from src.detect_v2_3d import DetectorV2

        success_count = 0
        total_videos = len(self.output_videos)

        for camera_prefix, video_path in self.output_videos.items():
            print(f"\n{'='*70}")
            print(f"Processing: {camera_prefix}")
            print(f"Video: {video_path}")
            print(f"{'='*70}")

            with self._timer(f"Detection for {camera_prefix}"):
                try:
                    # Initialize detector
                    print(f"🔧 Initializing detector...")
                    try:
                        detector = DetectorV2(config_path='src/config/config_v1.yaml')
                    except ImportError as ie:
                        raise DependencyError(
                            f"Missing modules for detection: {ie}\n"
                            "Install with: pip install mediapipe ultralytics torch opencv-python"
                        ) from ie
                    except Exception as e:
                        raise ProcessorError(f"Detector initialization failed: {e}") from e

                    # Apply configuration overrides
                    detector_overrides = self.integrated_config.get('detector_config', {})
                    if detector_overrides:
                        print(f"🔧 Applying detector config overrides...")
                        for section, overrides in detector_overrides.items():
                            print(f"   📝 Overriding {section}: {overrides}")
                        detector.config = self._apply_config_overrides(detector.config, detector_overrides)

                    # Set video path
                    detector.config['video']['source_path'] = video_path
                    detector.processing_type = self.processing_type

                    # Re-initialize database if enabled
                    if detector.config.get('database', {}).get('enabled', False) and not detector.db:
                        detector._init_database()

                    # Get camera offset for synced_time calculation
                    camera_offset = self._get_camera_offset(camera_prefix)

                    # Configure detection mode
                    self._configure_detection_mode(detector, camera_offset)

                    # Configure output paths
                    detection_output_path, temp_detection_path = self._generate_output_paths(camera_prefix)
                    detector.config['video']['output_video_path'] = detection_output_path
                    detector.config['video']['temp_video_path'] = temp_detection_path

                    # Update session ID
                    detector.session_id = f"{self.session_id}_{camera_prefix}"

                    # Print configuration
                    self._print_detection_config(detector, video_path, detection_output_path, camera_offset)

                    # Run detection
                    print(f"🎬 Starting video processing...")
                    try:
                        detector.process_video(video_path)
                    except ImportError as ie:
                        error_msg = self._handle_import_error(ie)
                        raise DependencyError(error_msg) from ie
                    except AttributeError as ae:
                        if 'face' in str(ae).lower():
                            raise ProcessorError(
                                f"Face detection failed: {ae}\n"
                                "This may indicate no faces were detected or face model not properly configured."
                            ) from ae
                        raise
                    except RuntimeError as re:
                        if 'database' in str(re).lower():
                            raise ProcessorError(
                                f"Database operation failed: {re}\n"
                                "Check database connection and credentials."
                            ) from re
                        raise

                    # Cleanup
                    detector.cleanup()

                    print(f"✅ Detection complete for {camera_prefix}")
                    success_count += 1
                    self.detection_output_videos[camera_prefix] = detection_output_path

                except (DependencyError, ProcessorError):
                    # Re-raise processor errors to stop execution
                    raise
                except Exception as e:
                    raise ProcessorError(f"Unexpected error during detection for {camera_prefix}: {e}") from e

        print(f"\n{'='*70}")
        print(f"✅ Detection complete: {success_count}/{total_videos} videos processed")
        print(f"{'='*70}\n")

        return {
            'success_count': success_count,
            'total_videos': total_videos,
            'detection_output_videos': self.detection_output_videos
        }

    def _get_camera_offset(self, camera_prefix: str) -> float:
        """Get camera offset from alignment results"""
        camera_offset = 0.0
        if self.alignment_results:
            for prefix, group in self.alignment_results.items():
                if camera_prefix == prefix and hasattr(group, 'chunks') and group.chunks:
                    camera_offset = group.chunks[0].start_time_offset
                    print(f"   🎯 Camera offset: {camera_offset:.3f}s")
                    break
        return camera_offset

    def _configure_detection_mode(self, detector, camera_offset: float):
        """Configure detection mode based on processing_type"""
        if self.processing_type == "full_frames":
            # Process full frames with alignment awareness
            detector.start_time_offset = camera_offset
            detector.config['video']['process_matching_duration_only'] = False
            detector.process_matching_duration_only = False

            if self.limit_processing_duration:
                detector.matching_duration = self.max_processing_duration
                detector.end_time_offset = self.max_processing_duration
                print(f"   🎬 Mode: Full frames (0s to {detector.end_time_offset:.3f}s)")
                print(f"   📊 Offset {camera_offset:.3f}s used for synced_time only")
            else:
                detector.matching_duration = 0.0
                detector.end_time_offset = None
                print(f"   🎬 Mode: Full frames (no duration limit)")
                print(f"   📊 Offset: {camera_offset:.3f}s")

        elif self.limit_processing_duration:
            # Duration limiting with offset
            detector.config['video']['process_matching_duration_only'] = True
            use_offset = self.processing_type == "use_offset"
            detector.start_time_offset = camera_offset if use_offset else 0.0
            detector.matching_duration = self.max_processing_duration
            detector.end_time_offset = detector.start_time_offset + self.max_processing_duration
            detector.process_matching_duration_only = True

            if use_offset:
                print(f"   🎬 Mode: use_offset ({camera_offset:.3f}s to {detector.end_time_offset:.3f}s)")
            else:
                print(f"   🎬 Mode: Duration limit (first {self.max_processing_duration}s)")

        else:
            # Standard processing
            detector.config['video']['process_matching_duration_only'] = True
            detector.start_time_offset = camera_offset if self.processing_type == "use_offset" else 0.0
            print(f"   🎬 Mode: {self.processing_type} (start: {detector.start_time_offset:.3f}s)")

    def _generate_output_paths(self, camera_prefix: str) -> tuple:
        """Generate output paths for detection videos"""
        source_name = extract_source_name(self.alignment_directory)
        output_format = self.integrated_config.get('detection_output_format', 'mp4')

        detection_output_path = generate_detection_video_path(
            self.detection_videos_dir, source_name, camera_prefix,
            self.timestamp_prefix, output_format
        )
        temp_detection_path = generate_temp_detection_video_path(
            self.detection_videos_dir, source_name, camera_prefix,
            self.timestamp_prefix, output_format
        )

        return detection_output_path, temp_detection_path

    def _print_detection_config(self, detector, video_path: str, output_path: str, offset: float):
        """Print detection configuration"""
        print(f"\n   📋 Detection Configuration:")
        print(f"      📹 Input: {video_path}")
        print(f"      🎬 Output: {output_path}")
        print(f"      💾 Database: {detector.config['database']['enabled']}")
        print(f"      📊 Session: {detector.session_id}")
        print(f"      ⏱️  Offset: {offset:.3f}s")
        print(f"      🎚️  Save video: {detector.config['video']['save_output_video']}")
        print(f"      🔊 Audio: {detector.config['video']['preserve_audio']}")
        print(f"      📄 Report: {detector.config['video']['generate_report']}\n")

    def _handle_import_error(self, ie: ImportError) -> str:
        """Generate helpful error message for import errors"""
        error_str = str(ie).lower()

        if 'whisper' in error_str:
            return (
                f"Transcript model (whisper) not available: {ie}\n"
                "Install with: pip install openai-whisper"
            )
        elif 'mediapipe' in error_str:
            return (
                f"MediaPipe not available: {ie}\n"
                "Install with: pip install mediapipe"
            )
        else:
            return f"Missing detection module: {ie}\nPlease install required packages"
