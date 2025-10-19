#!/usr/bin/env python3
"""
Integrated Video Processor
Combines video alignment and detection processing in a single workflow

This module:
1. Retrieves alignment offsets from database for videos in alignment_directory
2. Falls back to analyzing videos if no alignment data exists (using shape_based_aligner_multi.py logic)
3. Creates aligned output videos based on the alignment data
4. Runs detection processing on the aligned videos (using detect_v2.py logic)
"""

import os
import sys
import yaml
import time
import glob
import shutil
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import contextmanager

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import video alignment components
from src.video_aligner.shape_based_aligner_multi import (
    scan_and_group_chunk_videos,
    determine_reference_camera_group_by_audio_pattern,
    align_chunks_to_reference_timeline,
    generate_output_video_paths,
    combine_chunk_videos_timeline_based
)

# Import detection components
from src.detect_v2_3d import DetectorV2

# Import database components
from src.database.setup import VideoAlignmentDatabase, ChunkVideoAlignmentDatabase

# Import helper utilities
from src.helpers.video_path_helper import (
    extract_source_name,
    generate_aligned_video_path,
    generate_detection_video_path,
    generate_temp_detection_video_path,
    generate_unified_video_path,
    generate_output_paths_from_alignment_data,
    generate_output_paths_from_legacy_data
)


class IntegratedVideoProcessor:
    """
    Integrated processor that handles video alignment and detection in sequence
    """

    @contextmanager
    def _timer(self, step_name: str):
        """
        Context manager for timing operations

        Args:
            step_name: Name of the step being timed
        """
        start_time = time.time()
        print(f"⏱️  [{step_name}] Starting...")
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            if minutes > 0:
                print(f"⏱️  [{step_name}] Completed in {minutes}m {seconds:.2f}s")
            else:
                print(f"⏱️  [{step_name}] Completed in {seconds:.2f}s")

    def __init__(self, config_path: str = 'src/config/config_v1.yaml'):
        """
        Initialize the integrated processor

        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.session_id = f"integrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Video aligner settings
        self.video_aligner_config = self.config.get('video_aligner', {})
        self.alignment_directory = self.video_aligner_config.get('alignment_directory', '')
        self.save_output_videos = self.video_aligner_config.get('save_output_videos', True)
        self.enable_chunk_processing = self.video_aligner_config.get('alignment', {}).get('enable_chunk_processing', True)

        # Integrated processor settings
        self.integrated_config = self.config.get('integrated_processor', {})
        self.run_detection = self.integrated_config.get('run_detection', True)
        self.cleanup_temp_files = self.integrated_config.get('cleanup_temp_files', True)

        # Output directory settings
        self.aligned_videos_dir = self.integrated_config.get('aligned_videos_dir', 'src/output/aligned_videos')
        self.detection_videos_dir = self.integrated_config.get('detection_videos_dir', 'src/output/annotated_detection_videos')
        self.unified_videos_dir = self.integrated_config.get('unified_videos_dir', 'src/output/unified_videos')

        # Duration limiting settings
        self.limit_processing_duration = self.integrated_config.get('limit_processing_duration', False)
        self.max_processing_duration = self.integrated_config.get('max_processing_duration', 60.0)

        # Processing settings
        self.processing_type = self.integrated_config.get('processing_type', 'use_offset')
        self.unified_videos = self.integrated_config.get('unified_videos', False)
        self.unified_video_config = self.integrated_config.get('unified_video_config', {})

        # Initialize databases
        self.alignment_db = None
        self.chunk_alignment_db = None
        try:
            self.alignment_db = VideoAlignmentDatabase()
            self.chunk_alignment_db = ChunkVideoAlignmentDatabase()
            print("✅ Connected to alignment databases")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")

        # Store alignment results
        self.alignment_results = {}
        self.output_videos = {}

        print(f"🚀 IntegratedVideoProcessor initialized")
        print(f"📁 Alignment directory: {self.alignment_directory}")
        print(f"🎥 Save output videos: {self.save_output_videos}")
        print(f"🔍 Run detection: {self.run_detection}")
        print(f"📦 Chunk processing: {self.enable_chunk_processing}")
        if self.limit_processing_duration:
            print(f"⏱️ Duration limiting: ENABLED ({self.max_processing_duration}s per video)")
        else:
            print(f"⏱️ Duration limiting: DISABLED (full video processing)")

        print(f"🎬 Processing type: {self.processing_type.upper()}")

        if self.unified_videos:
            stack_direction = self.unified_video_config.get('stack_direction', 'vertical')
            print(f"📹 Unified videos: ENABLED ({stack_direction} stacking)")
        else:
            print(f"📹 Unified videos: DISABLED (separate detection videos)")

    def _apply_config_overrides(self, target_config: dict, overrides: dict) -> dict:
        """
        Apply configuration overrides to target config with deep merge

        Args:
            target_config: The original configuration dictionary
            overrides: The override configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        for key, value in overrides.items():
            if key in target_config and isinstance(target_config[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                target_config[key] = self._apply_config_overrides(target_config[key], value)
            else:
                # Override or add new key
                target_config[key] = value
        return target_config

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {}

    def check_existing_alignment_data(self) -> bool:
        """
        Check if alignment data already exists in database for videos in alignment_directory

        Returns:
            True if alignment data exists, False otherwise
        """
        with self._timer("Check Existing Alignment Data"):
            if not self.alignment_directory or not os.path.exists(self.alignment_directory):
                print(f"❌ Alignment directory not found: {self.alignment_directory}")
                return False

            video_files = glob.glob(os.path.join(self.alignment_directory, "*.mp4"))
            if not video_files:
                print(f"❌ No MP4 files found in {self.alignment_directory}")
                return False

            print(f"🔍 Checking existing alignment data for {len(video_files)} videos...")

            # Extract source name from directory path
            source_name = extract_source_name(self.alignment_directory)

            print(f"🔍 Looking for alignment data with source: {source_name}")

            try:
                if self.enable_chunk_processing and self.chunk_alignment_db:
                    # Check chunk alignment database
                    result = self.chunk_alignment_db.get_chunk_alignments_by_source(source_name)
                    if result and len(result) > 0:
                        print(f"✅ Found {len(result)} chunk alignment records in database")
                        self._load_chunk_alignment_data(result)
                        return True
                else:
                    # Check legacy alignment database
                    if self.alignment_db:
                        all_alignments = self.alignment_db.get_all_video_alignments()
                        source_alignments = [a for a in all_alignments if a.get('source', '').startswith(source_name)]
                        if source_alignments:
                            print(f"✅ Found {len(source_alignments)} alignment records in database")
                            self._load_legacy_alignment_data(source_alignments)
                            return True

                print(f"⚠️ No alignment data found in database for source: {source_name}")
                return False

            except Exception as e:
                print(f"❌ Error checking alignment data: {e}")
                return False

    def _save_chunk_alignment_data(self, camera_groups: dict, reference_prefix: str, method_type: str):
        """Save chunk alignment data to database"""
        if not self.chunk_alignment_db:
            print("⚠️ No chunk alignment database connection available")
            return

        # Extract source name from alignment directory
        source_name = extract_source_name(self.alignment_directory)
        print(f"💾 Saving alignment data to database for source: {source_name}")

        success_count = 0
        total_chunks = 0

        for prefix, group in camera_groups.items():
            for chunk in group.chunks:
                total_chunks += 1
                print(f"   💾 Saving: {chunk.filename} -> camera: {prefix}, offset: {chunk.start_time_offset:.3f}s")

                try:
                    # Get reference information
                    reference_group = camera_groups[reference_prefix]
                    reference_chunk = reference_group.chunks[0] if reference_group.chunks else None

                    if reference_chunk:
                        success = self.chunk_alignment_db.insert_chunk_alignment(
                            source=source_name,
                            chunk_filename=chunk.filename,
                            camera_prefix=prefix,
                            chunk_order=chunk.chunk_number,
                            start_time_offset=chunk.start_time_offset,
                            chunk_duration=chunk.duration,
                            reference_camera_prefix=reference_prefix,
                            reference_chunk_filename=reference_chunk.filename,
                            session_id=self.session_id,
                            method_type=method_type
                        )

                        if success:
                            success_count += 1
                            print(f"   ✅ Saved chunk alignment: {chunk.filename} (offset: {chunk.start_time_offset:.3f}s)")
                        else:
                            print(f"   ❌ Failed to save chunk alignment: {chunk.filename}")
                    else:
                        print(f"   ⚠️ No reference chunk found for {chunk.filename}")

                except Exception as e:
                    print(f"   ❌ Database error for {chunk.filename}: {e}")

        print(f"💾 Database save complete: {success_count}/{total_chunks} chunks saved successfully")

    def _load_chunk_alignment_data(self, chunk_data: List[Dict]):
        """Load chunk alignment data from database"""
        print(f"📊 Loading chunk alignment data...")

        # Import ChunkVideo and CameraGroup classes
        from src.video_aligner.shape_based_aligner_multi import ChunkVideo, CameraGroup

        # Group chunks by camera prefix
        camera_groups_dict = {}
        for chunk in chunk_data:
            camera_prefix = chunk['camera_prefix']
            if camera_prefix not in camera_groups_dict:
                camera_groups_dict[camera_prefix] = []
            camera_groups_dict[camera_prefix].append(chunk)

        # Sort chunks within each camera group by chunk_order
        for prefix in camera_groups_dict:
            camera_groups_dict[prefix].sort(key=lambda x: x['chunk_order'])

        # Convert to CameraGroup objects
        camera_groups = {}
        for prefix, chunks_list in camera_groups_dict.items():
            # Create ChunkVideo objects
            chunk_videos = []
            for chunk_dict in chunks_list:
                # Find the actual video file in alignment directory
                chunk_filename = chunk_dict['chunk_filename']
                chunk_filepath = os.path.join(self.alignment_directory, chunk_filename)

                chunk_video = ChunkVideo(
                    filename=chunk_filename,
                    filepath=chunk_filepath,
                    prefix=prefix,
                    chunk_number=chunk_dict['chunk_order'],
                    duration=float(chunk_dict['chunk_duration']),
                    start_time_offset=float(chunk_dict['start_time_offset'])
                )
                chunk_videos.append(chunk_video)

            # Calculate total duration and earliest start
            total_duration = sum(cv.duration for cv in chunk_videos)
            earliest_start = min(cv.start_time_offset for cv in chunk_videos) if chunk_videos else 0.0

            # Create CameraGroup object
            camera_group = CameraGroup(
                prefix=prefix,
                chunks=chunk_videos,
                total_duration=total_duration,
                earliest_start=earliest_start
            )
            camera_groups[prefix] = camera_group

        self.alignment_results = camera_groups
        print(f"✅ Loaded alignment data for {len(camera_groups)} cameras")

        # Generate output video paths based on existing data
        self._generate_output_paths_from_alignment_data()

    def _load_legacy_alignment_data(self, alignment_data: List[Dict]):
        """Load legacy alignment data from database"""
        print(f"📊 Loading legacy alignment data...")

        self.alignment_results = {}
        for data in alignment_data:
            video_name = data.get('source', 'unknown')
            self.alignment_results[video_name] = {
                'start_time_offset': data.get('start_time_offset', 0.0),
                'matching_duration': data.get('matching_duration', 0.0),
                'camera_type': data.get('camera_type', 1)
            }

        print(f"✅ Loaded alignment data for {len(self.alignment_results)} videos")
        self._generate_output_paths_from_legacy_data()

    def _generate_output_paths_from_alignment_data(self):
        """Generate output video paths from alignment data"""
        print(f"📁 Generating output paths in: {self.aligned_videos_dir}")

        self.output_videos = generate_output_paths_from_alignment_data(
            self.alignment_results,
            self.alignment_directory,
            self.aligned_videos_dir,
            self.timestamp_prefix
        )

        for camera_prefix, output_path in self.output_videos.items():
            print(f"  {camera_prefix} -> {output_path}")

    def _generate_output_paths_from_legacy_data(self):
        """Generate output paths from legacy alignment data"""
        print(f"📁 Generating output paths in: {self.aligned_videos_dir}")

        self.output_videos = generate_output_paths_from_legacy_data(
            self.alignment_results,
            self.alignment_directory,
            self.aligned_videos_dir,
            self.timestamp_prefix
        )

        for camera_prefix, output_path in self.output_videos.items():
            print(f"  {camera_prefix} -> {output_path}")

    def analyze_video_alignment(self) -> bool:
        """
        Analyze video alignment using shape_based_aligner_multi.py logic

        Returns:
            True if analysis successful, False otherwise
        """
        with self._timer("Analyze Video Alignment"):
            print(f"\n{'='*60}")
            print("ANALYZING VIDEO ALIGNMENT")
            print(f"{'='*60}")

            if not os.path.exists(self.alignment_directory):
                print(f"❌ Alignment directory not found: {self.alignment_directory}")
                return False

            try:
                # Use the same logic as shape_based_aligner_multi.py main function
                if self.enable_chunk_processing:
                    return self._analyze_chunk_alignment()
                else:
                    return self._analyze_legacy_alignment()

            except Exception as e:
                print(f"❌ Error during alignment analysis: {e}")
                import traceback
                traceback.print_exc()
                return False

    def _analyze_chunk_alignment(self) -> bool:
        """Analyze alignment using chunk processing logic"""
        print("🔍 Using chunk processing alignment logic...")

        # Step 1: Scan and group videos
        camera_groups = scan_and_group_chunk_videos(self.alignment_directory)
        if not camera_groups:
            print("❌ No camera groups found")
            return False

        # Step 2: Determine reference camera group
        # Auto-pair processing_type with reference_strategy:
        #
        # Strategy Selection Logic:
        # 1. unified_videos=true  → always earliest_start (need full timeline for sync)
        # 2. processing_type=full_frames → earliest_start (process all content)
        # 3. processing_type=use_offset → latest_start (only synchronized portion)
        #
        # Note: reference_strategy from config is ignored - determined automatically

        if self.unified_videos:
            # Unified videos always need earliest_start for complete timeline
            use_earliest_start = True
            method_type = 'earliest_start'
            reason = "unified_videos requires full timeline"
        elif self.processing_type == "full_frames":
            # Full frames processing needs all content from earliest camera
            use_earliest_start = True
            method_type = 'earliest_start'
            reason = "full_frames processes all content"
        else:  # use_offset
            # Use offset only processes synchronized portion
            use_earliest_start = False
            method_type = 'latest_start'
            reason = "use_offset processes only synchronized portion"

        print(f"🎯 Auto-selected alignment: {method_type} ({reason})")
        print(f"   processing_type={self.processing_type}, unified_videos={self.unified_videos}")
        reference_prefix = determine_reference_camera_group_by_audio_pattern(camera_groups, use_earliest_start)

        # Step 3: Align chunks
        camera_groups = align_chunks_to_reference_timeline(camera_groups, reference_prefix, use_earliest_start)

        # Step 4: Generate output paths
        camera_groups = generate_output_video_paths(camera_groups, self.config)

        # Store results
        self.alignment_results = camera_groups

        # Extract output video paths
        self.output_videos = {}
        for prefix, group in camera_groups.items():
            if hasattr(group, 'output_video_path'):
                self.output_videos[prefix] = group.output_video_path

        print(f"✅ Alignment analysis complete for {len(camera_groups)} camera groups")

        # Step 5: Save alignment data to database
        self._save_chunk_alignment_data(camera_groups, reference_prefix, method_type)

        return True

    def _analyze_legacy_alignment(self) -> bool:
        """Analyze alignment using legacy logic"""
        print("🔍 Using legacy alignment logic...")

        # Import and use the legacy alignment logic from shape_based_aligner_multi.py
        # This would require extracting the legacy video processing workflow
        # For now, use a simplified approach

        video_files = glob.glob(os.path.join(self.alignment_directory, "*.mp4"))
        if not video_files:
            print("❌ No video files found")
            return False

        # Create simple alignment results (placeholder for now)
        self.alignment_results = {}
        self.output_videos = {}

        # Use config-based directory
        os.makedirs(self.aligned_videos_dir, exist_ok=True)

        # Extract source name from alignment directory
        source_name = extract_source_name(self.alignment_directory)

        for i, video_file in enumerate(video_files):
            camera_prefix = f"cam_{i+1}"

            self.alignment_results[camera_prefix] = {
                'start_time_offset': 0.0,  # Placeholder
                'matching_duration': 0.0,  # Placeholder
                'source_file': video_file
            }

            output_path = generate_aligned_video_path(
                self.aligned_videos_dir, source_name, camera_prefix, self.timestamp_prefix
            )
            self.output_videos[camera_prefix] = output_path

        print(f"✅ Legacy alignment setup complete for {len(video_files)} videos")
        return True

    def create_aligned_videos(self) -> bool:
        """
        Create aligned output videos based on alignment results
        - For chunk videos: merge chunks into aligned_videos folder
        - For regular videos: map original paths directly (no copying)

        Returns:
            True if video creation successful, False otherwise
        """
        if not self.alignment_results:
            print("❌ No alignment results available for video creation")
            return False

        print(f"\n{'='*60}")
        print("PREPARING VIDEOS FOR DETECTION")
        print(f"{'='*60}")

        try:
            # Check if we're dealing with actual chunk videos
            has_chunks = self._has_chunk_videos()

            if has_chunks:
                print("📦 Detected chunk videos - merging into aligned videos...")
                return self._create_chunk_aligned_videos()
            else:
                print("📁 Detected regular videos - using original paths directly (no copying)...")
                return self._map_original_video_paths()

        except Exception as e:
            print(f"❌ Error preparing videos: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _has_chunk_videos(self) -> bool:
        """
        Check if alignment results contain chunk videos (multiple chunks per camera)

        Returns:
            True if chunk videos detected, False for regular videos
        """
        if not self.alignment_results:
            return False

        # Check if alignment_results contains CameraGroup objects with chunks
        for prefix, group in self.alignment_results.items():
            if hasattr(group, 'chunks'):
                # If there are multiple chunks for any camera, it's chunk processing
                if len(group.chunks) > 1:
                    print(f"   🔍 Camera {prefix} has {len(group.chunks)} chunks - chunk processing needed")
                    return True

        return False

    def _map_original_video_paths(self) -> bool:
        """
        Map original video paths for detection without copying files

        Returns:
            True if mapping successful, False otherwise
        """
        print("🗺️ Mapping original video paths for detection...")

        self.output_videos = {}

        # Get video files from alignment directory
        video_files = glob.glob(os.path.join(self.alignment_directory, "*.mp4"))

        if not video_files:
            print("❌ No video files found in alignment directory")
            return False

        # Map each video file to its original path
        for video_file in sorted(video_files):
            # Extract camera identifier from filename (e.g., cam_1.mp4 -> cam_1)
            filename = os.path.basename(video_file)
            camera_prefix = os.path.splitext(filename)[0]  # Remove .mp4

            # Store original path directly
            self.output_videos[camera_prefix] = video_file
            print(f"   ✅ {camera_prefix} -> {video_file} (original)")

        print(f"✅ Mapped {len(self.output_videos)} original video paths for detection")
        return len(self.output_videos) > 0

    def _create_chunk_aligned_videos(self) -> bool:
        """Create aligned videos using chunk processing logic"""
        print("🎬 Creating aligned videos using chunk processing...")

        # Use the combine_chunk_videos_timeline_based function
        combined_videos = combine_chunk_videos_timeline_based(self.alignment_results)

        success_count = 0
        for prefix, output_path in combined_videos.items():
            if os.path.exists(output_path):
                print(f"✅ Created: {output_path}")
                success_count += 1
            else:
                print(f"❌ Failed to create: {output_path}")

        print(f"✅ Successfully created {success_count}/{len(combined_videos)} aligned videos")
        return success_count > 0

    def _create_legacy_aligned_videos(self) -> bool:
        """Create aligned videos using legacy logic"""
        print("🎬 Creating aligned videos using legacy processing...")

        success_count = 0
        for camera_prefix, data in self.alignment_results.items():
            if 'source_file' not in data:
                continue

            source_file = data['source_file']
            output_path = self.output_videos.get(camera_prefix)

            if not output_path:
                continue

            try:
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Simple copy for now (could be enhanced with ffmpeg processing)
                shutil.copy2(source_file, output_path)
                print(f"✅ Copied: {source_file} -> {output_path}")
                success_count += 1

            except Exception as e:
                print(f"❌ Failed to copy {source_file}: {e}")

        print(f"✅ Successfully created {success_count}/{len(self.alignment_results)} aligned videos")
        return success_count > 0

    def run_detection_on_aligned_videos(self) -> bool:
        """
        Run detection processing on the aligned videos

        Returns:
            True if detection successful, False otherwise
        """
        with self._timer("Run Detection on All Videos"):
            if not self.run_detection:
                print("⏭️ Skipping detection processing (run_detection disabled)")
                return True

            if not self.output_videos:
                print("❌ No output videos available for detection")
                return False

            print(f"\n{'='*60}")
            print("RUNNING DETECTION ON ALIGNED VIDEOS")
            print(f"{'='*60}")

            success_count = 0
            total_videos = len(self.output_videos)
            self.detection_output_videos = {}  # Track detection output videos for unified processing

            for camera_prefix, video_path in self.output_videos.items():
                if not os.path.exists(video_path):
                    print(f"❌ Video not found: {video_path}")
                    continue

                print(f"\n🎯 Processing detection for {camera_prefix}: {video_path}")

                with self._timer(f"Detection for {camera_prefix}"):
                    try:
                        # Create a new detector instance for this video
                        detector = DetectorV2(config_path='src/config/config_v1.yaml')

                        # Apply configuration overrides from integrated_processor.detector_config
                        detector_overrides = self.integrated_config.get('detector_config', {})
                        if detector_overrides:
                            print(f"🔧 Applying detector config overrides from integrated_processor.detector_config")
                            for section, overrides in detector_overrides.items():
                                print(f"   📝 Overriding {section}: {overrides}")
                            detector.config = self._apply_config_overrides(detector.config, detector_overrides)

                        # Set the specific video path for this detector instance
                        detector.config['video']['source_path'] = video_path

                        # Set processing_type for synced_time calculation
                        detector.processing_type = self.processing_type
                        print(f"🎯 Setting detector processing_type: {self.processing_type}")

                        # Re-initialize database if database config was overridden and enabled
                        if detector.config.get('database', {}).get('enabled', False) and not detector.db:
                            detector._init_database()

                        # Get alignment data for this camera
                        # Offset is needed for BOTH processing modes:
                        # - full_frames: used for synced_time calculation (synced_time = original_time - offset)
                        # - use_offset: used for seeking and synced_time calculation
                        camera_offset = 0.0
                        use_offset = self.processing_type == "use_offset"
                        if hasattr(self, 'alignment_results') and self.alignment_results:
                            for prefix, group in self.alignment_results.items():
                                if camera_prefix == prefix and group.chunks:
                                    camera_offset = group.chunks[0].start_time_offset
                                    print(f"🎯 Retrieved camera offset for {camera_prefix}: {camera_offset:.3f}s")
                                    break

                        # Configure frame processing mode based on processing_type
                        if self.processing_type == "full_frames":
                            # Process full frames with alignment awareness
                            # IMPORTANT: Never seek in full_frames mode, always start from frame 0
                            detector.start_time_offset = camera_offset
                            detector.config['video']['process_matching_duration_only'] = False
                            detector.process_matching_duration_only = False

                            if self.limit_processing_duration:
                                # Apply duration limiting: process from 0 to max_processing_duration
                                detector.matching_duration = self.max_processing_duration
                                detector.end_time_offset = self.max_processing_duration
                                print(f"🎬 Full frame processing with duration limit: 0s to {detector.end_time_offset:.3f}s ({self.max_processing_duration}s)")
                                print(f"   Camera offset {camera_offset:.3f}s will be used for synced_time calculation only")
                            else:
                                # No duration limiting - process full frames to end
                                detector.matching_duration = 0.0  # Process to end
                                detector.end_time_offset = None  # No end limit
                                print(f"🎬 Full frame processing: camera offset {camera_offset:.3f}s (no duration limit)")

                        elif self.limit_processing_duration:
                            # Duration limiting with respect to processing type
                            detector.config['video']['process_matching_duration_only'] = True

                            # Start from alignment offset (if use_offset) and process for max_processing_duration
                            detector.start_time_offset = camera_offset if use_offset else 0.0
                            detector.matching_duration = self.max_processing_duration
                            detector.end_time_offset = detector.start_time_offset + self.max_processing_duration
                            detector.process_matching_duration_only = True

                            if use_offset:
                                print(f"⏱️ Duration limiting ({self.processing_type}): {camera_offset:.3f}s to {detector.end_time_offset:.3f}s ({self.max_processing_duration}s)")
                            else:
                                print(f"⏱️ Duration limiting: first {self.max_processing_duration}s from start")

                        else:
                            # Standard processing based on processing_type
                            detector.config['video']['process_matching_duration_only'] = True
                            detector.start_time_offset = camera_offset if use_offset else 0.0
                            # Let detector determine matching_duration from database
                            print(f"📏 Standard processing ({self.processing_type}): starting from {detector.start_time_offset:.3f}s")

                        # Configure output paths using configuration
                        os.makedirs(self.detection_videos_dir, exist_ok=True)

                        # Extract source name and configure output paths for detection video
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

                        detector.config['video']['output_video_path'] = detection_output_path
                        detector.config['video']['temp_video_path'] = temp_detection_path

                        # Update session ID to include camera prefix for unique database records
                        detector.session_id = f"{self.session_id}_{camera_prefix}"

                        # Configure video file name for database records
                        video_file_for_db = f"{source_name}_{camera_prefix}.mp4"

                        print(f"🎯 Detection configuration:")
                        print(f"   📹 Input video: {video_path}")
                        print(f"   💾 Database enabled: {detector.config['database']['enabled']}")
                        print(f"   🎬 Output video: {detection_output_path}")
                        print(f"   📊 Session ID: {detector.session_id}")
                        print(f"   📝 Video file (DB): {video_file_for_db}")
                        print(f"   🎚️ Save output video: {detector.config['video']['save_output_video']}")
                        print(f"   🔊 Preserve audio: {detector.config['video']['preserve_audio']}")
                        print(f"   📄 Generate report: {detector.config['video']['generate_report']}")

                        # Run detection with full workflow
                        detector.process_video(video_path)

                        # Cleanup detector
                        detector.cleanup()

                        print(f"✅ Full detection workflow complete for {camera_prefix}")
                        print(f"   💾 Database records saved with session: {detector.session_id}")
                        print(f"   🎬 Annotated video saved to: {detection_output_path}")

                        print(f"✅ Detection complete for {camera_prefix}")
                        success_count += 1

                        # Track detection output for unified video creation
                        self.detection_output_videos[camera_prefix] = detection_output_path

                    except Exception as e:
                        print(f"❌ Detection failed for {camera_prefix}: {e}")
                        import traceback
                        traceback.print_exc()

            print(f"\n✅ Detection processing complete: {success_count}/{total_videos} videos processed successfully")
            return success_count > 0

    def _create_unified_video(self) -> bool:
        """
        Create unified video by stacking all detection videos vertically with synchronized playback

        Returns:
            True if unified video creation successful, False otherwise
        """
        try:
            print(f"📹 Creating unified video from {len(self.detection_output_videos)} detection videos...")

            # Get configuration
            unified_config = self.unified_video_config
            output_filename = unified_config.get('output_filename', 'unified_detection_output.mp4')

            # Create output path with timestamp prefix
            unified_output_path = generate_unified_video_path(
                self.unified_videos_dir, output_filename, self.timestamp_prefix
            )

            # Get alignment data and video information
            video_info = {}
            max_duration = 0
            reference_fps = 30.0

            print("📊 Analyzing video properties and alignment data...")

            for camera_prefix, video_path in self.detection_output_videos.items():
                if not os.path.exists(video_path):
                    print(f"⚠️ Video file not found: {video_path}")
                    continue

                # Get video properties
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
                offset = 0.0
                if hasattr(self, 'alignment_results') and self.alignment_results:
                    for prefix, group in self.alignment_results.items():
                        if camera_prefix == prefix and group.chunks:
                            offset = group.chunks[0].start_time_offset
                            break

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
                reference_fps = fps  # Use the last fps as reference

                print(f"   📹 {camera_prefix}: {width}x{height}, {duration:.1f}s, offset: {offset:.3f}s")

            if not video_info:
                print("❌ No valid videos found for unified creation")
                return False

            print(f"🎬 Unified video specs: {max_duration:.1f}s total duration, {reference_fps:.1f} fps")

            return self._create_stacked_video_with_sync(video_info, unified_output_path, max_duration, reference_fps, unified_config)

        except Exception as e:
            print(f"❌ Error creating unified video: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_stacked_video_with_sync(self, video_info: dict, output_path: str, total_duration: float, fps: float, config: dict) -> bool:
        """
        Create stacked video with synchronized playback and freeze frame behavior

        Args:
            video_info: Dictionary containing video information for each camera
            output_path: Path for the output unified video
            total_duration: Total duration of the unified video
            fps: Frame rate for the output video
            config: Unified video configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            print("🎥 Starting synchronized video creation...")

            # Configuration
            stack_direction = config.get('stack_direction', 'vertical')
            add_labels = config.get('add_camera_labels', True)
            label_color = config.get('label_color', [255, 255, 255])
            label_font_scale = config.get('label_font_scale', 1.0)
            background_color = config.get('background_color', [0, 0, 0])

            # Determine output dimensions
            if stack_direction == 'vertical':
                output_width = max(info['width'] for info in video_info.values())
                output_height = sum(info['height'] for info in video_info.values())
            else:  # horizontal
                output_width = sum(info['width'] for info in video_info.values())
                output_height = max(info['height'] for info in video_info.values())

            print(f"📐 Output dimensions: {output_width}x{output_height}")

            # Initialize video captures and first/last frames
            video_captures = {}
            first_frames = {}
            last_frames = {}

            for camera_prefix, info in video_info.items():
                cap = cv2.VideoCapture(info['path'])
                if cap.isOpened():
                    video_captures[camera_prefix] = cap

                    # Get first frame
                    ret, frame = cap.read()
                    if ret:
                        first_frames[camera_prefix] = frame.copy()

                        # Get last frame by seeking to end
                        cap.set(cv2.CAP_PROP_POS_FRAMES, info['frame_count'] - 1)
                        ret, frame = cap.read()
                        if ret:
                            last_frames[camera_prefix] = frame.copy()

                        # Reset to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

            if not out.isOpened():
                print("❌ Failed to create video writer")
                for cap in video_captures.values():
                    cap.release()
                return False

            # Process frames
            total_frames = int(total_duration * fps)
            print(f"🎬 Processing {total_frames} frames...")

            for frame_idx in range(total_frames):
                current_time = frame_idx / fps
                unified_frame = np.full((output_height, output_width, 3), background_color, dtype=np.uint8)

                if stack_direction == 'vertical':
                    y_offset = 0
                    for camera_prefix, info in video_info.items():
                        frame = self._get_frame_at_time(video_captures.get(camera_prefix), info, current_time,
                                                      first_frames.get(camera_prefix), last_frames.get(camera_prefix))

                        if frame is not None:
                            # Resize frame to output width if needed
                            if frame.shape[1] != output_width:
                                frame = cv2.resize(frame, (output_width, info['height']))

                            # Add camera label if enabled
                            if add_labels:
                                text_size = cv2.getTextSize(camera_prefix, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 2)[0]
                                text_x = (frame.shape[1] - text_size[0]) // 2
                                cv2.putText(frame, camera_prefix, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                          label_font_scale, label_color, 2)

                            # Place frame in unified frame
                            unified_frame[y_offset:y_offset + info['height'], :] = frame

                        y_offset += info['height']
                else:  # horizontal
                    x_offset = 0
                    for camera_prefix, info in video_info.items():
                        frame = self._get_frame_at_time(video_captures.get(camera_prefix), info, current_time,
                                                      first_frames.get(camera_prefix), last_frames.get(camera_prefix))

                        if frame is not None:
                            # Resize frame to output height if needed
                            if frame.shape[0] != output_height:
                                frame = cv2.resize(frame, (info['width'], output_height))

                            # Add camera label if enabled
                            if add_labels:
                                text_size = cv2.getTextSize(camera_prefix, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 2)[0]
                                text_x = (frame.shape[1] - text_size[0]) // 2
                                cv2.putText(frame, camera_prefix, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                          label_font_scale, label_color, 2)

                            # Place frame in unified frame
                            unified_frame[:, x_offset:x_offset + info['width']] = frame

                        x_offset += info['width']

                out.write(unified_frame)

                # Progress feedback
                if frame_idx % int(fps * 5) == 0:  # Every 5 seconds
                    progress = (frame_idx / total_frames) * 100
                    print(f"   📹 Progress: {progress:.1f}% ({current_time:.1f}s / {total_duration:.1f}s)")

            # Cleanup
            out.release()
            for cap in video_captures.values():
                cap.release()

            print(f"✅ Unified video created successfully: {output_path}")

            # Add audio to the unified video
            print("🎵 Adding audio track to unified video...")
            final_output_path = self._add_audio_to_unified_video(output_path, video_info, total_duration)

            if final_output_path and final_output_path != output_path:
                # Replace the video without audio with the one with audio
                import os
                os.replace(final_output_path, output_path)
                print(f"✅ Audio integration completed: {output_path}")

            return True

        except Exception as e:
            print(f"❌ Error in stacked video creation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_frame_at_time(self, cap, video_info: dict, current_time: float, first_frame, last_frame):
        """
        Get the appropriate frame for a camera at the current time, handling freeze behavior

        Args:
            cap: Video capture object
            video_info: Video information dictionary
            current_time: Current time in the unified timeline
            first_frame: First frame of the video for freeze behavior
            last_frame: Last frame of the video for freeze behavior

        Returns:
            Frame to display at current time, or None if no frame available
        """
        if cap is None or not cap.isOpened():
            return first_frame if first_frame is not None else None

        video_start_time = video_info['offset']
        video_end_time = video_info['offset'] + video_info['duration']

        if current_time < video_start_time:
            # Before video starts - show first frame (frozen)
            return first_frame
        elif current_time > video_end_time:
            # After video ends - show last frame (frozen)
            return last_frame
        else:
            # During video playback - get the actual frame
            video_time = current_time - video_start_time
            frame_number = int(video_time * video_info['fps'])

            # Ensure frame number is within bounds
            frame_number = min(frame_number, video_info['frame_count'] - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                return frame
            else:
                # Fallback to last frame if read fails
                return last_frame if last_frame is not None else first_frame

    def _add_audio_to_unified_video(self, video_path: str, video_info: dict, total_duration: float) -> Optional[str]:
        """
        Add audio track to unified video using reference camera audio + extension if needed

        Args:
            video_path: Path to the video without audio
            video_info: Dictionary containing video information for each camera
            total_duration: Total duration of the unified video

        Returns:
            Path to final video with audio, or None if failed
        """
        try:
            import subprocess
            import tempfile

            if not video_info:
                print("⚠️ No video info available for audio processing")
                return None

            # Find reference camera (earliest start - this should match the logic used in alignment)
            reference_camera = None
            min_offset = float('inf')

            for camera_prefix, info in video_info.items():
                offset = info.get('offset', 0.0)
                if offset < min_offset:
                    min_offset = offset
                    reference_camera = camera_prefix

            if not reference_camera:
                print("⚠️ Could not determine reference camera for audio")
                return None

            print(f"🎯 Using {reference_camera} as reference audio source")

            # Get reference video info
            ref_info = video_info[reference_camera]
            ref_video_path = ref_info['path']
            ref_offset = ref_info.get('offset', 0.0)
            ref_duration = ref_info.get('duration', 0.0)
            ref_end_time = ref_offset + ref_duration

            # Determine audio segments needed
            audio_segments = []

            # 1. Primary audio: Reference camera from its start
            ref_audio_duration = min(ref_duration, total_duration - ref_offset) if ref_offset < total_duration else 0
            if ref_audio_duration > 0:
                audio_segments.append({
                    'path': ref_video_path,
                    'start': 0,  # Start from beginning of reference video
                    'duration': ref_audio_duration,
                    'camera': reference_camera
                })
                print(f"   📻 Primary audio: {reference_camera} (0s → {ref_audio_duration:.1f}s)")

            # 2. Extension audio: If reference ends before total duration, find next camera
            if ref_end_time < total_duration:
                remaining_duration = total_duration - ref_end_time

                # Find camera with content after reference ends (sorted by camera name for consistency)
                extension_candidates = []
                for camera_prefix, info in video_info.items():
                    if camera_prefix != reference_camera:
                        cam_offset = info.get('offset', 0.0)
                        cam_duration = info.get('duration', 0.0)
                        cam_end_time = cam_offset + cam_duration

                        if cam_end_time > ref_end_time:
                            # This camera has content after reference ends
                            available_duration = cam_end_time - ref_end_time
                            extension_candidates.append((camera_prefix, info, available_duration))

                if extension_candidates:
                    # Sort by camera name for consistent ordering
                    extension_candidates.sort(key=lambda x: x[0])

                    for camera_prefix, info, available_duration in extension_candidates:
                        if remaining_duration <= 0:
                            break

                        # Calculate how much audio to take from this camera
                        audio_duration = min(remaining_duration, available_duration)

                        # Calculate start time in the camera's video
                        cam_offset = info.get('offset', 0.0)
                        if ref_end_time >= cam_offset:
                            # Reference ended after this camera started
                            start_in_video = ref_end_time - cam_offset
                        else:
                            # This camera starts after reference ended (gap case)
                            start_in_video = 0

                        audio_segments.append({
                            'path': info['path'],
                            'start': start_in_video,
                            'duration': audio_duration,
                            'camera': camera_prefix
                        })

                        remaining_duration -= audio_duration
                        print(f"   📻 Extension audio: {camera_prefix} ({start_in_video:.1f}s → {start_in_video + audio_duration:.1f}s)")

            if not audio_segments:
                print("⚠️ No audio segments found")
                return None

            # Create temporary audio file for concatenation
            temp_dir = tempfile.mkdtemp()
            temp_audio_files = []
            concat_list_file = os.path.join(temp_dir, 'audio_concat.txt')

            # Extract audio segments
            for i, segment in enumerate(audio_segments):
                temp_audio_file = os.path.join(temp_dir, f'audio_segment_{i}.wav')

                # Extract audio segment using ffmpeg
                cmd = [
                    'ffmpeg', '-y',
                    '-i', segment['path'],
                    '-ss', str(segment['start']),
                    '-t', str(segment['duration']),
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # Consistent audio format
                    '-ar', '44100',  # Sample rate
                    '-ac', '2',  # Stereo
                    temp_audio_file
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    temp_audio_files.append(temp_audio_file)
                    print(f"   ✅ Extracted audio segment from {segment['camera']}")
                else:
                    print(f"   ❌ Failed to extract audio from {segment['camera']}: {result.stderr}")

            if not temp_audio_files:
                print("❌ No audio segments extracted successfully")
                return None

            # Create concatenation list
            with open(concat_list_file, 'w') as f:
                for audio_file in temp_audio_files:
                    f.write(f"file '{audio_file}'\n")

            # Concatenate audio files
            final_audio_file = os.path.join(temp_dir, 'final_audio.wav')
            if len(temp_audio_files) == 1:
                # Single file, just copy
                import shutil
                shutil.copy2(temp_audio_files[0], final_audio_file)
            else:
                # Multiple files, concatenate
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_list_file,
                    '-c', 'copy',
                    final_audio_file
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ Failed to concatenate audio: {result.stderr}")
                    return None

            # Combine video with audio
            output_with_audio = video_path.replace('.mp4', '_with_audio.mp4')
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,      # Video input
                '-i', final_audio_file, # Audio input
                '-c:v', 'copy',        # Copy video stream
                '-c:a', 'aac',         # Re-encode audio to AAC
                '-shortest',           # Match shortest stream duration
                output_with_audio
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            if result.returncode == 0:
                print(f"✅ Successfully created unified video with audio")
                return output_with_audio
            else:
                print(f"❌ Failed to combine video with audio: {result.stderr}")
                return None

        except Exception as e:
            print(f"❌ Error in audio processing: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        """Cleanup temporary files and resources"""
        if self.cleanup_temp_files:
            print("\n🧹 Cleaning up temporary files...")

            # Clean up any temporary files created during processing
            temp_patterns = [
                "temp_audio*",
                "*_concat.txt",
                "temp_video_no_audio.mp4",
                f"{self.detection_videos_dir}/*temp_detection*.mp4",  # Temp detection videos
                f"{self.unified_videos_dir}/*_with_audio.mp4"  # Intermediate unified videos
            ]

            cleanup_count = 0
            for pattern in temp_patterns:
                for temp_file in glob.glob(pattern):
                    try:
                        os.remove(temp_file)
                        print(f"🗑️ Removed: {temp_file}")
                        cleanup_count += 1
                    except Exception as e:
                        print(f"⚠️ Could not remove {temp_file}: {e}")

            if cleanup_count > 0:
                print(f"✅ Cleaned up {cleanup_count} temporary file(s)")
            else:
                print("✅ No temporary files to clean up")

        print("✅ Cleanup complete")

    def process(self) -> bool:
        """
        Main processing workflow

        Returns:
            True if processing successful, False otherwise
        """
        print(f"\n{'='*80}")
        print("INTEGRATED VIDEO PROCESSOR - STARTING")
        print(f"{'='*80}")
        print(f"Session ID: {self.session_id}")
        print(f"Processing directory: {self.alignment_directory}")

        try:
            # Step 1: Check for existing alignment data
            has_existing_data = self.check_existing_alignment_data()

            # Step 2: Analyze alignment if no existing data
            if not has_existing_data:
                print("\n📊 No existing alignment data found - analyzing videos...")
                if not self.analyze_video_alignment():
                    print("❌ Video alignment analysis failed")
                    return False
            else:
                print("\n✅ Using existing alignment data from database")

            # Step 3: Create aligned videos
            if not self.create_aligned_videos():
                print("❌ Aligned video creation failed")
                return False

            # Step 4: Run detection on aligned videos
            if not self.run_detection_on_aligned_videos():
                print("❌ Detection processing failed")
                return False

            # Step 5: Create unified video if enabled
            if self.unified_videos and hasattr(self, 'detection_output_videos') and self.detection_output_videos:
                print(f"\n{'='*80}")
                print("UNIFIED VIDEO CREATION")
                print(f"{'='*80}")
                if not self._create_unified_video():
                    print("⚠️ Unified video creation failed, but individual videos are available")
                else:
                    print("✅ Unified video created successfully")

            print(f"\n{'='*80}")
            print("INTEGRATED VIDEO PROCESSOR - COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            return True

        except Exception as e:
            print(f"❌ Fatal error in integrated processing: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Always cleanup
            self.cleanup()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Integrated Video Processor - Alignment + Detection')
    parser.add_argument('--config', '-c', default='src/config/config_v1.yaml',
                       help='Path to configuration file')
    parser.add_argument('--alignment-dir', '-a', type=str,
                       help='Path to alignment directory (overrides config)')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip detection processing (only do alignment)')
    parser.add_argument('--skip-video-creation', action='store_true',
                       help='Skip video creation (only do alignment analysis)')
    parser.add_argument('--max-duration', type=float,
                       help='Maximum duration to process per video (seconds)')
    parser.add_argument('--no-duration-limit', action='store_true',
                       help='Disable duration limiting (process full videos)')

    args = parser.parse_args()

    # Initialize processor
    processor = IntegratedVideoProcessor(config_path=args.config)

    # Override config with command line arguments
    if args.alignment_dir:
        processor.alignment_directory = args.alignment_dir
        processor.video_aligner_config['alignment_directory'] = args.alignment_dir

    if args.skip_detection:
        processor.run_detection = False

    if args.skip_video_creation:
        processor.save_output_videos = False

    # Handle duration limiting arguments
    if args.no_duration_limit:
        processor.limit_processing_duration = False
        print("🔄 Duration limiting disabled - will process full videos")
    elif args.max_duration:
        processor.limit_processing_duration = True
        processor.max_processing_duration = args.max_duration
        print(f"⏱️ Duration limiting set to {args.max_duration} seconds per video")

    try:
        # Run processing
        success = processor.process()
        exit_code = 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        exit_code = 130

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        # Final cleanup
        try:
            processor.cleanup()
        except:
            pass

    print(f"\n🏁 Process finished with exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()