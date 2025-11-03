#!/usr/bin/env python3
"""
Integrated Video Processor (Refactored)
Combines video alignment and detection processing using modular processors

This module orchestrates the complete workflow:
1. Check for existing alignment data (AlignmentChecker)
2. Analyze videos if no alignment exists (AlignmentAnalyzer)
3. Create aligned output videos (VideoAligner)
4. Run detection processing (DetectionProcessor)
5. Create unified stacked videos (UnifiedVideoCreator)
"""

import os
import sys
import yaml
from datetime import datetime
from typing import Dict

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import modular processors
from src.processors.alignment_checker import AlignmentChecker
from src.processors.alignment_analyzer import AlignmentAnalyzer
from src.processors.video_aligner import VideoAligner
from src.processors.detection_processor import DetectionProcessor
from src.processors.unified_video_creator import UnifiedVideoCreator


class IntegratedVideoProcessor:
    """
    Refactored integrated processor using modular components
    Orchestrates the complete video processing pipeline
    """

    def __init__(self, config_path: str = 'src/config/config_v1.yaml'):
        """
        Initialize the integrated processor

        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.session_id = f"integrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Extract configuration
        self.video_aligner_config = self.config.get('video_aligner', {})
        self.integrated_config = self.config.get('integrated_processor', {})

        # Processing settings
        self.alignment_directory = self.integrated_config.get(
            'alignment_directory',
            self.video_aligner_config.get('alignment_directory', '')
        )
        self.run_detection = self.integrated_config.get('run_detection', True)
        self.create_aligned_videos_flag = self.integrated_config.get('create_aligned_videos', True)
        self.check_existing_alignment = self.integrated_config.get('check_existing_alignment', True)
        self.unified_videos = self.integrated_config.get('unified_videos', False)

        # Output directories
        self.aligned_videos_dir = self.integrated_config.get('aligned_videos_dir', 'src/output/aligned_videos')
        self.detection_videos_dir = self.integrated_config.get('detection_videos_dir', 'src/output/annotated_detection_videos')
        self.unified_videos_dir = self.integrated_config.get('unified_videos_dir', 'src/output/unified_videos')

        # Results storage
        self.alignment_results = {}
        self.output_videos = {}
        self.detection_output_videos = {}

        # Create output directories
        os.makedirs(self.aligned_videos_dir, exist_ok=True)
        os.makedirs(self.detection_videos_dir, exist_ok=True)
        if self.unified_videos:
            os.makedirs(self.unified_videos_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print("INTEGRATED VIDEO PROCESSOR (REFACTORED)")
        print(f"{'='*80}")
        print(f"Session ID: {self.session_id}")
        print(f"Alignment directory: {self.alignment_directory}")
        print(f"Check existing alignment: {self.check_existing_alignment}")
        print(f"Create aligned videos: {self.create_aligned_videos_flag}")
        print(f"Run detection: {self.run_detection}")
        print(f"Unified videos: {self.unified_videos}")
        print(f"{'='*80}\n")

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
        Step 1: Check if alignment data already exists in database

        Returns:
            True if alignment data exists and was loaded, False otherwise
        """
        if not self.check_existing_alignment:
            print("⏭️  Skipping alignment check (check_existing_alignment disabled)")
            return False

        print(f"\n{'='*70}")
        print("STEP 1: CHECK EXISTING ALIGNMENT DATA")
        print(f"{'='*70}\n")

        try:
            checker = AlignmentChecker(
                config=self.config,
                session_id=self.session_id,
                alignment_directory=self.alignment_directory
            )

            results = checker.run()

            if results.get('has_existing_data'):
                data_type = results.get('data_type')
                print(f"✅ Found existing {data_type} alignment data")

                # Store results based on type
                if data_type == 'chunk':
                    # Convert chunk data to camera groups
                    self.alignment_results = self._load_chunk_alignment_data(results['data'])
                elif data_type == 'legacy':
                    self.alignment_results = results['data']

                return True
            else:
                print("⚠️  No existing alignment data found")
                return False

        except Exception as e:
            print(f"⚠️  Alignment check failed: {e}")
            print("   Will proceed with alignment analysis")
            return False

    def analyze_video_alignment(self) -> bool:
        """
        Step 2: Analyze video alignment using audio patterns

        Returns:
            True if analysis successful, False otherwise
        """
        print(f"\n{'='*70}")
        print("STEP 2: ANALYZE VIDEO ALIGNMENT")
        print(f"{'='*70}\n")

        try:
            analyzer = AlignmentAnalyzer(
                config=self.config,
                session_id=self.session_id,
                alignment_directory=self.alignment_directory
            )

            results = analyzer.run()

            # Store results
            self.alignment_results = results.get('camera_groups', {})
            self.output_videos = results.get('output_videos', {})

            print(f"✅ Alignment analysis complete")
            print(f"   Camera groups: {len(self.alignment_results)}")
            print(f"   Reference camera: {results.get('reference_prefix', 'N/A')}")

            return True

        except Exception as e:
            print(f"❌ Alignment analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_aligned_videos(self) -> bool:
        """
        Step 3: Create aligned videos from alignment results

        Returns:
            True if video creation successful, False otherwise
        """
        if not self.create_aligned_videos_flag:
            print("\n⏭️  Skipping aligned video creation (create_aligned_videos disabled)")
            # Map original video paths for detection
            self._map_original_videos()
            return True

        print(f"\n{'='*70}")
        print("STEP 3: CREATE ALIGNED VIDEOS")
        print(f"{'='*70}\n")

        try:
            aligner = VideoAligner(
                config=self.config,
                session_id=self.session_id,
                alignment_results=self.alignment_results,
                alignment_directory=self.alignment_directory
            )

            results = aligner.run()

            # Store output video paths
            self.output_videos = results.get('output_videos', {})

            print(f"✅ Video alignment complete")
            print(f"   Created: {results.get('success_count', 0)} videos")

            return results.get('success_count', 0) > 0

        except Exception as e:
            print(f"❌ Video alignment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _map_original_videos(self):
        """Map original video paths when skipping aligned video creation"""
        import glob

        video_files = glob.glob(os.path.join(self.alignment_directory, "*.mp4"))
        self.output_videos = {}

        for video_file in sorted(video_files):
            filename = os.path.basename(video_file)
            camera_prefix = os.path.splitext(filename)[0]
            self.output_videos[camera_prefix] = video_file

        print(f"✅ Mapped {len(self.output_videos)} original video paths")

    def run_detection_on_aligned_videos(self) -> bool:
        """
        Step 4: Run detection processing on aligned videos

        Returns:
            True if detection successful, False otherwise
        """
        if not self.run_detection:
            print("\n⏭️  Skipping detection processing (run_detection disabled)")
            return True

        print(f"\n{'='*70}")
        print("STEP 4: RUN DETECTION ON ALIGNED VIDEOS")
        print(f"{'='*70}\n")

        try:
            processor = DetectionProcessor(
                config=self.config,
                session_id=self.session_id,
                output_videos=self.output_videos,
                alignment_results=self.alignment_results,
                alignment_directory=self.alignment_directory
            )

            results = processor.run()

            # Store detection output videos
            self.detection_output_videos = results.get('detection_output_videos', {})

            print(f"✅ Detection processing complete")
            print(f"   Processed: {results.get('success_count', 0)}/{results.get('total_videos', 0)} videos")

            return results.get('success_count', 0) > 0

        except Exception as e:
            print(f"❌ Detection processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_chunk_alignment_data(self, chunk_data: list) -> dict:
        """
        Convert chunk alignment data from database to camera groups

        Args:
            chunk_data: List of chunk alignment records from database

        Returns:
            Dictionary mapping camera prefix to CameraGroup objects
        """
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

        print(f"   Loaded {len(camera_groups)} camera groups from database")
        return camera_groups

    def create_unified_video(self) -> bool:
        """
        Step 5: Create unified stacked video from detection outputs

        Returns:
            True if unified video creation successful, False otherwise
        """
        if not self.unified_videos:
            print("\n⏭️  Skipping unified video creation (unified_videos disabled)")
            return True

        if not self.detection_output_videos:
            print("\n⚠️  No detection videos available for unified creation")
            return False

        print(f"\n{'='*70}")
        print("STEP 5: CREATE UNIFIED VIDEO")
        print(f"{'='*70}\n")

        try:
            creator = UnifiedVideoCreator(
                config=self.config,
                session_id=self.session_id,
                detection_output_videos=self.detection_output_videos,
                alignment_results=self.alignment_results,
                unified_videos_dir=self.unified_videos_dir,
                timestamp_prefix=self.timestamp_prefix
            )

            results = creator.run()

            print(f"✅ Unified video creation complete")
            print(f"   Output: {results.get('unified_video_path')}")
            print(f"   Duration: {results.get('duration', 0):.1f}s")
            print(f"   Videos combined: {results.get('num_videos', 0)}")

            return True

        except Exception as e:
            print(f"⚠️  Unified video creation failed: {e}")
            print("   Individual detection videos are still available")
            import traceback
            traceback.print_exc()
            return False

    def process(self) -> bool:
        """
        Main processing workflow
        Orchestrates all steps in sequence

        Returns:
            True if processing successful, False otherwise
        """
        print(f"\n{'='*80}")
        print("INTEGRATED VIDEO PROCESSING - STARTING")
        print(f"{'='*80}\n")

        try:
            # Step 1: Check for existing alignment data
            has_existing_data = self.check_existing_alignment_data()

            # Step 2: Analyze alignment if no existing data
            if not has_existing_data:
                if not self.analyze_video_alignment():
                    print("❌ Video alignment analysis failed")
                    return False
            else:
                print("✅ Using existing alignment data from database")

            # Step 3: Create aligned videos
            if not self.create_aligned_videos():
                print("❌ Aligned video creation failed")
                return False

            # Step 4: Run detection on aligned videos
            if not self.run_detection_on_aligned_videos():
                print("❌ Detection processing failed")
                return False

            # Step 5: Create unified video
            if not self.create_unified_video():
                print("⚠️  Unified video creation skipped or failed")
                # Not a fatal error - individual videos are available

            # Success
            print(f"\n{'='*80}")
            print("INTEGRATED VIDEO PROCESSING - COMPLETED SUCCESSFULLY")
            print(f"{'='*80}\n")

            self._print_summary()
            return True

        except Exception as e:
            print(f"\n❌ Fatal error in integrated processing: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _print_summary(self):
        """Print processing summary"""
        print("\n📊 PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Session ID: {self.session_id}")
        print(f"Alignment directory: {self.alignment_directory}")
        print(f"\nResults:")
        print(f"  Camera groups analyzed: {len(self.alignment_results)}")
        print(f"  Aligned videos created: {len(self.output_videos)}")
        print(f"  Detection videos created: {len(self.detection_output_videos)}")

        if self.output_videos:
            print(f"\nAligned videos:")
            for camera, path in self.output_videos.items():
                print(f"  {camera}: {path}")

        if self.detection_output_videos:
            print(f"\nDetection videos:")
            for camera, path in self.detection_output_videos.items():
                print(f"  {camera}: {path}")

        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrated Video Processor (Refactored) - Alignment + Detection'
    )
    parser.add_argument('--config', '-c', default='src/config/config_v1.yaml',
                       help='Path to configuration file')
    parser.add_argument('--alignment-dir', '-a', type=str,
                       help='Path to alignment directory (overrides config)')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip detection processing (only do alignment)')
    parser.add_argument('--skip-check', action='store_true',
                       help='Skip checking for existing alignment data')
    parser.add_argument('--skip-video-creation', action='store_true',
                       help='Skip creating aligned videos (use originals)')
    parser.add_argument('--unified', action='store_true',
                       help='Create unified stacked video')

    args = parser.parse_args()

    # Initialize processor
    processor = IntegratedVideoProcessor(config_path=args.config)

    # Override config with command line arguments
    if args.alignment_dir:
        processor.alignment_directory = args.alignment_dir
        print(f"🔄 Overriding alignment directory: {args.alignment_dir}")

    if args.skip_detection:
        processor.run_detection = False
        print("🔄 Detection processing disabled")

    if args.skip_check:
        processor.check_existing_alignment = False
        print("🔄 Alignment check disabled")

    if args.skip_video_creation:
        processor.create_aligned_videos_flag = False
        print("🔄 Aligned video creation disabled")

    if args.unified:
        processor.unified_videos = True
        print("🔄 Unified video creation enabled")

    try:
        # Run processing
        success = processor.process()
        exit_code = 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        exit_code = 130

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    print(f"\n🏁 Process finished with exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
