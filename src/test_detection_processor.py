#!/usr/bin/env python3
"""
Test Detection Processor
Test script for running detection processing independently
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.processors.detection_processor import DetectionProcessor, ProcessorError


def load_config(config_path: str = 'src/config/config_v1.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Test Detection Processor')
    parser.add_argument('--video', '-v', required=True, help='Path to video file')
    parser.add_argument('--camera', '-c', default='test_camera', help='Camera prefix (default: test_camera)')
    parser.add_argument('--config', default='src/config/config_v1.yaml', help='Config file path')
    parser.add_argument('--offset', type=float, default=0.0, help='Camera offset in seconds')
    parser.add_argument('--processing-type', choices=['use_offset', 'full_frames'],
                       default='full_frames', help='Processing type')
    parser.add_argument('--max-duration', type=float, help='Maximum duration to process (seconds)')

    args = parser.parse_args()

    # Validate video exists
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Override configuration with command line arguments
    integrated_config = config.get('integrated_processor', {})
    integrated_config['processing_type'] = args.processing_type

    if args.max_duration:
        integrated_config['limit_processing_duration'] = True
        integrated_config['max_processing_duration'] = args.max_duration
        print(f"⏱️  Duration limiting: {args.max_duration}s")
    else:
        integrated_config['limit_processing_duration'] = False
        print(f"⏱️  Duration limiting: DISABLED (full video)")

    config['integrated_processor'] = integrated_config

    # Create output videos dictionary
    output_videos = {
        args.camera: args.video
    }

    # Create session ID
    session_id = f"test_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create mock alignment results if offset provided
    alignment_results = {}
    if args.offset != 0.0:
        from types import SimpleNamespace
        chunk = SimpleNamespace(start_time_offset=args.offset)
        group = SimpleNamespace(chunks=[chunk])
        alignment_results[args.camera] = group
        print(f"📊 Using camera offset: {args.offset}s")

    # Create and run detection processor
    print(f"\n{'='*80}")
    print("DETECTION PROCESSOR TEST")
    print(f"{'='*80}")
    print(f"Video: {args.video}")
    print(f"Camera: {args.camera}")
    print(f"Processing type: {args.processing_type}")
    print(f"Session ID: {session_id}\n")

    try:
        processor = DetectionProcessor(
            config=config,
            session_id=session_id,
            output_videos=output_videos,
            alignment_results=alignment_results,
            alignment_directory=os.path.dirname(args.video)
        )

        results = processor.run()

        print(f"\n{'='*80}")
        print("TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"✅ Processed {results['success_count']}/{results['total_videos']} videos")
        print(f"\nOutput videos:")
        for camera, path in results['detection_output_videos'].items():
            print(f"  {camera}: {path}")
        print()

        return 0

    except ProcessorError as e:
        print(f"\n{'='*80}")
        print("TEST FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}\n")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n{'='*80}")
        print("TEST FAILED - UNEXPECTED ERROR")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
