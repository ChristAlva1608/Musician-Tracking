#!/usr/bin/env python3
"""
Complete Video Conversion Workflow

Orchestrates the entire video preparation pipeline:
1. Scan and validate participant data
2. Interactive angle selection for 360° cameras
3. Batch convert 360° to 2D perspective
4. Process and rename 2D camera videos
5. Generate summary report

Usage:
    python video_conversion_workflow.py --participant P03
    python video_conversion_workflow.py --participant P03 --skip-interactive
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Import our conversion modules
from video_converter_360_to_2d import VideoConverter360to2D
from camera_2d_renamer import Camera2DRenamer


class VideoConversionWorkflow:
    """
    Complete workflow for converting participant videos to pose detection format.
    """

    def __init__(self, data_root: str, output_root: str):
        """
        Initialize workflow.

        Args:
            data_root: Path to participant data
            output_root: Path to output processed videos
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)

        self.converter_360 = VideoConverter360to2D(data_root, output_root)
        self.renamer_2d = Camera2DRenamer(data_root, output_root)

        self.report_file = self.output_root / "conversion_report.json"

    def run_full_workflow(self, participant_id: str, date_folder: str = None,
                          skip_interactive: bool = False):
        """
        Run complete conversion workflow.

        Args:
            participant_id: Participant ID (e.g., P03)
            date_folder: Date folder name (auto-detected if not provided)
            skip_interactive: Skip interactive angle selection (use saved angles)
        """
        print("\n" + "="*70)
        print("🎬 VIDEO CONVERSION WORKFLOW")
        print("="*70)
        print(f"Participant: {participant_id}")
        print(f"Data root: {self.data_root}")
        print(f"Output root: {self.output_root}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        report = {
            'participant': participant_id,
            'start_time': datetime.now().isoformat(),
            'steps': []
        }

        # Step 1: Scan videos
        print("\n📊 STEP 1: Scanning participant videos...")
        print("-" * 70)

        videos = self.converter_360.scan_participant_videos(participant_id, date_folder)
        videos_2d = self.renamer_2d.scan_2d_cameras(participant_id, date_folder)

        # Merge results for reporting
        all_videos = {**videos, **videos_2d}

        total_videos = sum(len(v) for v in all_videos.values())

        print(f"\n📹 Video Summary:")
        for camera_type, video_list in all_videos.items():
            count = len(video_list)
            if count > 0:
                # Handle both list and dict formats
                if isinstance(video_list, list) and len(video_list) > 0:
                    if isinstance(video_list[0], dict):
                        print(f"   {camera_type:10s}: {count:3d} videos")
                    else:
                        print(f"   {camera_type:10s}: {count:3d} videos")

        print(f"\n   TOTAL: {total_videos} videos")

        report['steps'].append({
            'step': 1,
            'name': 'scan',
            'status': 'completed',
            'video_count': total_videos,
            'cameras': {k: len(v) for k, v in all_videos.items() if len(v) > 0}
        })

        if total_videos == 0:
            print("\n⚠️  No videos found! Please check the participant folder.")
            return

        # Step 2: Check for 360° cameras
        has_360_cameras = any(len(all_videos[cam]) > 0 for cam in ['X3', 'X4', 'X5'])

        if has_360_cameras:
            # Step 2a: Interactive angle selection (if needed)
            if not skip_interactive:
                print("\n\n🎯 STEP 2: Interactive angle selection for 360° cameras...")
                print("-" * 70)
                print("\nThis step allows you to select the viewing angle for each 360° camera.")
                print("You'll see a preview window where you can adjust the angle using arrow keys.")
                print("\nPress Enter to continue, or 's' to skip...")

                response = input().strip().lower()
                if response != 's':
                    try:
                        self.converter_360.interactive_angle_selection(participant_id, date_folder)
                        report['steps'].append({
                            'step': 2,
                            'name': 'interactive_angles',
                            'status': 'completed'
                        })
                    except Exception as e:
                        print(f"\n⚠️  Interactive selection failed: {e}")
                        print("You can run this step separately later.")
                        report['steps'].append({
                            'step': 2,
                            'name': 'interactive_angles',
                            'status': 'skipped',
                            'reason': str(e)
                        })
                else:
                    print("⏭️  Skipped interactive selection")
                    report['steps'].append({
                        'step': 2,
                        'name': 'interactive_angles',
                        'status': 'skipped',
                        'reason': 'user_skipped'
                    })

            # Step 3: Batch convert 360° videos
            print("\n\n🚀 STEP 3: Batch converting 360° videos to 2D perspective...")
            print("-" * 70)

            # Check if angles are saved
            angles = self.converter_360.load_camera_angles()
            if participant_id not in angles or not angles[participant_id]:
                print("\n⚠️  No saved angles found!")
                print("Please run interactive mode first to select viewing angles.")
                print("\nRun: python video_converter_360_to_2d.py --mode interactive --participant", participant_id)
                report['steps'].append({
                    'step': 3,
                    'name': 'batch_convert_360',
                    'status': 'skipped',
                    'reason': 'no_angles_saved'
                })
            else:
                print(f"\n✅ Found saved angles for {len(angles[participant_id])} cameras")
                print("\nStarting batch conversion...")

                try:
                    self.converter_360.batch_convert(participant_id, date_folder)
                    report['steps'].append({
                        'step': 3,
                        'name': 'batch_convert_360',
                        'status': 'completed'
                    })
                except Exception as e:
                    print(f"\n❌ Batch conversion failed: {e}")
                    report['steps'].append({
                        'step': 3,
                        'name': 'batch_convert_360',
                        'status': 'failed',
                        'error': str(e)
                    })

        # Step 4: Process 2D cameras
        has_2d_cameras = len(videos_2d.get('iPhone', [])) > 0 or len(videos_2d.get('GoPro', [])) > 0

        if has_2d_cameras:
            print("\n\n📱 STEP 4: Processing 2D cameras (iPhone/GoPro)...")
            print("-" * 70)
            print("\nChoose processing mode:")
            print("  1. Copy (fast, keeps original format)")
            print("  2. Transcode (slower, standardizes format to 1920x1080)")
            print("\nEnter choice (1/2) or 's' to skip: ", end='')

            response = input().strip()

            if response == '1':
                mode = 'rename'
                print("\n📋 Copying and renaming 2D camera videos...")
            elif response == '2':
                mode = 'transcode'
                print("\n🎬 Transcoding 2D camera videos...")
            else:
                print("⏭️  Skipped 2D camera processing")
                report['steps'].append({
                    'step': 4,
                    'name': 'process_2d_cameras',
                    'status': 'skipped',
                    'reason': 'user_skipped'
                })
                mode = None

            if mode:
                try:
                    process_mode = 'copy' if mode == 'rename' else 'transcode'
                    self.renamer_2d.rename_videos(participant_id, date_folder, mode=process_mode)
                    report['steps'].append({
                        'step': 4,
                        'name': 'process_2d_cameras',
                        'status': 'completed',
                        'mode': mode
                    })
                except Exception as e:
                    print(f"\n❌ 2D camera processing failed: {e}")
                    report['steps'].append({
                        'step': 4,
                        'name': 'process_2d_cameras',
                        'status': 'failed',
                        'error': str(e)
                    })

        # Final summary
        print("\n\n" + "="*70)
        print("✅ WORKFLOW COMPLETE")
        print("="*70)

        report['end_time'] = datetime.now().isoformat()
        report['status'] = 'completed'

        # Save report
        self._save_report(report)

        print(f"\n📊 Summary:")
        print(f"   Total videos scanned: {total_videos}")
        print(f"   Steps completed: {len([s for s in report['steps'] if s['status'] == 'completed'])}/{len(report['steps'])}")
        print(f"\n📄 Full report saved to: {self.report_file}")

        print(f"\n📁 Output location: {self.output_root / participant_id}")

        print("\n🎯 Next Steps:")
        print("   1. Review conversion logs in:", self.output_root)
        print("   2. Run hybrid alignment to detect gaps and synchronize videos")
        print("   3. Run pose detection on converted videos")

        print("\n" + "="*70)

    def _save_report(self, report: dict):
        """Save workflow report to JSON."""
        self.report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.report_file, 'w') as f:
            json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Complete video conversion workflow for pose detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow with interactive angle selection
  python video_conversion_workflow.py --participant P03

  # Skip interactive mode (use saved angles)
  python video_conversion_workflow.py --participant P03 --skip-interactive

  # Specify custom paths
  python video_conversion_workflow.py --participant P03 \\
      --data-root "/Volumes/MyDrive/Data" \\
      --output-root "./output"
        """
    )

    parser.add_argument('--participant', required=True,
                       help='Participant ID (e.g., P03)')
    parser.add_argument('--date-folder',
                       help='Date folder name (auto-detected if not provided)')
    parser.add_argument('--skip-interactive', action='store_true',
                       help='Skip interactive angle selection (use saved angles)')
    parser.add_argument('--data-root', default='/Volumes/X10 Pro 1/Phan Dissertation Data',
                       help='Root path to participant data')
    parser.add_argument('--output-root', default='./Processed_Videos',
                       help='Root path for output processed videos')

    args = parser.parse_args()

    workflow = VideoConversionWorkflow(args.data_root, args.output_root)
    workflow.run_full_workflow(args.participant, args.date_folder, args.skip_interactive)


if __name__ == '__main__':
    main()
