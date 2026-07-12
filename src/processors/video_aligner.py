"""
Video Aligner
Create aligned videos from alignment data
"""

import os
import glob
from typing import Dict, Any

from src.processors.base_processor import BaseProcessor, ValidationError, ProcessorError
from src.video_aligner.shape_based_aligner_multi import combine_chunk_videos_timeline_based


class VideoAligner(BaseProcessor):
    """Create aligned videos based on alignment results"""

    def __init__(self, config: Dict[str, Any], session_id: str, alignment_results: Dict,
                 alignment_directory: str):
        super().__init__(config, session_id)
        self.alignment_results = alignment_results
        self.alignment_directory = alignment_directory
        self.output_videos = {}

    def validate_dependencies(self):
        """Validate dependencies for video alignment"""
        required_packages = [
            {'name': 'cv2', 'import_name': 'cv2', 'install_name': 'opencv-python'},
            {'name': 'numpy', 'import_name': 'numpy'},
        ]
        self.require_packages(required_packages)

    def validate_inputs(self):
        """Validate alignment results and directory"""
        if not self.alignment_results:
            raise ValidationError("No alignment results provided")

        if not os.path.exists(self.alignment_directory):
            raise ValidationError(f"Alignment directory not found: {self.alignment_directory}")

        print(f"   📁 Directory: {self.alignment_directory}")
        print(f"   📊 {len(self.alignment_results)} camera groups to align")

    def process(self) -> Dict[str, Any]:
        """Create aligned videos"""
        has_chunks = self._has_chunk_videos()

        if has_chunks:
            print("📦 Detected chunk videos - merging into aligned videos...")
            return self._create_chunk_aligned_videos()
        else:
            print("📁 Detected regular videos - using original paths...")
            return self._map_original_video_paths()

    def _has_chunk_videos(self) -> bool:
        """Check if alignment results contain chunk videos"""
        for prefix, group in self.alignment_results.items():
            if hasattr(group, 'chunks') and len(group.chunks) > 1:
                print(f"   🔍 Camera {prefix} has {len(group.chunks)} chunks")
                return True
        return False

    def _create_chunk_aligned_videos(self) -> Dict[str, Any]:
        """Create aligned videos from chunks"""
        combined_videos = combine_chunk_videos_timeline_based(self.alignment_results)

        success_count = 0
        for prefix, output_path in combined_videos.items():
            if os.path.exists(output_path):
                print(f"✅ Created: {output_path}")
                self.output_videos[prefix] = output_path
                success_count += 1
            else:
                print(f"❌ Failed to create: {output_path}")

        if success_count == 0:
            raise ProcessorError("No aligned videos were created")

        print(f"✅ Created {success_count}/{len(combined_videos)} aligned videos")
        return {'output_videos': self.output_videos, 'success_count': success_count}

    def _map_original_video_paths(self) -> Dict[str, Any]:
        """Map original video paths without copying"""
        video_files = glob.glob(os.path.join(self.alignment_directory, "*.mp4"))
        if not video_files:
            raise ProcessorError("No video files found in alignment directory")

        for video_file in sorted(video_files):
            filename = os.path.basename(video_file)
            camera_prefix = os.path.splitext(filename)[0]
            self.output_videos[camera_prefix] = video_file
            print(f"   ✅ {camera_prefix} -> {video_file}")

        print(f"✅ Mapped {len(self.output_videos)} video paths")
        return {'output_videos': self.output_videos, 'success_count': len(self.output_videos)}
