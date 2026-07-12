"""
Alignment Analyzer
Analyze video alignment using shape-based aligner logic
"""

import os
from typing import Dict, Any

from src.processors.base_processor import BaseProcessor, ValidationError, ProcessorError
from src.video_aligner.shape_based_aligner_multi import (
    scan_and_group_chunk_videos,
    determine_reference_camera_group_by_audio_pattern,
    align_chunks_to_reference_timeline,
    generate_output_video_paths
)
from src.helpers.video_path_helper import extract_source_name


class AlignmentAnalyzer(BaseProcessor):
    """Analyze video alignment using chunk processing or legacy logic"""

    def __init__(self, config: Dict[str, Any], session_id: str, alignment_directory: str):
        super().__init__(config, session_id)
        self.alignment_directory = alignment_directory
        self.video_aligner_config = config.get('video_aligner', {})
        self.integrated_config = config.get('integrated_processor', {})
        self.enable_chunk_processing = self.video_aligner_config.get('alignment', {}).get(
            'enable_chunk_processing', True
        )
        self.processing_type = self.integrated_config.get('processing_type', 'use_offset')
        self.unified_videos = self.integrated_config.get('unified_videos', False)

        self.alignment_results = {}
        self.output_videos = {}

    def validate_dependencies(self):
        """Validate dependencies for alignment analysis"""
        required_packages = [
            {'name': 'cv2', 'import_name': 'cv2', 'install_name': 'opencv-python'},
            {'name': 'numpy', 'import_name': 'numpy'},
        ]
        self.require_packages(required_packages)

    def validate_inputs(self):
        """Validate alignment directory"""
        if not os.path.exists(self.alignment_directory):
            raise ValidationError(f"Alignment directory not found: {self.alignment_directory}")

        print(f"   📁 Directory: {self.alignment_directory}")
        print(f"   📦 Chunk processing: {self.enable_chunk_processing}")

    def process(self) -> Dict[str, Any]:
        """Analyze video alignment"""
        if self.enable_chunk_processing:
            return self._analyze_chunk_alignment()
        else:
            raise ProcessorError("Legacy alignment not yet implemented in modular version")

    def _analyze_chunk_alignment(self) -> Dict[str, Any]:
        """Analyze alignment using chunk processing logic"""
        print("🔍 Scanning and grouping chunk videos...")
        camera_groups = scan_and_group_chunk_videos(self.alignment_directory)
        if not camera_groups:
            raise ProcessorError("No camera groups found")

        # Determine reference strategy
        if self.unified_videos:
            use_earliest_start = True
            method_type = 'earliest_start'
            reason = "unified_videos requires full timeline"
        elif self.processing_type == "full_frames":
            use_earliest_start = True
            method_type = 'earliest_start'
            reason = "full_frames processes all content"
        else:
            use_earliest_start = False
            method_type = 'latest_start'
            reason = "use_offset processes only synchronized portion"

        print(f"🎯 Alignment strategy: {method_type} ({reason})")

        reference_prefix = determine_reference_camera_group_by_audio_pattern(camera_groups, use_earliest_start)
        camera_groups = align_chunks_to_reference_timeline(camera_groups, reference_prefix, use_earliest_start)
        camera_groups = generate_output_video_paths(camera_groups, self.config)

        # Extract output paths
        output_videos = {}
        for prefix, group in camera_groups.items():
            if hasattr(group, 'output_video_path'):
                output_videos[prefix] = group.output_video_path

        print(f"✅ Alignment complete for {len(camera_groups)} camera groups")

        return {
            'camera_groups': camera_groups,
            'output_videos': output_videos,
            'reference_prefix': reference_prefix,
            'method_type': method_type
        }
