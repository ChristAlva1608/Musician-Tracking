"""
Alignment Checker
Check if alignment data already exists in database
"""

import os
import glob
from typing import Dict, Any

from src.processors.base_processor import BaseProcessor, ValidationError
from src.database.setup import VideoAlignmentDatabase, ChunkVideoAlignmentDatabase
from src.helpers.video_path_helper import extract_source_name


class AlignmentChecker(BaseProcessor):
    """Check for existing alignment data in database"""

    def __init__(self, config: Dict[str, Any], session_id: str, alignment_directory: str):
        super().__init__(config, session_id)
        self.alignment_directory = alignment_directory
        self.enable_chunk_processing = config.get('video_aligner', {}).get('alignment', {}).get(
            'enable_chunk_processing', True
        )

        self.alignment_db = None
        self.chunk_alignment_db = None
        self.found_data = None

    def validate_dependencies(self):
        """Validate database dependencies"""
        required_packages = [
            {'name': 'sqlalchemy', 'import_name': 'sqlalchemy'},
            {'name': 'psycopg2', 'import_name': 'psycopg2'},
        ]
        self.require_packages(required_packages)

    def validate_inputs(self):
        """Validate alignment directory"""
        if not self.alignment_directory or not os.path.exists(self.alignment_directory):
            raise ValidationError(f"Alignment directory not found: {self.alignment_directory}")

        video_files = glob.glob(os.path.join(self.alignment_directory, "*.mp4"))
        if not video_files:
            raise ValidationError(f"No MP4 files found in {self.alignment_directory}")

        print(f"   📁 Directory: {self.alignment_directory}")
        print(f"   🎥 Found {len(video_files)} video files")

    def process(self) -> Dict[str, Any]:
        """Check for existing alignment data"""
        # Initialize databases
        try:
            self.alignment_db = VideoAlignmentDatabase()
            self.chunk_alignment_db = ChunkVideoAlignmentDatabase()
            print("✅ Connected to alignment databases")
        except Exception as e:
            raise ValidationError(f"Database connection failed: {e}")

        source_name = extract_source_name(self.alignment_directory)
        print(f"🔍 Looking for alignment data: {source_name}")

        if self.enable_chunk_processing:
            result = self.chunk_alignment_db.get_chunk_alignments_by_source(source_name)
            if result and len(result) > 0:
                print(f"✅ Found {len(result)} chunk alignment records")
                self.found_data = {'type': 'chunk', 'data': result}
                return {'has_existing_data': True, 'data_type': 'chunk', 'data': result}
        else:
            all_alignments = self.alignment_db.get_all_video_alignments()
            source_alignments = [a for a in all_alignments if a.get('source', '').startswith(source_name)]
            if source_alignments:
                print(f"✅ Found {len(source_alignments)} legacy alignment records")
                self.found_data = {'type': 'legacy', 'data': source_alignments}
                return {'has_existing_data': True, 'data_type': 'legacy', 'data': source_alignments}

        print(f"⚠️  No existing alignment data found")
        return {'has_existing_data': False}
