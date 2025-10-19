"""
Helpers module for video processing utilities

Version Control:
- Use VIDEO_PATH_HELPER_VERSION to switch between v1 and v2
- v1: Original implementation for flat folder structure (e.g., vid_shot1/cam_1.mp4)
- v2: Enhanced for nested camera group structure (e.g., Source/CameraGroup1/chunks.mp4)

Usage:
    # To use V1 (default):
    from src.helpers import video_path_helper as path_helper

    # To use V2:
    from src.helpers import video_path_helper_v2 as path_helper

    # Or set version explicitly:
    from src.helpers import set_path_helper_version
    set_path_helper_version('v2')
"""

# Default to V1 for backward compatibility
VIDEO_PATH_HELPER_VERSION = 'v1'

# Import from v1 by default
from . import video_path_helper as _v1
from . import video_path_helper_v2 as _v2

# Current active version (default v1)
_active_helper = _v1


def set_path_helper_version(version: str):
    """
    Set the active path helper version

    Args:
        version: 'v1' or 'v2'
    """
    global _active_helper, VIDEO_PATH_HELPER_VERSION

    if version.lower() == 'v2':
        _active_helper = _v2
        VIDEO_PATH_HELPER_VERSION = 'v2'
    else:
        _active_helper = _v1
        VIDEO_PATH_HELPER_VERSION = 'v1'


# Export functions from active helper
def extract_source_name(*args, **kwargs):
    return _active_helper.extract_source_name(*args, **kwargs)

def generate_aligned_video_path(*args, **kwargs):
    return _active_helper.generate_aligned_video_path(*args, **kwargs)

def generate_detection_video_path(*args, **kwargs):
    return _active_helper.generate_detection_video_path(*args, **kwargs)

def generate_temp_detection_video_path(*args, **kwargs):
    return _active_helper.generate_temp_detection_video_path(*args, **kwargs)

def generate_unified_video_path(*args, **kwargs):
    return _active_helper.generate_unified_video_path(*args, **kwargs)

def generate_output_paths_from_alignment_data(*args, **kwargs):
    return _active_helper.generate_output_paths_from_alignment_data(*args, **kwargs)

def generate_output_paths_from_legacy_data(*args, **kwargs):
    return _active_helper.generate_output_paths_from_legacy_data(*args, **kwargs)


# Export version-specific modules for direct import
video_path_helper = _v1
video_path_helper_v2 = _v2


__all__ = [
    # Version control
    'set_path_helper_version',
    'VIDEO_PATH_HELPER_VERSION',
    'video_path_helper',
    'video_path_helper_v2',

    # Functions (dynamically routed based on version)
    'extract_source_name',
    'generate_aligned_video_path',
    'generate_detection_video_path',
    'generate_temp_detection_video_path',
    'generate_unified_video_path',
    'generate_output_paths_from_alignment_data',
    'generate_output_paths_from_legacy_data',
]
