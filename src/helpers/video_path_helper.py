#!/usr/bin/env python3
"""
Video Path Helper
Utilities for managing video file paths in the integrated video processing pipeline
"""

import os
from typing import Dict, Optional
from datetime import datetime


def extract_source_name(alignment_directory: str) -> str:
    """
    Extract source name from alignment directory path

    Args:
        alignment_directory: Path to the alignment directory

    Returns:
        Source name extracted from the path (e.g., 'vid_shot1')

    Example:
        >>> extract_source_name('/path/to/vid_shot1/videos')
        'vid_shot1'
    """
    # Extract source name from directory path (e.g., vid_shot1)
    path_parts = alignment_directory.split('/')
    vid_shot_parts = [part for part in path_parts if part.startswith('vid_shot')]

    if vid_shot_parts:
        return vid_shot_parts[0]
    else:
        # Fallback to directory name
        return os.path.basename(os.path.dirname(alignment_directory.rstrip('/')))


def generate_aligned_video_path(
    output_dir: str,
    source_name: str,
    camera_prefix: str,
    timestamp_prefix: Optional[str] = None
) -> str:
    """
    Generate output path for aligned video

    Args:
        output_dir: Directory for aligned videos
        source_name: Source name (e.g., 'vid_shot1')
        camera_prefix: Camera identifier (e.g., 'cam_1')
        timestamp_prefix: Optional timestamp prefix (generated if not provided)

    Returns:
        Full path for the aligned video file

    Example:
        >>> generate_aligned_video_path('output/aligned', 'vid_shot1', 'cam_1')
        'output/aligned/20250119_143052_vid_shot1_cam_1.mp4'
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{timestamp_prefix}_{source_name}_{camera_prefix}.mp4"
    return os.path.join(output_dir, filename)


def generate_detection_video_path(
    output_dir: str,
    source_name: str,
    camera_prefix: str,
    timestamp_prefix: Optional[str] = None,
    output_format: str = 'mp4'
) -> str:
    """
    Generate output path for detection video

    Args:
        output_dir: Directory for detection videos
        source_name: Source name (e.g., 'vid_shot1')
        camera_prefix: Camera identifier (e.g., 'cam_1')
        timestamp_prefix: Optional timestamp prefix (generated if not provided)
        output_format: Video output format (default: 'mp4')

    Returns:
        Full path for the detection video file

    Example:
        >>> generate_detection_video_path('output/detection', 'vid_shot1', 'cam_1')
        'output/detection/20250119_143052_detection_vid_shot1_cam_1.mp4'
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{timestamp_prefix}_detection_{source_name}_{camera_prefix}.{output_format}"
    return os.path.join(output_dir, filename)


def generate_temp_detection_video_path(
    output_dir: str,
    source_name: str,
    camera_prefix: str,
    timestamp_prefix: Optional[str] = None,
    output_format: str = 'mp4'
) -> str:
    """
    Generate temporary path for detection video (used during processing)

    Args:
        output_dir: Directory for detection videos
        source_name: Source name (e.g., 'vid_shot1')
        camera_prefix: Camera identifier (e.g., 'cam_1')
        timestamp_prefix: Optional timestamp prefix (generated if not provided)
        output_format: Video output format (default: 'mp4')

    Returns:
        Full path for the temporary detection video file
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    filename = f"{timestamp_prefix}_temp_detection_{source_name}_{camera_prefix}.{output_format}"
    return os.path.join(output_dir, filename)


def generate_unified_video_path(
    output_dir: str,
    output_filename: str,
    timestamp_prefix: Optional[str] = None
) -> str:
    """
    Generate output path for unified video

    Args:
        output_dir: Directory for unified videos
        output_filename: Base filename for unified video
        timestamp_prefix: Optional timestamp prefix (generated if not provided)

    Returns:
        Full path for the unified video file

    Example:
        >>> generate_unified_video_path('output/unified', 'unified_detection_output.mp4')
        'output/unified/20250119_143052_unified_detection_output.mp4'
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(output_dir, exist_ok=True)
    timestamped_filename = f"{timestamp_prefix}_{output_filename}"
    return os.path.join(output_dir, timestamped_filename)


def generate_output_paths_from_alignment_data(
    alignment_results: Dict,
    alignment_directory: str,
    aligned_videos_dir: str,
    timestamp_prefix: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate output video paths from chunk alignment data

    Args:
        alignment_results: Dictionary of camera groups with alignment data
        alignment_directory: Path to alignment directory (for extracting source name)
        aligned_videos_dir: Directory for aligned videos output
        timestamp_prefix: Optional timestamp prefix (generated if not provided)

    Returns:
        Dictionary mapping camera prefix to output video path

    Example:
        >>> alignment_results = {'cam_1': camera_group_1, 'cam_2': camera_group_2}
        >>> generate_output_paths_from_alignment_data(
        ...     alignment_results, '/path/vid_shot1', 'output/aligned'
        ... )
        {'cam_1': 'output/aligned/20250119_143052_vid_shot1_cam_1.mp4',
         'cam_2': 'output/aligned/20250119_143052_vid_shot1_cam_2.mp4'}
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    source_name = extract_source_name(alignment_directory)
    output_videos = {}

    for camera_prefix, camera_group in alignment_results.items():
        output_path = generate_aligned_video_path(
            aligned_videos_dir, source_name, camera_prefix, timestamp_prefix
        )
        output_videos[camera_prefix] = output_path

        # Set the output_video_path on the CameraGroup object if it has that attribute
        if hasattr(camera_group, 'output_video_path'):
            camera_group.output_video_path = output_path

    return output_videos


def generate_output_paths_from_legacy_data(
    alignment_results: Dict,
    alignment_directory: str,
    aligned_videos_dir: str,
    timestamp_prefix: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate output paths from legacy alignment data

    Args:
        alignment_results: Dictionary of legacy alignment data
        alignment_directory: Path to alignment directory (for extracting source name)
        aligned_videos_dir: Directory for aligned videos output
        timestamp_prefix: Optional timestamp prefix (generated if not provided)

    Returns:
        Dictionary mapping camera prefix to output video path

    Example:
        >>> alignment_results = {'video1.mp4': {'camera_type': 1, ...}}
        >>> generate_output_paths_from_legacy_data(
        ...     alignment_results, '/path/vid_shot1', 'output/aligned'
        ... )
        {'cam_1': 'output/aligned/20250119_143052_vid_shot1_cam_1.mp4'}
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    source_name = extract_source_name(alignment_directory)
    output_videos = {}

    for video_name, data in alignment_results.items():
        camera_type = data.get('camera_type', 1)
        camera_prefix = f"cam_{camera_type}"

        output_path = generate_aligned_video_path(
            aligned_videos_dir, source_name, camera_prefix, timestamp_prefix
        )
        output_videos[camera_prefix] = output_path

    return output_videos
