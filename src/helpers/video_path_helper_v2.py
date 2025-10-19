#!/usr/bin/env python3
"""
Video Path Helper V2
Enhanced utilities for managing video file paths with nested camera group structure

New folder structure support:
    Source Folder/
    ├── Camera Group 1/
    │   ├── chunk1.mp4
    │   └── chunk2.mp4
    ├── Camera Group 2/
    │   └── chunk1.mp4
    └── ...

Example:
    Jennifer - MultiCam Data - Violin and Piano - 2025-07-09/
    ├── 360 Camera 1 - Jennifer - Violin and Piano - 2025-07-09 X4/
    │   └── video chunks...
    └── 360 Camera 2 - Jennifer - Violin and Piano - 2025-07-09 X5 1/
        └── video chunks...
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def sanitize_source_name(source_name: str, max_length: int = 40) -> str:
    """
    Create a cleaner, shorter source name from full folder name

    Args:
        source_name: Full source folder name
        max_length: Maximum length for the sanitized name

    Returns:
        Sanitized source name suitable for filenames

    Example:
        >>> sanitize_source_name("Jennifer - MultiCam Data - Violin and Piano - 2025-07-09")
        'Jennifer_MultiCam_ViolinPiano_20250709'
    """
    # Extract date pattern (YYYY-MM-DD or YYYY-DD-MM)
    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', source_name)
    date_str = date_match.group(0).replace('-', '') if date_match else ''

    # Remove common filler words and clean up
    name = source_name
    if date_match:
        name = name.replace(date_match.group(0), '').strip()

    # Remove common separators and filler words
    name = name.replace(' - ', '_')
    name = re.sub(r'\b(Data|and|the|a|an)\b', '', name, flags=re.IGNORECASE)

    # Clean up multiple spaces and underscores
    name = re.sub(r'[\s]+', '_', name.strip())
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')

    # Add date at the end if found
    if date_str:
        name = f"{name}_{date_str}"

    # Limit length while keeping meaningful parts
    if len(name) > max_length:
        parts = name.split('_')
        # Keep first part (usually name) and last part (usually date)
        if len(parts) > 2:
            name = f"{parts[0]}_{parts[-1]}"
        else:
            name = name[:max_length]

    return name


def extract_camera_group_identifier(camera_group_folder: str) -> str:
    """
    Extract clean camera identifier from camera group folder name

    Args:
        camera_group_folder: Camera group folder name

    Returns:
        Clean camera identifier

    Example:
        >>> extract_camera_group_identifier("360 Camera 1 - Jennifer - Violin and Piano - 2025-07-09 X4")
        'Cam1_X4'
        >>> extract_camera_group_identifier("360 Camera 2 - Jennifer - Violin and Piano - 2025-07-09 X5 1")
        'Cam2_X5'
    """
    # Extract camera number
    cam_match = re.search(r'Camera\s+(\d+)', camera_group_folder, re.IGNORECASE)
    cam_num = cam_match.group(1) if cam_match else '0'

    # Extract device identifier (X4, X5, etc.)
    device_match = re.search(r'X(\d+)', camera_group_folder)
    device_id = f"X{device_match.group(1)}" if device_match else ''

    # Combine
    if device_id:
        return f"Cam{cam_num}_{device_id}"
    else:
        return f"Cam{cam_num}"


def extract_source_name(alignment_directory: str) -> str:
    """
    Extract source name from alignment directory path (V2: supports nested structure)

    Args:
        alignment_directory: Path to the alignment directory (can be source folder or camera group folder)

    Returns:
        Sanitized source name extracted from the path

    Example:
        >>> extract_source_name('/path/to/Jennifer - MultiCam Data - Violin and Piano - 2025-07-09')
        'Jennifer_MultiCam_ViolinPiano_20250709'
    """
    # Get the base folder name
    folder_name = os.path.basename(alignment_directory.rstrip('/'))

    # Check if this is a camera group folder (contains "Camera" or similar)
    if re.search(r'Camera\s+\d+', folder_name, re.IGNORECASE):
        # This is a camera group folder, go up one level to get source
        parent_folder = os.path.dirname(alignment_directory.rstrip('/'))
        folder_name = os.path.basename(parent_folder)

    # Sanitize and return
    return sanitize_source_name(folder_name)


def scan_camera_groups(source_directory: str) -> List[Tuple[str, str]]:
    """
    Scan source directory for camera group folders

    Args:
        source_directory: Path to the source directory containing camera groups

    Returns:
        List of tuples: [(camera_group_folder_name, camera_group_path), ...]

    Example:
        >>> scan_camera_groups('/path/to/Jennifer - MultiCam Data')
        [('360 Camera 1 - Jennifer - Violin and Piano - 2025-07-09 X4', '/path/.../360 Camera 1...'),
         ('360 Camera 2 - Jennifer - Violin and Piano - 2025-07-09 X5 1', '/path/.../360 Camera 2...')]
    """
    camera_groups = []

    if not os.path.exists(source_directory):
        return camera_groups

    # List all subdirectories
    for item in sorted(os.listdir(source_directory)):
        item_path = os.path.join(source_directory, item)

        # Skip hidden files and non-directories
        if item.startswith('.') or not os.path.isdir(item_path):
            continue

        # Check if this looks like a camera group folder
        # (contains "Camera" followed by a number, or just assume all dirs are camera groups)
        camera_groups.append((item, item_path))

    return camera_groups


def generate_aligned_video_path(
    output_dir: str,
    source_name: str,
    camera_identifier: str,
    timestamp_prefix: Optional[str] = None
) -> str:
    """
    Generate output path for aligned video (V2: cleaner names)

    Args:
        output_dir: Directory for aligned videos
        source_name: Source name (already sanitized or will be sanitized)
        camera_identifier: Camera identifier (e.g., 'Cam1_X4')
        timestamp_prefix: Optional timestamp prefix (generated if not provided)

    Returns:
        Full path for the aligned video file

    Example:
        >>> generate_aligned_video_path('output/aligned', 'Jennifer_MultiCam_20250709', 'Cam1_X4')
        'output/aligned/20250119_143052_Jennifer_MultiCam_20250709_Cam1_X4.mp4'
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{timestamp_prefix}_{source_name}_{camera_identifier}.mp4"
    return os.path.join(output_dir, filename)


def generate_detection_video_path(
    output_dir: str,
    source_name: str,
    camera_identifier: str,
    timestamp_prefix: Optional[str] = None,
    output_format: str = 'mp4'
) -> str:
    """
    Generate output path for detection video (V2: cleaner names)

    Args:
        output_dir: Directory for detection videos
        source_name: Source name (already sanitized or will be sanitized)
        camera_identifier: Camera identifier (e.g., 'Cam1_X4')
        timestamp_prefix: Optional timestamp prefix (generated if not provided)
        output_format: Video output format (default: 'mp4')

    Returns:
        Full path for the detection video file

    Example:
        >>> generate_detection_video_path('output/detection', 'Jennifer_MultiCam_20250709', 'Cam1_X4')
        'output/detection/20250119_143052_detection_Jennifer_MultiCam_20250709_Cam1_X4.mp4'
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{timestamp_prefix}_detection_{source_name}_{camera_identifier}.{output_format}"
    return os.path.join(output_dir, filename)


def generate_temp_detection_video_path(
    output_dir: str,
    source_name: str,
    camera_identifier: str,
    timestamp_prefix: Optional[str] = None,
    output_format: str = 'mp4'
) -> str:
    """
    Generate temporary path for detection video (V2: cleaner names)

    Args:
        output_dir: Directory for detection videos
        source_name: Source name (already sanitized or will be sanitized)
        camera_identifier: Camera identifier (e.g., 'Cam1_X4')
        timestamp_prefix: Optional timestamp prefix (generated if not provided)
        output_format: Video output format (default: 'mp4')

    Returns:
        Full path for the temporary detection video file
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    filename = f"{timestamp_prefix}_temp_detection_{source_name}_{camera_identifier}.{output_format}"
    return os.path.join(output_dir, filename)


def generate_unified_video_path(
    output_dir: str,
    output_filename: str,
    timestamp_prefix: Optional[str] = None
) -> str:
    """
    Generate output path for unified video (V2: backward compatible with V1)

    Args:
        output_dir: Directory for unified videos
        output_filename: Output filename (e.g., 'unified_detection_output.mp4')
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


def generate_output_paths_from_camera_groups(
    source_directory: str,
    aligned_videos_dir: str,
    timestamp_prefix: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Generate output video paths for all camera groups in source directory (V2)

    Args:
        source_directory: Path to source directory containing camera groups
        aligned_videos_dir: Directory for aligned videos output
        timestamp_prefix: Optional timestamp prefix (generated if not provided)

    Returns:
        Dictionary mapping camera identifier to info dict with 'output_path' and 'camera_group_folder'

    Example:
        >>> generate_output_paths_from_camera_groups('/path/to/Jennifer - MultiCam Data', 'output/aligned')
        {
            'Cam1_X4': {
                'output_path': 'output/aligned/20250119_143052_Jennifer_MultiCam_20250709_Cam1_X4.mp4',
                'camera_group_folder': '360 Camera 1 - Jennifer - Violin and Piano - 2025-07-09 X4',
                'camera_group_path': '/path/to/.../360 Camera 1...'
            },
            'Cam2_X5': { ... }
        }
    """
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Extract and sanitize source name
    source_name = extract_source_name(source_directory)

    # Scan camera groups
    camera_groups = scan_camera_groups(source_directory)

    # Generate output paths
    output_paths = {}
    for camera_group_folder, camera_group_path in camera_groups:
        # Extract camera identifier
        camera_id = extract_camera_group_identifier(camera_group_folder)

        # Generate output path
        output_path = generate_aligned_video_path(
            aligned_videos_dir, source_name, camera_id, timestamp_prefix
        )

        output_paths[camera_id] = {
            'output_path': output_path,
            'camera_group_folder': camera_group_folder,
            'camera_group_path': camera_group_path,
            'source_name': source_name
        }

    return output_paths


# Backward compatibility aliases for easier switching from v1
def generate_output_paths_from_alignment_data(
    alignment_results: Dict,
    alignment_directory: str,
    aligned_videos_dir: str,
    timestamp_prefix: Optional[str] = None
) -> Dict[str, str]:
    """
    V2 compatibility wrapper for v1 function signature

    Note: In V2, camera identifiers are extracted from folder names,
    not from alignment_results keys
    """
    # Get source directory (parent of alignment_directory if it's a camera group)
    if re.search(r'Camera\s+\d+', os.path.basename(alignment_directory), re.IGNORECASE):
        source_directory = os.path.dirname(alignment_directory.rstrip('/'))
    else:
        source_directory = alignment_directory

    # Generate paths using V2 logic
    camera_group_paths = generate_output_paths_from_camera_groups(
        source_directory, aligned_videos_dir, timestamp_prefix
    )

    # Extract just the output paths for compatibility
    output_videos = {}
    for camera_id, info in camera_group_paths.items():
        output_videos[camera_id] = info['output_path']

        # Set output_video_path on alignment_results if it exists
        if camera_id in alignment_results:
            camera_group = alignment_results[camera_id]
            if hasattr(camera_group, 'output_video_path'):
                camera_group.output_video_path = info['output_path']

    return output_videos


def generate_output_paths_from_legacy_data(
    alignment_results: Dict,
    alignment_directory: str,
    aligned_videos_dir: str,
    timestamp_prefix: Optional[str] = None
) -> Dict[str, str]:
    """
    V2 compatibility wrapper for v1 legacy function signature
    """
    # For legacy data, fall back to V1 behavior but with cleaner names
    if timestamp_prefix is None:
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    source_name = extract_source_name(alignment_directory)
    output_videos = {}

    for video_name, data in alignment_results.items():
        camera_type = data.get('camera_type', 1)
        camera_id = f"Cam{camera_type}"

        output_path = generate_aligned_video_path(
            aligned_videos_dir, source_name, camera_id, timestamp_prefix
        )
        output_videos[camera_id] = output_path

    return output_videos
