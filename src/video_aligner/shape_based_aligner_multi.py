#!/usr/bin/env python3
"""
Shape-based audio alignment for multiple videos
Automatically determines reference video and aligns all others
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import subprocess
import os
import sys
import glob
import yaml
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
from src.database.setup import VideoAlignmentDatabase, ChunkVideoAlignmentDatabase

@dataclass
class ChunkVideo:
    """Represents a single video chunk"""
    filename: str
    filepath: str
    prefix: str
    chunk_number: int
    duration: float
    start_time_offset: float = 0.0  # Offset relative to reference video
    energy_profile: Optional[np.ndarray] = None
    time_axis: Optional[np.ndarray] = None

@dataclass
class CameraGroup:
    """Represents all chunks for a single camera"""
    prefix: str
    chunks: List[ChunkVideo]
    total_duration: float
    earliest_start: float  # Absolute start time relative to reference
    output_video_path: str = ""

def extract_audio_with_ffmpeg(video_path, output_path, duration=None, start_time=0):
    """Extract audio using ffmpeg directly
    
    Args:
        video_path: Path to video file
        output_path: Path to save audio file
        duration: Duration to extract (None for full video)
        start_time: Start time in seconds
    """
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_path,
    ]
    
    # Only add duration limit if specified
    if duration is not None:
        cmd.extend(['-t', str(duration)])
    
    cmd.extend([
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', '22050',  # Sample rate
        '-ac', '1',  # Mono
        output_path
    ])
    
    print(f"Extracting audio from {os.path.basename(video_path)}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None, None
    
    # Read the WAV file
    import wave
    with wave.open(output_path, 'rb') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        sample_rate = wav.getframerate()
    
    return audio, sample_rate

def extract_prefix_and_chunk_number(filename: str) -> Tuple[str, int]:
    """
    Extract camera prefix and chunk number from filename

    Examples:
        cam_1_1.mp4 -> ("cam_1", 1)      # Multi-chunk: cam_1 camera, chunk 1
        cam_1_2.mp4 -> ("cam_1", 2)      # Multi-chunk: cam_1 camera, chunk 2
        cam_1.mp4 -> ("cam_1", 0)        # Single chunk: cam_1 camera, chunk 0
        cam_2.mp4 -> ("cam_2", 0)        # Single chunk: cam_2 camera, chunk 0
        camera_1_2.mp4 -> ("camera_1", 2) # Multi-chunk: camera_1, chunk 2

    Args:
        filename: Video filename

    Returns:
        Tuple of (prefix, chunk_number)
    """
    # Remove file extension
    basename = os.path.splitext(filename)[0]

    # Split by underscore to find parts
    parts = basename.split('_')

    if len(parts) == 1:
        # No underscores, treat as single chunk
        return basename, 0

    elif len(parts) == 2:
        # Two parts: could be cam_1 (single chunk) or other_chunk format
        if parts[0] in ['cam', 'camera'] and parts[1].isdigit():
            # This is cam_1.mp4 format - single chunk camera
            return basename, 0  # Full name as prefix, chunk 0
        else:
            # Try to determine if last part is chunk number
            if parts[1].isdigit():
                # Could be chunk number, but be conservative
                # For cam_1 format, treat as single chunk
                return basename, 0
            else:
                # Not a number, treat whole thing as prefix
                return basename, 0

    else:
        # Three or more parts: likely cam_1_2 format (camera_chunk)
        # Work backwards to find the last number as chunk number
        chunk_number = 0

        # Check if last part is a number (potential chunk number)
        if parts[-1].isdigit():
            chunk_number = int(parts[-1])
            prefix_parts = parts[:-1]  # Everything except last part
        else:
            # No chunk number found, treat as single chunk
            prefix_parts = parts
            chunk_number = 0

        prefix = '_'.join(prefix_parts)
        return prefix, chunk_number

def scan_and_group_chunk_videos(video_dir: str) -> Dict[str, CameraGroup]:
    """
    Scan directory and group videos by camera prefix

    Args:
        video_dir: Directory containing video files

    Returns:
        Dictionary mapping camera prefix to CameraGroup
    """
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    camera_groups = defaultdict(list)

    print(f"\nScanning {len(video_files)} video files for chunk grouping...")

    for video_path in video_files:
        filename = os.path.basename(video_path)
        prefix, chunk_number = extract_prefix_and_chunk_number(filename)

        # Get video duration
        duration = get_video_duration(video_path)
        if duration is None:
            print(f"Warning: Could not get duration for {filename}, skipping")
            continue

        chunk = ChunkVideo(
            filename=filename,
            filepath=video_path,
            prefix=prefix,
            chunk_number=chunk_number,
            duration=duration
        )

        camera_groups[prefix].append(chunk)
        print(f"  {filename} -> prefix: {prefix}, chunk: {chunk_number}, duration: {duration:.1f}s")

    # Sort chunks within each camera group by chunk number
    result = {}
    for prefix, chunks in camera_groups.items():
        sorted_chunks = sorted(chunks, key=lambda x: x.chunk_number)
        total_duration = sum(chunk.duration for chunk in sorted_chunks)

        result[prefix] = CameraGroup(
            prefix=prefix,
            chunks=sorted_chunks,
            total_duration=total_duration,
            earliest_start=0.0  # Will be calculated later
        )

        print(f"\nCamera group '{prefix}': {len(sorted_chunks)} chunks, total: {total_duration:.1f}s")
        for chunk in sorted_chunks:
            print(f"  - {chunk.filename} (chunk {chunk.chunk_number}): {chunk.duration:.1f}s")

    return result

def calculate_intra_camera_gaps(camera_groups: Dict[str, CameraGroup]) -> Dict[str, CameraGroup]:
    """
    Calculate gaps between chunks within each camera group based on their absolute timeline positions

    This function determines gaps by comparing the timeline positions that were calculated
    by audio pattern matching against the reference.

    Args:
        camera_groups: Dictionary of camera groups with chunks that have absolute timeline positions

    Returns:
        Updated camera groups with gap information
    """
    print(f"\nCalculating intra-camera gaps based on timeline positions...")

    for prefix, group in camera_groups.items():
        if len(group.chunks) <= 1:
            print(f"\nCamera '{prefix}': Single or no chunks, no gaps to calculate")
            continue

        print(f"\nCamera '{prefix}': Calculating gaps between {len(group.chunks)} chunks")

        # Sort chunks by their timeline position
        sorted_chunks = sorted(group.chunks, key=lambda x: x.start_time_offset)

        for i in range(len(sorted_chunks)):
            chunk = sorted_chunks[i]

            if i == 0:
                # First chunk in timeline order
                print(f"  {chunk.filename}: timeline position {chunk.start_time_offset:.3f}s (first)")
                continue

            prev_chunk = sorted_chunks[i-1]
            prev_end_time = prev_chunk.start_time_offset + prev_chunk.duration
            current_start_time = chunk.start_time_offset

            # Calculate gap between previous chunk end and current chunk start
            gap = current_start_time - prev_end_time

            if gap > 1.0:  # Significant gap (> 1 second)
                print(f"  {chunk.filename}: timeline position {current_start_time:.3f}s (gap: {gap:.3f}s from {prev_chunk.filename})")
            elif gap < -1.0:  # Overlapping content
                print(f"  {chunk.filename}: timeline position {current_start_time:.3f}s (overlap: {abs(gap):.3f}s with {prev_chunk.filename})")
            else:
                print(f"  {chunk.filename}: timeline position {current_start_time:.3f}s (continuous from {prev_chunk.filename})")

        # Update group earliest start to the first chunk's position
        group.earliest_start = min(chunk.start_time_offset for chunk in group.chunks)

    return camera_groups

def determine_reference_camera_group_by_audio_pattern(camera_groups: Dict[str, CameraGroup],
                                                   use_earliest_start: bool = True) -> str:
    """
    Determine which camera group should be the reference by analyzing audio patterns
    of first chunks using the EXACT same logic as legacy alignment

    Args:
        camera_groups: Dictionary of camera groups
        use_earliest_start: Use earliest start strategy (True) or latest start (False)

    Returns:
        Prefix of the reference camera group
    """
    if len(camera_groups) == 1:
        return list(camera_groups.keys())[0]

    print(f"\nDetermining reference camera group using LEGACY alignment logic...")
    print(f"Strategy: {'earliest_start' if use_earliest_start else 'latest_start'}")

    # Create temp directory for audio extraction
    os.makedirs('temp_audio_ref', exist_ok=True)

    try:
        # Step 1: Extract audio from first chunk of each camera group (EXACTLY like legacy)
        first_chunk_profiles = {}

        for prefix, group in camera_groups.items():
            # Get the first chunk (chunk number 0 or 1, or single chunk)
            first_chunk = group.chunks[0]  # Already sorted by chunk number

            print(f"  Analyzing first chunk: {first_chunk.filename} from {prefix}")

            # Extract audio from first chunk - use FULL audio like legacy does
            audio_path = f'temp_audio_ref/{first_chunk.filename}_audio.wav'
            audio, sr = extract_audio_with_ffmpeg(first_chunk.filepath, audio_path, duration=None, start_time=0)

            if audio is None:
                print(f"    ❌ Failed to extract audio from {first_chunk.filename}")
                continue

            # Calculate energy profile for pattern analysis (EXACTLY same as legacy)
            energy, time_axis = calculate_energy_profile(audio, sr, window_ms=100)
            first_chunk_profiles[first_chunk.filename] = (energy, time_axis)  # Use filename as key

            print(f"    ✅ Extracted {len(audio)/sr:.1f}s of audio for pattern analysis")

            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

        if len(first_chunk_profiles) < 2:
            print("  ⚠️ Not enough audio profiles for comparison, using first available camera")
            return list(camera_groups.keys())[0]

        # Step 2: Use EXACT legacy determine_reference_video logic
        reference_filename, offsets = determine_reference_video(first_chunk_profiles, use_earliest_start)

        # Step 3: Map the reference filename back to camera prefix
        reference_prefix = None
        for prefix, group in camera_groups.items():
            first_chunk = group.chunks[0]
            if first_chunk.filename == reference_filename:
                reference_prefix = prefix
                break

        if reference_prefix is None:
            print("  ⚠️ Could not map reference filename to camera prefix, using first available")
            return list(camera_groups.keys())[0]

        print(f"\n  Reference selection results:")
        for filename, offset in sorted(offsets.items(), key=lambda x: x[1]):
            # Find corresponding camera prefix
            camera_prefix = None
            for prefix, group in camera_groups.items():
                if group.chunks[0].filename == filename:
                    camera_prefix = prefix
                    break

            marker = " (SELECTED)" if filename == reference_filename else ""
            print(f"    {camera_prefix} ({filename}): {offset:.3f}s offset{marker}")

        strategy_desc = "earliest start (most content)" if use_earliest_start else "latest start (most concise)"
        print(f"  -> Choosing '{reference_prefix}' as reference ({strategy_desc})")
        return reference_prefix

    finally:
        # Clean up temp directory
        if os.path.exists('temp_audio_ref'):
            try:
                os.rmdir('temp_audio_ref')
            except:
                pass

def align_chunks_to_reference_timeline(camera_groups: Dict[str, CameraGroup],
                                      reference_prefix: str,
                                      use_earliest_start: bool = True) -> Dict[str, CameraGroup]:
    """
    Align chunks using LEGACY algorithm - treat first chunks like full videos and align them,
    then apply same offsets to remaining chunks in each camera group

    Args:
        camera_groups: Dictionary of camera groups
        reference_prefix: Prefix of reference camera
        use_earliest_start: Strategy for reference selection

    Returns:
        Updated camera groups with absolute timeline positions for each chunk
    """
    print(f"\nPerforming chunk alignment using LEGACY logic...")
    print(f"Reference camera: {reference_prefix}")

    # Create temp directory for audio extraction
    os.makedirs('temp_audio', exist_ok=True)

    try:
        # Step 1: Extract FULL audio from first chunk of each camera group (like legacy does)
        print(f"\nExtracting audio from first chunks (like legacy full video processing)...")
        first_chunk_profiles = {}

        for prefix, group in camera_groups.items():
            first_chunk = group.chunks[0]  # First chunk of each camera group

            print(f"  Extracting full audio from {first_chunk.filename} ({prefix})...")

            audio_path = f'temp_audio/{first_chunk.filename}_audio.wav'
            audio, sr = extract_audio_with_ffmpeg(first_chunk.filepath, audio_path, duration=None, start_time=0)

            if audio is None:
                print(f"    ❌ Failed to extract audio from {first_chunk.filename}")
                continue

            # Calculate energy profile (EXACTLY like legacy)
            energy, time_axis = calculate_energy_profile(audio, sr, window_ms=100)
            first_chunk_profiles[first_chunk.filename] = (energy, time_axis)

            print(f"    ✅ Processed {first_chunk.filename}: {len(audio)/sr:.1f}s at {sr}Hz")

            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

        if len(first_chunk_profiles) < 2:
            print("❌ Need at least 2 camera groups to perform alignment")
            return camera_groups

        # Step 2: Use EXACT legacy logic to determine reference and calculate offsets
        print(f"\nUsing LEGACY determine_reference_video logic...")
        reference_filename, legacy_offsets = determine_reference_video(first_chunk_profiles, use_earliest_start)

        # Step 3: Map legacy offsets to camera groups and apply to all chunks
        print(f"\nApplying legacy offsets to all chunks in each camera group...")

        for prefix, group in camera_groups.items():
            first_chunk = group.chunks[0]

            # Find the legacy offset for this camera group's first chunk
            camera_offset = legacy_offsets.get(first_chunk.filename, 0.0)

            print(f"\nCamera group '{prefix}': applying offset {camera_offset:.3f}s to all chunks")

            # Apply the SAME offset to ALL chunks in this camera group
            for i, chunk in enumerate(group.chunks):
                # For chunk processing, each chunk starts at: camera_offset + (sum of previous chunk durations)
                chunk_start_within_camera = sum(group.chunks[j].duration for j in range(i))
                chunk.start_time_offset = camera_offset + chunk_start_within_camera

                if chunk == first_chunk:
                    marker = " (first chunk - from legacy alignment)"
                else:
                    marker = f" (chunk {chunk.chunk_number} - offset from camera start)"

                print(f"  {chunk.filename}: {chunk.start_time_offset:.3f}s{marker}")

        # Step 4: Verification - show final timeline
        print(f"\nFinal timeline verification:")
        print(f"Reference camera: {reference_prefix}")

        all_chunks = []
        for prefix, group in camera_groups.items():
            for chunk in group.chunks:
                all_chunks.append((chunk.filename, chunk.start_time_offset, prefix))

        # Sort by timeline position
        all_chunks.sort(key=lambda x: x[1])

        print(f"Timeline order:")
        for filename, offset, prefix in all_chunks:
            marker = " (REFERENCE)" if prefix == reference_prefix and offset == 0.0 else ""
            print(f"  {filename} ({prefix}): {offset:.3f}s{marker}")

    finally:
        # Clean up temp directory
        if os.path.exists('temp_audio'):
            try:
                os.rmdir('temp_audio')
            except:
                pass

    return camera_groups

def get_absolute_chunk_timeline(camera_groups: Dict[str, CameraGroup]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Get absolute timeline positions for all chunks (already calculated by pattern matching)

    Args:
        camera_groups: Dictionary of camera groups with chunks positioned on absolute timeline

    Returns:
        Dictionary mapping camera prefix to list of (filename, absolute_offset) tuples
    """
    print(f"\nCollecting absolute timeline positions...")

    result = {}

    for prefix, group in camera_groups.items():
        chunk_positions = []

        for chunk in group.chunks:
            # Use the absolute timeline position calculated by pattern matching
            absolute_position = chunk.start_time_offset
            chunk_positions.append((chunk.filename, absolute_position))

            print(f"  {chunk.filename}: {absolute_position:.3f}s (absolute timeline position)")

        result[prefix] = chunk_positions

    return result

def generate_output_video_paths(camera_groups: Dict[str, CameraGroup], config: dict) -> Dict[str, CameraGroup]:
    """
    Generate output video paths for each camera group based on config

    Args:
        camera_groups: Dictionary of camera groups
        config: Configuration dictionary

    Returns:
        Updated camera groups with output video paths
    """
    # Get base output path from config and make it absolute
    base_output_path = config.get('video_aligner', {}).get('output_video_path', 'output/original_video.mp4')

    # Convert to absolute path if it's relative
    if not os.path.isabs(base_output_path):
        base_output_path = os.path.abspath(base_output_path)

    base_dir = os.path.dirname(base_output_path)
    base_filename = os.path.splitext(os.path.basename(base_output_path))[0]

    print(f"\nGenerating output video paths...")
    print(f"Base output directory: {base_dir}")

    for prefix, group in camera_groups.items():
        # Generate output path: original_cam_1_sk_0.mp4
        output_filename = f"original_{prefix}_sk_0.mp4"
        output_path = os.path.join(base_dir, output_filename)
        group.output_video_path = output_path

        print(f"  Camera '{prefix}' -> {output_path}")

    return camera_groups

def combine_chunk_videos_timeline_based(camera_groups: Dict[str, CameraGroup]) -> Dict[str, str]:
    """
    Combine chunk videos based on their absolute timeline positions with proper gap handling

    Args:
        camera_groups: Dictionary of camera groups with timeline-positioned chunks

    Returns:
        Dictionary mapping camera prefix to output video path
    """
    print(f"\nCombining chunk videos based on timeline positions...")

    result = {}

    for prefix, group in camera_groups.items():
        if len(group.chunks) == 1:
            # Single chunk, copy to output location
            chunk = group.chunks[0]
            print(f"  Camera '{prefix}': Single chunk, copying {chunk.filename} to output location")

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(group.output_video_path)
            os.makedirs(output_dir, exist_ok=True)

            # Copy the single chunk to the output location
            try:
                shutil.copy2(chunk.filepath, group.output_video_path)
                print(f"    ✅ Copied {chunk.filename} to {group.output_video_path}")
                result[prefix] = group.output_video_path
            except Exception as e:
                print(f"    ❌ Failed to copy {chunk.filename}: {e}")
                result[prefix] = chunk.filepath  # Fallback to original path
            continue

        print(f"  Camera '{prefix}': Combining {len(group.chunks)} chunks based on timeline...")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(group.output_video_path)
        os.makedirs(output_dir, exist_ok=True)

        # Sort chunks by timeline position for proper ordering
        sorted_chunks = sorted(group.chunks, key=lambda x: x.start_time_offset)

        # Simplified approach: Use ffmpeg concat demuxer for multiple chunks
        if len(sorted_chunks) > 1:
            # For multiple chunks, create a concat file and use concat demuxer
            concat_file_path = os.path.join(os.path.dirname(group.output_video_path), f"{prefix}_concat.txt")

            try:
                # Create concat file
                with open(concat_file_path, 'w') as f:
                    for chunk in sorted_chunks:
                        f.write(f"file '{chunk.filepath}'\n")

                # Simple concat command
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file_path,
                    '-c', 'copy',  # Copy streams without re-encoding
                    group.output_video_path
                ]

                print(f"    Executing simple concat for {len(sorted_chunks)} chunks...")
                print(f"    Command: {' '.join(cmd[:8])}...")

                result_proc = subprocess.run(cmd, capture_output=True, text=True)

                # Clean up concat file
                if os.path.exists(concat_file_path):
                    os.remove(concat_file_path)

                if result_proc.returncode == 0:
                    print(f"    ✅ Combined video saved to {group.output_video_path}")
                    result[prefix] = group.output_video_path
                else:
                    print(f"    ❌ Failed to combine videos: {result_proc.stderr}")
                    print(f"    Falling back to first chunk...")
                    result[prefix] = sorted_chunks[0].filepath

                continue

            except Exception as e:
                print(f"    ❌ Error creating concat file: {e}")
                print(f"    Falling back to first chunk...")
                result[prefix] = sorted_chunks[0].filepath
                continue

        else:
            # This shouldn't happen as we handle single chunks above
            result[prefix] = sorted_chunks[0].filepath
            continue


    return result

def calculate_energy_profile(audio, sample_rate, window_ms=100):
    """
    Calculate RMS energy profile with specified window size
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        window_ms: Window size in milliseconds
    
    Returns:
        energy_profile: RMS energy values
        time_axis: Time axis in seconds
    """
    window_size = int(sample_rate * window_ms / 1000)
    hop_size = window_size // 2  # 50% overlap
    
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        rms = np.sqrt(np.mean(window**2))
        energy.append(rms)
    
    energy = np.array(energy)
    time_axis = np.arange(len(energy)) * (hop_size / sample_rate)
    
    return energy, time_axis

def normalize_shape(profile):
    """Normalize profile to focus on shape rather than amplitude"""
    # Remove DC component
    profile = profile - np.mean(profile)
    # Normalize to unit variance
    std = np.std(profile)
    if std > 0:
        profile = profile / std
    return profile

def smooth_profile(profile, window_length=11):
    """Apply Savitzky-Golay filter to smooth the profile"""
    if len(profile) > window_length:
        return savgol_filter(profile, window_length, 3)
    return profile

def find_shape_based_offset(energy_ref, energy_target, time_axis_ref, time_axis_target, max_offset=30):
    """
    Find offset based on energy profile shape matching
    Returns: how much to shift target to align with reference
    
    Args:
        energy_ref: Energy profile of reference
        energy_target: Energy profile to align
        time_axis_ref: Time axis for energy_ref
        time_axis_target: Time axis for energy_target (currently unused)
        max_offset: Maximum offset to search in seconds
    
    Returns:
        best_offset: Offset in seconds (positive = target starts later, negative = target starts earlier)
        correlation_scores: Array of correlation scores
        offset_range: Array of tested offsets
    """
    
    # Smooth and normalize profiles
    energy_ref_smooth = smooth_profile(energy_ref)
    energy_target_smooth = smooth_profile(energy_target)
    
    energy_ref_norm = normalize_shape(energy_ref_smooth)
    energy_target_norm = normalize_shape(energy_target_smooth)
    
    # Time step (assuming uniform spacing)
    dt = time_axis_ref[1] - time_axis_ref[0]
    max_offset_samples = int(max_offset / dt)
    
    correlation_scores = []
    offset_range = []
    
    # Search range: negative offsets (target starts earlier) to positive offsets (target starts later)
    for offset_samples in range(-max_offset_samples, max_offset_samples + 1):
        offset_seconds = offset_samples * dt
        
        if offset_samples < 0:
            # Target starts earlier: compare ref[0:] with target[|offset|:]
            target_start = abs(offset_samples)
            if target_start < len(energy_target_norm):
                ref_segment = energy_ref_norm
                target_segment = energy_target_norm[target_start:]
                
                # Use shorter length
                min_len = min(len(ref_segment), len(target_segment))
                if min_len > 0:
                    ref_segment = ref_segment[:min_len]
                    target_segment = target_segment[:min_len]
                    
                    if len(ref_segment) > 10:  # Ensure meaningful comparison
                        corr = np.corrcoef(ref_segment, target_segment)[0, 1]
                        if not np.isnan(corr):
                            correlation_scores.append(corr)
                            offset_range.append(offset_seconds)
        
        elif offset_samples > 0:
            # Target starts later: compare ref[offset:] with target[0:]
            ref_start = offset_samples
            if ref_start < len(energy_ref_norm):
                ref_segment = energy_ref_norm[ref_start:]
                target_segment = energy_target_norm
                
                # Use shorter length
                min_len = min(len(ref_segment), len(target_segment))
                if min_len > 0:
                    ref_segment = ref_segment[:min_len]
                    target_segment = target_segment[:min_len]
                    
                    if len(ref_segment) > 10:  # Ensure meaningful comparison
                        corr = np.corrcoef(ref_segment, target_segment)[0, 1]
                        if not np.isnan(corr):
                            correlation_scores.append(corr)
                            offset_range.append(offset_seconds)
        
        else:  # offset_samples == 0
            # No offset: direct comparison
            min_len = min(len(energy_ref_norm), len(energy_target_norm))
            if min_len > 10:
                ref_segment = energy_ref_norm[:min_len]
                target_segment = energy_target_norm[:min_len]
                corr = np.corrcoef(ref_segment, target_segment)[0, 1]
                if not np.isnan(corr):
                    correlation_scores.append(corr)
                    offset_range.append(0.0)
    
    correlation_scores = np.array(correlation_scores)
    offset_range = np.array(offset_range)
    
    if len(correlation_scores) == 0:
        return 0.0, np.array([]), np.array([])
    
    # Find peak correlation
    best_idx = np.argmax(correlation_scores)
    best_offset = offset_range[best_idx]
    
    return best_offset, correlation_scores, offset_range

def determine_reference_video(video_profiles, use_earliest_start=False):
    """
    Determine which video should be the reference based on matching pattern timing

    Args:
        video_profiles: Dictionary with video names as keys and (energy, time_axis) as values
        use_earliest_start: If True, choose video with earliest start (most content) as reference.
                           If False, choose video with latest start (most concise content) as reference.

    Returns:
        reference_video: Name of the video to use as reference
        offsets: Dictionary of initial offsets for each video
    """
    videos = list(video_profiles.keys())
    n_videos = len(videos)
    
    if n_videos == 1:
        return videos[0], {videos[0]: 0.0}
    
    print("Analyzing pairwise alignments to find reference video...")
    
    # Calculate pairwise offsets
    offset_matrix = np.zeros((n_videos, n_videos))
    
    for i in range(n_videos):
        for j in range(n_videos):
            if i != j:
                energy_i, time_i = video_profiles[videos[i]]
                energy_j, time_j = video_profiles[videos[j]]
                
                # Find offset from i to j (how much to shift j to align with i)
                offset, _, _ = find_shape_based_offset(energy_i, energy_j, time_i, time_j)
                offset_matrix[i, j] = offset
                
                print(f"  {videos[i]} -> {videos[j]}: {offset:.3f}s")
    
    # Find reference video based on the chosen strategy
    avg_offsets_from_video = np.mean(offset_matrix, axis=1)  # Average of each row

    if use_earliest_start:
        # Choose video with earliest start (highest average offset - most content)
        reference_idx = np.argmax(avg_offsets_from_video)
        strategy_desc = "highest average - earliest start (most content)"
    else:
        # Choose video with latest start (lowest average offset - most concise content)
        reference_idx = np.argmin(avg_offsets_from_video)
        strategy_desc = "lowest average - latest start (most concise content)"

    reference_video = videos[reference_idx]

    print(f"\nAverage offsets FROM each video:")
    for i, video in enumerate(videos):
        print(f"  {video} -> others: {avg_offsets_from_video[i]:.3f}s")
    print(f"  -> Choosing {reference_video} ({strategy_desc})")
    print(f"  -> Strategy: {'Earliest start' if use_earliest_start else 'Latest start'} as reference")
    
    # Calculate offsets relative to reference
    # All offsets should be positive (other videos need to skip beginning to align)
    offsets = {}
    for i, video in enumerate(videos):
        if i == reference_idx:
            offsets[video] = 0.0
        else:
            # Use the offset from reference to this video
            raw_offset = offset_matrix[reference_idx, i]
            # Ensure positive offsets by taking absolute value if needed
            offsets[video] = abs(raw_offset)
    
    return reference_video, offsets

def calculate_matching_duration(video_profiles, offsets, min_overlap_duration=30):
    """
    Calculate the duration where all videos have matching content
    
    Args:
        video_profiles: Dictionary with video names as keys and (energy, time_axis) as values
        offsets: Dictionary of start offsets for each video
        min_overlap_duration: Minimum duration to consider for matching
    
    Returns:
        matching_duration: Duration in seconds where all videos overlap with similar content
    """
    print("DEBUG: Calculating matching duration...")
    
    # Get video durations from energy profiles (full audio)
    profile_durations = {}
    for video, (_, time_axis) in video_profiles.items():
        profile_durations[video] = time_axis[-1]
        print(f"  {video} profile duration: {profile_durations[video]:.1f}s (from full audio)")
    
    # Since we're now extracting full audio, profile durations ARE the actual durations
    actual_durations = profile_durations.copy()
    print("\nActual video durations (from full audio extraction):")
    for video, duration in actual_durations.items():
        print(f"  {video}: {duration:.1f}s ({duration/60:.1f} minutes)")
    
    # Calculate remaining content duration for each video after applying offsets
    remaining_durations = {}
    print("\nDEBUG: Remaining content calculation after alignment:")
    for video in video_profiles:
        offset = offsets[video]
        total_duration = actual_durations[video]
        # After skipping 'offset' seconds, how much content remains?
        remaining_content = total_duration - offset
        remaining_durations[video] = remaining_content
        print(f"  {video}: total={total_duration:.1f}s, offset={offset:.1f}s, remaining_content={remaining_content:.1f}s")
    
    # The matching duration is the minimum remaining content (shortest video after alignment)
    # This represents the length of identical content across all videos
    matching_duration = min(remaining_durations.values())
    
    print(f"\nDEBUG: Matching duration calculation:")
    print(f"  Common content duration: {matching_duration:.1f}s")
    print(f"  All videos will have exactly {matching_duration:.1f}s of matching content:")
    for video in video_profiles:
        print(f"    {video}: matching_duration = {matching_duration:.1f}s")
    
    print(f"  Raw matching duration: {matching_duration:.1f}s")
    
    # Ensure minimum duration
    if matching_duration < min_overlap_duration:
        print(f"Warning: Matching duration ({matching_duration:.1f}s) is less than minimum ({min_overlap_duration}s)")
    
    final_duration = max(0, matching_duration)
    print(f"  Final matching duration: {final_duration:.1f}s ({final_duration/60:.1f} minutes)")
    
    return final_duration

def plot_energy_profiles_overlay(video_profiles, offsets, matching_duration, output_path):
    """
    Create overlay plot of all energy profiles before and after alignment
    
    Args:
        video_profiles: Dictionary with video names as keys and (energy, time_axis) as values
        offsets: Dictionary of offsets for each video
        matching_duration: Duration of matching content
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    colors = plt.cm.Set1(np.linspace(0, 1, len(video_profiles)))
    
    # Plot 1: Original energy profiles
    for (video, (energy, time_axis)), color in zip(video_profiles.items(), colors):
        axes[0].plot(time_axis, energy, alpha=0.7, label=video, color=color)
    
    axes[0].set_title('Original Energy Profiles')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Energy (RMS)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 60)
    
    # Plot 2: Aligned energy profiles
    for (video, (energy, time_axis)), color in zip(video_profiles.items(), colors):
        aligned_time = time_axis - offsets[video]
        axes[1].plot(aligned_time, energy, alpha=0.7, label=f'{video} (offset: {offsets[video]:.2f}s)', color=color)
    
    # Mark matching duration
    axes[1].axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Start of matching')
    axes[1].axvline(x=matching_duration, color='red', linestyle='--', alpha=0.5, label=f'End of matching ({matching_duration:.1f}s)')
    axes[1].axvspan(0, matching_duration, alpha=0.1, color='green')
    
    axes[1].set_title('Aligned Energy Profiles')
    axes[1].set_xlabel('Aligned Time (seconds)')
    axes[1].set_ylabel('Energy (RMS)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-10, matching_duration + 10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    return fig

def get_video_duration(video_path):
    """Get video duration in seconds"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return float(result.stdout.strip())
    return None

def load_config(config_path='src/config/config_v1.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file (relative to parent directory)
        
    Returns:
        Dictionary with configuration settings
    """
    # Look for config in parent directory
    config_full_path = os.path.join(parent_dir, config_path)
    
    try:
        with open(config_full_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✅ Configuration loaded from {config_full_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️ Config file {config_full_path} not found")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def main():
    # Try to load config first
    config = load_config()
    
    # Parse command line arguments or use config
    if len(sys.argv) == 2:
        # Command line argument provided
        video_dir = sys.argv[1]
        print(f"Using command line argument: {video_dir}")
    elif config and 'video_aligner' in config and 'alignment_directory' in config['video_aligner']:
        # Use config file
        video_dir = config['video_aligner']['alignment_directory']
        print(f"Using config file directory: {video_dir}")
    else:
        print("Usage: python shape_based_aligner_multi.py <video_directory>")
        print("   OR: Configure video_aligner.alignment_directory in src/config/config_v1.yaml")
        print("Example: python shape_based_aligner_multi.py vid_shot1")
        print("  - video_directory: Directory containing videos to align")
        print("\nThe script will:")
        print("  1. Process all MP4 files in the directory")
        print("  2. Automatically determine the reference video (started recording first)")
        print("  3. Calculate positive alignment offsets and matching duration")
        print("  4. Store results in the database")
        return 1
    
    # Check if directory exists
    if not os.path.exists(video_dir):
        print(f"❌ Directory not found: {video_dir}")
        return 1
    
    # Check if chunk processing is enabled and get video aligner config
    enable_chunk_processing = False
    save_output_videos = True

    if config and 'video_aligner' in config:
        video_aligner_config = config['video_aligner']
        enable_chunk_processing = video_aligner_config.get('alignment', {}).get('enable_chunk_processing', False)
        save_output_videos = video_aligner_config.get('save_output_videos', True)

    print("="*60)
    print("MULTI-VIDEO CHUNK-AWARE ALIGNMENT")
    print("="*60)
    print(f"Chunk processing: {'Enabled' if enable_chunk_processing else 'Disabled'}")
    print(f"Save output videos: {'Enabled' if save_output_videos else 'Disabled'}")

    # Initialize database connection based on processing mode
    try:
        if enable_chunk_processing:
            db = ChunkVideoAlignmentDatabase()
            print("\n✅ Connected to chunk video alignment database")
        else:
            db = VideoAlignmentDatabase()
            print("\n✅ Connected to legacy video alignment database")
    except Exception as e:
        print(f"\n❌ Database connection failed: {e}")
        print("Continuing without database storage...")
        db = None

    if enable_chunk_processing:
        # New chunk processing workflow
        print("\n" + "="*60)
        print("SCANNING AND GROUPING CHUNK VIDEOS")
        print("="*60)

        # Step 1: Scan and group videos by camera prefix
        camera_groups = scan_and_group_chunk_videos(video_dir)

        if len(camera_groups) == 0:
            print("❌ No video files found")
            return 1

        # Step 2: Determine reference camera group using legacy logic with config strategy
        print("\n" + "="*60)
        print("DETERMINING REFERENCE CAMERA GROUP")
        print("="*60)

        # Get reference selection strategy from config or default
        use_earliest_start = False  # Default to latest start (current logic)
        method_type = 'latest_start'  # Default method type
        if config and 'video_aligner' in config and 'alignment' in config['video_aligner']:
            alignment_config = config['video_aligner']['alignment']
            if 'reference_strategy' in alignment_config:
                strategy = alignment_config['reference_strategy']
                if strategy == 'earliest_start':
                    use_earliest_start = True
                    method_type = 'earliest_start'
                elif strategy == 'latest_start':
                    use_earliest_start = False
                    method_type = 'latest_start'
                else:
                    print(f"Warning: Unknown reference_strategy '{strategy}', using default (latest_start)")

        reference_prefix = determine_reference_camera_group_by_audio_pattern(camera_groups, use_earliest_start)

        # Step 3: Align all chunks using legacy logic
        print("\n" + "="*60)
        print("CHUNK ALIGNMENT USING LEGACY LOGIC")
        print("="*60)

        camera_groups = align_chunks_to_reference_timeline(camera_groups, reference_prefix, use_earliest_start)

        # Step 4: Calculate intra-camera gaps based on timeline positions
        print("\n" + "="*60)
        print("CALCULATING INTRA-CAMERA GAPS")
        print("="*60)

        camera_groups = calculate_intra_camera_gaps(camera_groups)

        # Step 5: Get absolute timeline positions for all chunks
        print("\n" + "="*60)
        print("ABSOLUTE TIMELINE POSITIONS")
        print("="*60)

        final_chunk_offsets = get_absolute_chunk_timeline(camera_groups)

        # Step 6: Generate output video paths
        print("\n" + "="*60)
        print("GENERATING OUTPUT PATHS")
        print("="*60)

        camera_groups = generate_output_video_paths(camera_groups, config)

        # Step 7: Combine chunk videos using timeline-based approach (if enabled)
        if save_output_videos:
            print("\n" + "="*60)
            print("TIMELINE-BASED VIDEO COMBINATION")
            print("="*60)

            combined_videos = combine_chunk_videos_timeline_based(camera_groups)
        else:
            print("\n" + "="*60)
            print("SKIPPING VIDEO COMBINATION")
            print("="*60)
            print("Save output videos is disabled - no video files will be created")

    else:
        # Legacy single video processing workflow
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        if len(video_files) == 0:
            print(f"❌ No MP4 files found in {video_dir}")
            return 1

        print(f"Found {len(video_files)} videos in {video_dir}:")
        for video in video_files:
            print(f"  - {os.path.basename(video)}")

        # Create temp directory for audio extraction
        os.makedirs('temp_audio', exist_ok=True)

        # Extract audio and calculate energy profiles for all videos
        print("\n" + "="*60)
        print("EXTRACTING AUDIO AND CALCULATING ENERGY PROFILES")
        print("="*60)

        video_profiles = {}
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            audio_path = f'temp_audio/{os.path.splitext(video_name)[0]}_audio.wav'

            # Extract full audio for analysis
            print(f"Extracting full audio from {video_name}...")
            audio, sr = extract_audio_with_ffmpeg(video_path, audio_path, duration=None, start_time=0)

            if audio is None:
                print(f"❌ Failed to extract audio from {video_name}")
                continue

            # Calculate energy profile
            energy, time_axis = calculate_energy_profile(audio, sr, window_ms=100)
            video_profiles[video_name] = (energy, time_axis)

            print(f"✅ Processed {video_name}: {len(audio)/sr:.1f}s at {sr}Hz")

            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

        if len(video_profiles) < 2:
            print("❌ Need at least 2 videos to perform alignment")
            return 1

        # Determine reference video and calculate initial offsets
        print("\n" + "="*60)
        print("DETERMINING REFERENCE VIDEO")
        print("="*60)

        # Get reference selection strategy from config or default
        use_earliest_start = False  # Default to latest start (current logic)
        if config and 'video_aligner' in config and 'alignment' in config['video_aligner']:
            alignment_config = config['video_aligner']['alignment']
            if 'reference_strategy' in alignment_config:
                strategy = alignment_config['reference_strategy']
                if strategy == 'earliest_start':
                    use_earliest_start = True
                elif strategy == 'latest_start':
                    use_earliest_start = False
                else:
                    print(f"Warning: Unknown reference_strategy '{strategy}', using default (latest_start)")

        reference_video, offsets = determine_reference_video(video_profiles, use_earliest_start)
        print(f"\n✅ Reference video: {reference_video}")
        print("\nInitial offsets:")
        for video, offset in offsets.items():
            print(f"  {video}: {offset:.3f}s")

        # Calculate matching duration
        print("\n" + "="*60)
        print("CALCULATING MATCHING DURATION")
        print("="*60)

        matching_duration = calculate_matching_duration(video_profiles, offsets)
        print(f"\n✅ Matching duration: {matching_duration:.1f} seconds")

        # Create visualization
        output_plot_path = os.path.join(video_dir, 'alignment_analysis.png')
        plot_energy_profiles_overlay(video_profiles, offsets, matching_duration, output_plot_path)
    
    # Store results in database
    if db:
        print("\n" + "="*60)
        print("STORING RESULTS IN DATABASE")
        print("="*60)

        # Extract source name from directory path (e.g., "vid_shot1" from full path)
        # Look for vid_shot pattern in the path, or use parent directory
        path_parts = video_dir.split('/')
        vid_shot_parts = [part for part in path_parts if part.startswith('vid_shot')]
        if vid_shot_parts:
            source_name = vid_shot_parts[0]  # Use first vid_shot found
        else:
            # Fallback: use parent directory of current directory
            source_name = os.path.basename(os.path.dirname(video_dir.rstrip('/')))

        print(f"Source name: {source_name}")

        if enable_chunk_processing:
            # Store chunk processing results using ChunkVideoAlignmentDatabase
            for prefix, group in camera_groups.items():
                for chunk in group.chunks:
                    print(f"Chunk: {chunk.filename} -> source: {source_name}, camera: {prefix}, offset: {chunk.start_time_offset:.3f}s")

                    try:
                        # Get reference information
                        reference_group = camera_groups[reference_prefix]
                        reference_chunk = reference_group.chunks[0]

                        success = db.insert_chunk_alignment(
                            source=source_name,
                            chunk_filename=chunk.filename,
                            camera_prefix=prefix,
                            chunk_order=chunk.chunk_number,
                            start_time_offset=chunk.start_time_offset,
                            chunk_duration=chunk.duration,
                            reference_camera_prefix=reference_prefix,
                            reference_chunk_filename=reference_chunk.filename,
                            session_id=None,  # Could be derived from source_name if needed
                            method_type=method_type
                        )
                        if success:
                            print(f"✅ Saved chunk alignment data for {chunk.filename} (position: {chunk.start_time_offset:.3f}s, duration: {chunk.duration:.1f}s)")
                        else:
                            print(f"❌ Failed to save chunk alignment data for {chunk.filename}")
                    except Exception as e:
                        print(f"❌ Database error for {chunk.filename}: {e}")
        else:
            # Store legacy processing results
            for video_name, offset in offsets.items():
                # Extract camera type from video filename (e.g., "1" from "cam_1.mp4")
                video_basename = os.path.basename(video_name)
                try:
                    # Extract number from cam_X.mp4 format
                    if video_basename.startswith('cam_') and video_basename.endswith('.mp4'):
                        camera_type = int(video_basename.split('_')[1].split('.')[0])
                    else:
                        # Fallback: try to extract any number from filename
                        numbers = re.findall(r'\d+', video_basename)
                        camera_type = int(numbers[0]) if numbers else 1
                except (ValueError, IndexError):
                    camera_type = 1  # Default fallback

                print(f"Video: {video_basename} -> source: {source_name}, camera_type: {camera_type}")

                try:
                    success = db.insert_video_alignment(
                        source=source_name,
                        start_time_offset=offset,
                        matching_duration=matching_duration,
                        camera_type=camera_type
                    )
                    if success:
                        print(f"✅ Saved alignment data for {video_name}")
                    else:
                        print(f"❌ Failed to save data for {video_name}")
                except Exception as e:
                    print(f"❌ Database error for {video_name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("ALIGNMENT SUMMARY")
    print("="*60)

    if enable_chunk_processing:
        print(f"Processing mode:    Timeline-based chunk alignment")
        print(f"Reference camera:   {reference_prefix} (selected via audio pattern analysis - has most extra content)")
        print(f"Camera groups:      {len(camera_groups)}")
        print(f"Total chunks:       {sum(len(group.chunks) for group in camera_groups.values())}")

        print("\nCamera group summary:")
        for prefix, group in sorted(camera_groups.items(), key=lambda x: min(chunk.start_time_offset for chunk in x[1].chunks)):
            total_chunks = len(group.chunks)
            earliest_chunk_start = min(chunk.start_time_offset for chunk in group.chunks)
            latest_chunk_end = max(chunk.start_time_offset + chunk.duration for chunk in group.chunks)
            timeline_span = latest_chunk_end - earliest_chunk_start

            if prefix == reference_prefix:
                print(f"  {prefix}: {total_chunks} chunks, timeline span: {earliest_chunk_start:.1f}s to {latest_chunk_end:.1f}s ({timeline_span:.1f}s) (REFERENCE)")
            else:
                print(f"  {prefix}: {total_chunks} chunks, timeline span: {earliest_chunk_start:.1f}s to {latest_chunk_end:.1f}s ({timeline_span:.1f}s)")

        if save_output_videos:
            print("\nGenerated output videos:")
            for prefix, group in camera_groups.items():
                if group.output_video_path:
                    print(f"  {prefix}: {group.output_video_path}")
        else:
            print("\nOutput video generation: Disabled")

        print("\nAbsolute timeline positions (all chunks):")
        all_chunks = []
        for prefix, chunk_list in final_chunk_offsets.items():
            for filename, offset in chunk_list:
                all_chunks.append((filename, offset, prefix))

        # Sort by timeline position to show chronological order
        all_chunks.sort(key=lambda x: x[1])

        for filename, offset, prefix in all_chunks:
            if prefix == reference_prefix and offset == 0.0:
                print(f"  {filename} ({prefix}): {offset:.3f}s (REFERENCE START)")
            else:
                print(f"  {filename} ({prefix}): {offset:.3f}s")
    else:
        print(f"Processing mode:    Legacy single-video alignment")
        print(f"Reference video:    {reference_video}")
        print(f"Matching duration:  {matching_duration:.1f} seconds")
        print("\nVideo offsets:")
        for video, offset in sorted(offsets.items(), key=lambda x: x[1]):
            if video == reference_video:
                print(f"  {video}: {offset:.3f}s (REFERENCE)")
            else:
                print(f"  {video}: {offset:.3f}s")
    
    # Clean up
    if os.path.exists('temp_audio'):
        try:
            os.rmdir('temp_audio')
        except:
            pass
    
    print("\n✅ Alignment analysis completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())