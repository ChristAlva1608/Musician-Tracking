#!/usr/bin/env python3
"""
Shape-based audio alignment for multiple videos
Automatically determines reference video and aligns all others
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, resample, savgol_filter
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
import subprocess
import os
import sys
import glob
import yaml
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from src.database.setup import MusicianDatabase

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
        time_axis_target: Time axis for energy_target
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

def determine_reference_video(video_profiles):
    """
    Determine which video should be the reference based on which started recording first
    The reference video is the one where similar patterns appear LATEST (has extra content at beginning)
    This ensures all other videos have positive offsets
    
    Args:
        video_profiles: Dictionary with video names as keys and (energy, time_axis) as values
    
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
    
    # Find video that started recording first
    # The video that started first will have POSITIVE offsets TO other videos
    # (meaning other videos start later relative to it)
    avg_offsets_from_video = np.mean(offset_matrix, axis=1)  # Average of each row
    reference_idx = np.argmax(avg_offsets_from_video)
    reference_video = videos[reference_idx]
    
    print(f"\nAverage offsets FROM each video (positive = others start later):")
    for i, video in enumerate(videos):
        print(f"  {video} -> others: {avg_offsets_from_video[i]:.3f}s")
    print(f"  -> Choosing {reference_video} (highest average - started recording first)")
    
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
    for video, (energy, time_axis) in video_profiles.items():
        profile_durations[video] = time_axis[-1]
        print(f"  {video} profile duration: {profile_durations[video]:.1f}s (from full audio)")
    
    # Since we're now extracting full audio, profile durations ARE the actual durations
    actual_durations = profile_durations.copy()
    print("\nActual video durations (from full audio extraction):")
    for video, duration in actual_durations.items():
        print(f"  {video}: {duration:.1f}s ({duration/60:.1f} minutes)")
    
    # Calculate effective start and end times after alignment using ACTUAL durations
    aligned_ranges = {}
    print("\nDEBUG: Aligned ranges calculation:")
    for video in video_profiles:
        start = offsets[video]
        end = actual_durations[video]
        aligned_ranges[video] = (start, end)
        print(f"  {video}: start={start:.1f}s, end={end:.1f}s, effective_duration={end-start:.1f}s")
    
    # Find overlapping region
    latest_start = max(start for start, _ in aligned_ranges.values())
    earliest_end = min(end for _, end in aligned_ranges.values())
    
    print(f"\nDEBUG: Overlap calculation:")
    print(f"  Latest start: {latest_start:.1f}s")
    print(f"  Earliest end: {earliest_end:.1f}s")
    
    # The matching duration is from latest start to earliest end
    matching_duration = earliest_end - latest_start
    
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

def load_config(config_path='config.yaml'):
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
    elif config and 'video' in config and 'alignment_directory' in config['video']:
        # Use config file
        video_dir = config['video']['alignment_directory']
        print(f"Using config file directory: {video_dir}")
    else:
        print("Usage: python shape_based_aligner_multi.py <video_directory>")
        print("   OR: Configure alignment_directory in config.yaml")
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
    
    # Find all MP4 files in the directory
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if len(video_files) == 0:
        print(f"❌ No MP4 files found in {video_dir}")
        return 1
    
    print("="*60)
    print("MULTI-VIDEO SHAPE-BASED ALIGNMENT")
    print("="*60)
    print(f"Found {len(video_files)} videos in {video_dir}:")
    for video in video_files:
        print(f"  - {os.path.basename(video)}")
    
    # Initialize database connection
    try:
        db = MusicianDatabase()
        print("\n✅ Connected to database")
    except Exception as e:
        print(f"\n❌ Database connection failed: {e}")
        print("Continuing without database storage...")
        db = None
    
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
    
    reference_video, offsets = determine_reference_video(video_profiles)
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
        
        for video_name, offset in offsets.items():
            try:
                success = db.insert_video_alignment(
                    video_name=video_name,
                    start_time_offset=offset,
                    matching_duration=matching_duration,
                    reference_video=reference_video if video_name != reference_video else None
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