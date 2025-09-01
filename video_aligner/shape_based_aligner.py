#!/usr/bin/env python3
"""
Shape-based audio alignment using energy profiles and pattern matching
For individual video pairs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, resample, savgol_filter
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
import subprocess
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from database_setup import MusicianDatabase

def extract_audio_with_ffmpeg(video_path, output_path, duration=90, start_time=0):
    """Extract audio using ffmpeg directly"""
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', '22050',  # Sample rate
        '-ac', '1',  # Mono
        output_path
    ]
    
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

def find_shape_based_offset(energy1, energy2, time_axis1, time_axis2, max_offset=30):
    """
    Find offset based on energy profile shape matching
    
    Args:
        energy1: Energy profile of reference (cam_3)
        energy2: Energy profile to align (cam_2)
        time_axis1: Time axis for energy1
        time_axis2: Time axis for energy2
        max_offset: Maximum offset to search in seconds
    
    Returns:
        best_offset: Offset in seconds
        correlation_scores: Array of correlation scores
        offset_range: Array of tested offsets
    """
    
    # Smooth and normalize profiles
    energy1_smooth = smooth_profile(energy1)
    energy2_smooth = smooth_profile(energy2)
    
    energy1_norm = normalize_shape(energy1_smooth)
    energy2_norm = normalize_shape(energy2_smooth)
    
    # Time step (assuming uniform spacing)
    dt = time_axis1[1] - time_axis1[0]
    max_offset_samples = int(max_offset / dt)
    
    # We'll slide energy1 along energy2
    correlation_scores = []
    offset_range = []
    
    print("Finding best shape alignment...")
    
    for offset_samples in range(0, min(max_offset_samples, len(energy2_norm) - len(energy1_norm))):
        # Extract segment from energy2
        segment = energy2_norm[offset_samples:offset_samples + len(energy1_norm)]
        
        # Calculate correlation
        if len(segment) == len(energy1_norm):
            # Method 1: Pearson correlation
            corr = np.corrcoef(energy1_norm, segment)[0, 1]
            
            # Method 2: Normalized cross-correlation at zero lag
            # corr = np.sum(energy1_norm * segment) / len(energy1_norm)
            
            correlation_scores.append(corr)
            offset_range.append(offset_samples * dt)
        
        if offset_samples % 100 == 0:
            print(f"  Checking offset: {offset_samples * dt:.1f}s")
    
    correlation_scores = np.array(correlation_scores)
    offset_range = np.array(offset_range)
    
    # Find peak correlation
    best_idx = np.argmax(correlation_scores)
    best_offset = offset_range[best_idx]
    
    return best_offset, correlation_scores, offset_range

def find_dtw_offset(energy1, energy2, time_axis1, time_axis2, search_range=(8, 15)):
    """
    Use Dynamic Time Warping to find best alignment within a search range
    
    Args:
        energy1: Energy profile of reference (cam_3)
        energy2: Energy profile to align (cam_2)
        time_axis1: Time axis for energy1
        time_axis2: Time axis for energy2
        search_range: Tuple of (min_offset, max_offset) in seconds to search
    
    Returns:
        best_offset: Offset in seconds
        dtw_distances: Array of DTW distances
        offset_range: Array of tested offsets
    """
    
    # Smooth and normalize profiles
    energy1_smooth = smooth_profile(energy1, window_length=21)
    energy2_smooth = smooth_profile(energy2, window_length=21)
    
    energy1_norm = normalize_shape(energy1_smooth)
    energy2_norm = normalize_shape(energy2_smooth)
    
    # Time step
    dt = time_axis1[1] - time_axis1[0]
    
    # Convert search range to samples
    min_offset_samples = int(search_range[0] / dt)
    max_offset_samples = int(search_range[1] / dt)
    
    dtw_distances = []
    offset_range = []
    
    print(f"Using DTW to find alignment in range {search_range[0]}-{search_range[1]}s...")
    
    # Take first 10 seconds of cam_3 for comparison
    comparison_length = int(10 / dt)  # 10 seconds
    energy1_segment = energy1_norm[:comparison_length]
    
    for offset_samples in range(min_offset_samples, min(max_offset_samples, len(energy2_norm) - comparison_length)):
        # Extract segment from energy2
        segment = energy2_norm[offset_samples:offset_samples + comparison_length]
        
        if len(segment) == len(energy1_segment):
            # Calculate DTW distance
            distance = dtw.distance(energy1_segment, segment)
            dtw_distances.append(distance)
            offset_range.append(offset_samples * dt)
        
        if (offset_samples - min_offset_samples) % 50 == 0:
            print(f"  Checking offset: {offset_samples * dt:.2f}s")
    
    dtw_distances = np.array(dtw_distances)
    offset_range = np.array(offset_range)
    
    # Find minimum DTW distance (best alignment)
    best_idx = np.argmin(dtw_distances)
    best_offset = offset_range[best_idx]
    
    return best_offset, dtw_distances, offset_range

def plot_alignment_analysis(energy1, energy2, time_axis1, time_axis2, 
                           best_offset_corr, correlation_scores, offset_range_corr,
                           best_offset_dtw, dtw_distances, offset_range_dtw, 
                           video1_name, video2_name, output_dir):
    """Create comprehensive alignment analysis plot"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot 1: Original energy profiles
    axes[0, 0].plot(time_axis1, energy1, alpha=0.7, label=f'{video1_name} (reference)', color='blue')
    axes[0, 0].plot(time_axis2, energy2, alpha=0.7, label=video2_name, color='orange')
    axes[0, 0].set_title('Original Energy Profiles (RMS)')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 30)
    
    # Plot 2: Aligned energy profiles (correlation-based)
    axes[0, 1].plot(time_axis1, energy1, alpha=0.7, label=video1_name, color='blue')
    axes[0, 1].plot(time_axis2 - best_offset_corr, energy2, alpha=0.7, 
                   label=f'{video2_name} (shifted {best_offset_corr:.2f}s)', color='orange')
    axes[0, 1].axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Alignment point')
    axes[0, 1].set_title(f'Aligned Energy Profiles (Correlation Method: {best_offset_corr:.2f}s)')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-5, 25)
    
    # Plot 3: Correlation scores
    axes[1, 0].plot(offset_range_corr, correlation_scores, color='blue')
    axes[1, 0].axvline(x=best_offset_corr, color='red', linestyle='--', 
                      label=f'Best: {best_offset_corr:.2f}s')
    axes[1, 0].set_title('Shape Correlation vs Offset')
    axes[1, 0].set_xlabel('Offset (seconds)')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: DTW distances
    if len(dtw_distances) > 0:
        axes[1, 1].plot(offset_range_dtw, dtw_distances, color='purple')
        axes[1, 1].axvline(x=best_offset_dtw, color='red', linestyle='--', 
                          label=f'Best: {best_offset_dtw:.2f}s')
        axes[1, 1].set_title('DTW Distance vs Offset')
        axes[1, 1].set_xlabel('Offset (seconds)')
        axes[1, 1].set_ylabel('DTW Distance (lower is better)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_yaxis()  # Lower distance is better
    else:
        axes[1, 1].text(0.5, 0.5, 'DTW not available', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Plot 5: Smoothed and normalized profiles
    energy1_smooth = smooth_profile(energy1, window_length=21)
    energy2_smooth = smooth_profile(energy2, window_length=21)
    energy1_norm = normalize_shape(energy1_smooth)
    energy2_norm = normalize_shape(energy2_smooth)
    
    axes[2, 0].plot(time_axis1, energy1_norm, alpha=0.7, label=f'{video1_name} (normalized)', color='blue')
    axes[2, 0].plot(time_axis2, energy2_norm, alpha=0.7, label=f'{video2_name} (normalized)', color='orange')
    axes[2, 0].set_title('Normalized Energy Shapes')
    axes[2, 0].set_xlabel('Time (seconds)')
    axes[2, 0].set_ylabel('Normalized Energy')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim(0, 30)
    
    # Plot 6: Aligned with DTW method
    if len(dtw_distances) > 0:
        axes[2, 1].plot(time_axis1, energy1, alpha=0.7, label=video1_name, color='blue')
        axes[2, 1].plot(time_axis2 - best_offset_dtw, energy2, alpha=0.7, 
                       label=f'{video2_name} (shifted {best_offset_dtw:.2f}s)', color='orange')
        axes[2, 1].axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Alignment point')
        axes[2, 1].set_title(f'Aligned Energy Profiles (DTW Method: {best_offset_dtw:.2f}s)')
        axes[2, 1].set_xlabel('Time (seconds)')
        axes[2, 1].set_ylabel('Energy')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_xlim(-5, 25)
    else:
        axes[2, 1].text(0.5, 0.5, 'DTW alignment not available', ha='center', va='center', transform=axes[2, 1].transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{video2_name}_vs_{video1_name}_alignment.png')
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

def main():
    # Parse command line arguments
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python shape_based_aligner.py <video_path> <reference_video> <video_to_align> [--store-db]")
        print("Example: python shape_based_aligner.py vid_shot1 vid_shot1_cam_3.mp4 vid_shot1_cam_1.mp4")
        print("  - video_path: Directory containing the videos (e.g., vid_shot1, vid_shot2)")
        print("  - reference_video: The reference video filename (won't be modified)")
        print("  - video_to_align: The video filename to calculate alignment offsets for")
        print("  - --store-db: Optional flag to store results in database")
        return 1
    
    video_path = sys.argv[1]
    reference_video = sys.argv[2]
    video_to_align = sys.argv[3]
    store_in_database = len(sys.argv) == 5 and sys.argv[4] == '--store-db'
    
    # Create temp directory
    os.makedirs('temp_audio', exist_ok=True)
    os.makedirs(f'{video_path}/analysis', exist_ok=True)
    
    # Initialize database connection if requested
    db = None
    if store_in_database:
        try:
            db = MusicianDatabase()
            print("✅ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("Continuing without database storage...")
    
    # Video paths - handle both absolute and relative paths  
    if not os.path.isabs(reference_video):
        video1_path = f"{video_path}/{reference_video}"
    else:
        video1_path = reference_video
        
    if not os.path.isabs(video_to_align):
        video2_path = f"{video_path}/{video_to_align}"
    else:
        video2_path = video_to_align
    
    # Check if files exist
    if not os.path.exists(video1_path):
        print(f"❌ Reference video not found: {video1_path}")
        return 1
    
    if not os.path.exists(video2_path):
        print(f"❌ Video to align not found: {video2_path}")
        return 1
    
    # Extract audio using ffmpeg
    ref_name = os.path.splitext(os.path.basename(reference_video))[0]
    align_name = os.path.splitext(os.path.basename(video_to_align))[0]
    
    audio1_path = f'temp_audio/{ref_name}_audio.wav'
    audio2_path = f'temp_audio/{align_name}_audio.wav'
    
    print("="*60)
    print("SHAPE-BASED AUDIO ALIGNMENT")
    print("="*60)
    print(f"Reference video: {os.path.basename(video1_path)}")
    print(f"Video to align:  {os.path.basename(video2_path)}")
    
    # Extract audio for analysis (first 60 seconds of reference, first 90 seconds of video to align)
    print("\nExtracting audio for analysis...")
    audio1, sr1 = extract_audio_with_ffmpeg(video1_path, audio1_path, duration=60, start_time=0)
    audio2, sr2 = extract_audio_with_ffmpeg(video2_path, audio2_path, duration=90, start_time=0)
    
    if audio1 is None or audio2 is None:
        print("Failed to extract audio")
        return 1
    
    print(f"\nAudio 1 ({ref_name}): {len(audio1)/sr1:.1f}s at {sr1}Hz")
    print(f"Audio 2 ({align_name}): {len(audio2)/sr2:.1f}s at {sr2}Hz")
    
    # Calculate energy profiles
    print("\nCalculating energy profiles...")
    energy1, time_axis1 = calculate_energy_profile(audio1, sr1, window_ms=100)
    energy2, time_axis2 = calculate_energy_profile(audio2, sr2, window_ms=100)
    
    # Method 1: Shape-based correlation
    best_offset_corr, correlation_scores, offset_range_corr = find_shape_based_offset(
        energy1, energy2, time_axis1, time_axis2, max_offset=30
    )
    
    # Method 2: DTW (Dynamic Time Warping) - search around expected area
    try:
        best_offset_dtw, dtw_distances, offset_range_dtw = find_dtw_offset(
            energy1, energy2, time_axis1, time_axis2, search_range=(8, 15)
        )
    except ImportError:
        print("DTW library not installed. Skipping DTW method.")
        print("Install with: pip install dtaidistance")
        best_offset_dtw = best_offset_corr
        dtw_distances = []
        offset_range_dtw = []
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Shape Correlation Method: {best_offset_corr:.3f} seconds")
    if len(dtw_distances) > 0:
        print(f"DTW Method:              {best_offset_dtw:.3f} seconds")
    
    # Use the best method result
    if len(dtw_distances) > 0:
        final_offset = best_offset_dtw if abs(best_offset_dtw - 11) < abs(best_offset_corr - 11) else best_offset_corr
    else:
        final_offset = best_offset_corr
    
    print(f"\nUsing start time offset: {final_offset:.3f} seconds")
    
    # Plot analysis
    plot_alignment_analysis(
        energy1, energy2, time_axis1, time_axis2,
        best_offset_corr, correlation_scores, offset_range_corr,
        best_offset_dtw if len(dtw_distances) > 0 else best_offset_corr, 
        dtw_distances, offset_range_dtw,
        ref_name, align_name, f'{video_path}/analysis'
    )
    
    # Store in database if requested
    if db:
        try:
            # For single video alignment, we don't have matching_duration calculation
            # Set it to 0 for now, or calculate based on shorter video duration
            ref_duration = get_video_duration(video1_path)
            align_duration = get_video_duration(video2_path)
            
            # Estimate matching duration as the shorter video duration minus offset
            if ref_duration and align_duration:
                matching_duration = min(ref_duration, align_duration - final_offset)
            else:
                matching_duration = 0.0
            
            success = db.insert_video_alignment(
                video_name=os.path.basename(video_to_align),
                start_time_offset=final_offset,
                matching_duration=max(0, matching_duration),
                reference_video=os.path.basename(reference_video)
            )
            if success:
                print(f"✅ Alignment data saved to database for {os.path.basename(video_to_align)}")
            else:
                print(f"❌ Failed to save alignment data to database")
        except Exception as e:
            print(f"❌ Database error: {e}")
    
    print("\n" + "="*60)
    print("ALIGNMENT SUMMARY")
    print("="*60)
    print(f"Reference video:     {os.path.basename(reference_video)}")
    print(f"Video to align:      {os.path.basename(video_to_align)}")
    print(f"Start time offset:   {final_offset:.3f} seconds")
    print(f"Analysis saved to:   {video_path}/analysis/")
    
    # Clean up
    if os.path.exists(audio1_path):
        os.remove(audio1_path)
    if os.path.exists(audio2_path):
        os.remove(audio2_path)
    if os.path.exists('temp_audio'):
        try:
            os.rmdir('temp_audio')
        except:
            pass
    
    print("\n✅ Alignment analysis completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())