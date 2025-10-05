#!/bin/bash

# Batch video processing script for Musician Tracking
# Processes all videos in vid_shot1 and vid_shot2 directories

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DETECT_SCRIPT="$PROJECT_ROOT/src/detect.py"

# Video directories
VID_SHOT1_ORIGINAL_DIR="$PROJECT_ROOT/video/multi-cam video/vid_shot1/original_video"
VID_SHOT1_COWBOY_DIR="$PROJECT_ROOT/video/multi-cam video/vid_shot1/cowboy_shot"

# Log file
LOG_FILE="$PROJECT_ROOT/data/outputs/batch_processing_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJECT_ROOT/data/outputs"

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to process a single video
process_video() {
    video_path=$1
    folder_name=$2
    skip_frames=${3:-0}
    video_name=$(basename "$video_path" .mp4)
    
    # Generate output video name: {folder_name}_{input_video_name}_sk_{skip_frames}.mp4
    output_video_name="${folder_name}_${video_name}_sk_${skip_frames}.mp4"
    
    print_color "$BLUE" "Processing: $video_name (skip_frames: $skip_frames)"
    log_message "Starting processing: $video_path with skip_frames=$skip_frames"
    
    # Create temporary config file with video path
    temp_config="/tmp/config_$(date +%s).yaml"
    cp "$PROJECT_ROOT/src/config/config_v1.yaml" "$temp_config"
    
    # Update video path and output settings in config
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|source_path:.*|source_path: '$video_path'|" "$temp_config"
        sed -i '' "s|output_video_path:.*|output_video_path: 'output/$output_video_name'|" "$temp_config"
        sed -i '' "s|skip_frames:.*|skip_frames: $skip_frames|" "$temp_config"
    else
        # Linux
        sed -i "s|source_path:.*|source_path: '$video_path'|" "$temp_config"
        sed -i "s|output_video_path:.*|output_video_path: 'output/$output_video_name'|" "$temp_config"
        sed -i "s|skip_frames:.*|skip_frames: $skip_frames|" "$temp_config"
    fi
    
    # Get matching duration from database for synchronized processing
    video_basename=$(basename "$video_path")
    matching_duration=$(cd "$PROJECT_ROOT" && python3 -c "
import sys
import os
# Suppress all output except our result
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

from src.database.setup import MusicianDatabase
try:
    db = MusicianDatabase()
    result = db.supabase.table('video_alignment_offsets').select('matching_duration').eq('video_name', '$video_basename').execute()
    db.close()
    
    # Restore stdout only for our result
    sys.stdout = sys.__stdout__
    if result.data:
        print(f\"{result.data[0]['matching_duration']:.3f}\")
    else:
        print('0')
except:
    sys.stdout = sys.__stdout__
    print('0')
" 2>/dev/null)
    
    # Run detection script with synchronized end time
    start_time=$(date +%s)
    
    cd "$PROJECT_ROOT"
    if [ "$matching_duration" != "0" ]; then
        print_color "$BLUE" "Using synchronized duration: ${matching_duration}s"
        python3 src/detect_v2.py --config "$temp_config" 2>&1 | tee -a "$LOG_FILE"
    else
        print_color "$YELLOW" "No synchronization data found, processing entire video"
        python3 src/detect_v2.py --config "$temp_config" 2>&1 | tee -a "$LOG_FILE"
    fi
    exit_code=${PIPESTATUS[0]}
    
    end_time=$(date +%s)
    processing_time=$((end_time - start_time))
    
    # Clean up temp config
    rm -f "$temp_config"
    
    if [ $exit_code -eq 0 ]; then
        print_color "$GREEN" "✅ Successfully processed: $output_video_name (${processing_time}s)"
        log_message "Successfully completed: $output_video_name in ${processing_time} seconds"
        return 0
    else
        print_color "$RED" "❌ Failed to process: $output_video_name"
        log_message "ERROR: Failed to process $output_video_name"
        return 1
    fi
}

# Function to process videos in a specific folder (max 3 videos)
process_folder() {
    dir_path=$1
    folder_name=$2
    skip_frames=${3:-0}
    
    print_color "$YELLOW" "\n=== Processing folder: $folder_name (skip_frames: $skip_frames) ==="
    log_message "Starting folder: $dir_path with skip_frames=$skip_frames"
    
    # Check if directory exists
    if [ ! -d "$dir_path" ]; then
        print_color "$RED" "Directory not found: $dir_path"
        log_message "ERROR: Directory not found: $dir_path"
        return 1
    fi
    
    # Find all MP4 video files and limit to first 3
    video_count=0
    success_count=0
    fail_count=0
    
    for video in "$dir_path"/*.mp4 "$dir_path"/*.MP4; do
        if [ -f "$video" ] && [ $video_count -lt 3 ]; then
            ((video_count++))
            print_color "$YELLOW" "\n--- Video $video_count/3 ---"
            
            if process_video "$video" "$folder_name" "$skip_frames"; then
                ((success_count++))
            else
                ((fail_count++))
            fi
        fi
    done
    
    if [ $video_count -eq 0 ]; then
        print_color "$YELLOW" "No video files found in $folder_name"
        log_message "No video files found in $dir_path"
    else
        print_color "$BLUE" "\n=== Folder Summary: $folder_name ==="
        print_color "$GREEN" "Successfully processed: $success_count"
        print_color "$RED" "Failed: $fail_count"
        print_color "$BLUE" "Total: $video_count"
        log_message "Folder $folder_name complete: Success=$success_count, Failed=$fail_count, Total=$video_count"
    fi
}

# Main execution
main() {
    print_color "$YELLOW" "========================================="
    print_color "$YELLOW" "   Musician Tracking Batch Processor    "
    print_color "$YELLOW" "========================================="
    
    log_message "=== Starting batch video processing ==="
    log_message "Project root: $PROJECT_ROOT"
    log_message "Log file: $LOG_FILE"
    
    # Check if detect_v2.py exists
    DETECT_V2_SCRIPT="$PROJECT_ROOT/src/detect_v2.py"
    if [ ! -f "$DETECT_V2_SCRIPT" ]; then
        print_color "$RED" "Error: detect_v2.py not found at $DETECT_V2_SCRIPT"
        log_message "ERROR: detect_v2.py not found at $DETECT_V2_SCRIPT"
        exit 1
    fi
    
    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        print_color "$RED" "Error: Python3 is not installed or not in PATH"
        log_message "ERROR: Python3 not found"
        exit 1
    fi
    
    # Process specific folders
    total_start_time=$(date +%s)
    
    # Process 3 videos from original video folder
    process_folder "$VID_SHOT1_ORIGINAL_DIR" "original_video" 0
    
    # Process 3 videos from cowboy_shot folder
    process_folder "$VID_SHOT1_COWBOY_DIR" "cowboy_shot" 0
    
    total_end_time=$(date +%s)
    total_processing_time=$((total_end_time - total_start_time))
    
    # Final summary
    print_color "$YELLOW" "\n========================================="
    print_color "$GREEN" "   Batch Processing Complete!            "
    print_color "$YELLOW" "========================================="
    print_color "$BLUE" "Total processing time: ${total_processing_time} seconds"
    print_color "$BLUE" "Log file: $LOG_FILE"
    
    log_message "=== Batch processing complete in ${total_processing_time} seconds ==="
}

# Handle command line arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --original     Process only original video folder (3 videos max)"
    echo "  --cowboy       Process only cowboy_shot folder (3 videos max)"
    echo ""
    echo "By default, processes 3 videos from both original_video and cowboy_shot folders."
    echo "Output format: {folder_name}_{input_video_name}_sk_{skip_frames}.mp4"
    echo "Example: original_video_cam_1_sk_0.mp4"
    exit 0
elif [ "$1" == "--original" ]; then
    print_color "$YELLOW" "Processing only original video folder"
    process_folder "$VID_SHOT1_ORIGINAL_DIR" "original_video" 0
elif [ "$1" == "--cowboy" ]; then
    print_color "$YELLOW" "Processing only cowboy_shot folder"
    process_folder "$VID_SHOT1_COWBOY_DIR" "cowboy_shot" 0
else
    # Process both folders
    main
fi