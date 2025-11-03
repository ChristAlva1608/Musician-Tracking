#!/bin/bash
# Script to run 2-person landmark detection on multiple videos
# Created: 2025-01-29

echo "=================================================="
echo "2-Person Landmark Detection - Batch Processing"
echo "=================================================="
echo ""

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Set up Python environment (if using virtual environment)
# source venv/bin/activate  # Uncomment if you have a venv

# Function to run detection
run_detection() {
    local config_file=$1
    local video_name=$2

    echo "=================================================="
    echo "Processing: $video_name"
    echo "Config: $config_file"
    echo "=================================================="
    echo ""

    python3 src/detect_v2_3d.py --config "$config_file"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ Successfully processed: $video_name"
        echo ""
    else
        echo ""
        echo "❌ Error processing: $video_name (exit code: $exit_code)"
        echo ""
        return 1
    fi
}

# Track start time
start_time=$(date +%s)

# Run detection for video 1 (VID_20250709_203100_00_010-f.mp4)
run_detection "src/config/config-vid-f.yaml" "VID_20250709_203100_00_010-f.mp4"
video1_status=$?

echo ""
echo "=================================================="
echo "Video 1 completed. Waiting 5 seconds before next..."
echo "=================================================="
sleep 5

# Run detection for video 2 (VID_20250709_191503_00_007-fm.mp4)
run_detection "src/config/config-vid-fm.yaml" "VID_20250709_191503_00_007-fm.mp4"
video2_status=$?

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
minutes=$((total_time / 60))
seconds=$((total_time % 60))

echo ""
echo "=================================================="
echo "Batch Processing Complete!"
echo "=================================================="
echo ""
echo "Summary:"
echo "  Video 1 (f):  $([ $video1_status -eq 0 ] && echo '✅ Success' || echo '❌ Failed')"
echo "  Video 2 (fm): $([ $video2_status -eq 0 ] && echo '✅ Success' || echo '❌ Failed')"
echo ""
echo "Total time: ${minutes}m ${seconds}s"
echo ""
echo "Output locations:"
echo "  - Annotated videos: src/output/"
echo "  - Reports: src/analysis/"
echo "  - Database: musician_frame_analysis table"
echo ""
echo "=================================================="

# Exit with error if any video failed
if [ $video1_status -ne 0 ] || [ $video2_status -ne 0 ]; then
    exit 1
fi

exit 0
