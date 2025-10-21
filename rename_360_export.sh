#!/bin/bash

################################################################################
# 360° Exported File Renaming Script
#
# Purpose: Rename exported 2D files from Insta360 to match naming convention
#
# Usage: ./rename_360_export.sh <participant> <instrument> <date> <camera> <model> <chunk_number>
#
# Example: ./rename_360_export.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1" "020"
#
# Output: 2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_020.mp4
################################################################################

set -e

# Parse arguments
PARTICIPANT="$1"
INSTRUMENT="$2"
DATE="$3"
CAMERA="$4"
MODEL="$5"
CHUNK="$6"

if [ -z "$PARTICIPANT" ] || [ -z "$INSTRUMENT" ] || [ -z "$DATE" ] || [ -z "$CAMERA" ] || [ -z "$MODEL" ] || [ -z "$CHUNK" ]; then
    echo "Usage: $0 <participant> <instrument> <date> <camera> <model> <chunk_number>"
    echo ""
    echo "Example:"
    echo "  $0 \"Quan\" \"Violin\" \"2025-07-25\" \"Camera 1\" \"X5 1\" \"020\""
    echo ""
    exit 1
fi

# Format date (remove hyphens: 2025-07-25 → 20250725)
DATE_FORMATTED="${DATE//-/}"

# Format camera number (Camera 1 → Camera_1)
CAMERA_NUM="${CAMERA// /_}"

# Format model (X5 1 → X5_1)
MODEL_FORMATTED="${MODEL// /_}"

# Construct folder path
BASE_PATH="/Volumes/X10 Pro 1/Phan Dissertation Data"
PARTICIPANT_FOLDER="${PARTICIPANT} - MultiCam Data - ${INSTRUMENT} - ${DATE}"
CAMERA_FOLDER="2D Converted - 360 ${CAMERA} - ${PARTICIPANT} - ${INSTRUMENT} - ${DATE} ${MODEL}"
FULL_PATH="${BASE_PATH}/${PARTICIPANT_FOLDER}/${CAMERA_FOLDER}"

# Check if folder exists
if [ ! -d "$FULL_PATH" ]; then
    echo "ERROR: Output folder does not exist:"
    echo "  $FULL_PATH"
    exit 1
fi

# Change to output directory
cd "$FULL_PATH"

# Find the most recent MP4 file (likely the one just exported)
LATEST_FILE=$(ls -t *.mp4 2>/dev/null | head -1)

if [ -z "$LATEST_FILE" ]; then
    echo "ERROR: No MP4 files found in:"
    echo "  $FULL_PATH"
    echo ""
    echo "Make sure you've exported the file from Insta360 Studio first."
    exit 1
fi

# Construct new filename
NEW_NAME="2D_from_360_${CAMERA_NUM}-${PARTICIPANT}-${INSTRUMENT}-${DATE_FORMATTED}-${MODEL_FORMATTED}-chunk_${CHUNK}.mp4"

echo "========================================="
echo "360° Export File Renaming"
echo "========================================="
echo "Participant: $PARTICIPANT"
echo "Instrument:  $INSTRUMENT"
echo "Date:        $DATE"
echo "Camera:      $CAMERA"
echo "Model:       $MODEL"
echo "Chunk:       $CHUNK"
echo ""
echo "Working Directory:"
echo "  $FULL_PATH"
echo ""
echo "Latest exported file:"
echo "  $LATEST_FILE"
echo ""
echo "Renaming to:"
echo "  $NEW_NAME"
echo ""

# Rename the file
mv "$LATEST_FILE" "$NEW_NAME"

if [ $? -eq 0 ]; then
    echo "✓ Rename successful!"
    echo ""
    ls -lh "$NEW_NAME"
    echo ""

    # Get duration
    if command -v ffprobe &> /dev/null; then
        echo "Video Information:"
        ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$NEW_NAME" | awk '{printf "Duration: %.2f minutes (%.2f seconds)\n", $1/60, $1}'
        ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of default=noprint_wrappers=1 "$NEW_NAME"
    fi
    echo ""
else
    echo "✗ ERROR: Rename failed!"
    exit 1
fi
