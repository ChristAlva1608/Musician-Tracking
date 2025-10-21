#!/bin/bash

################################################################################
# 360° Video Chunk Concatenation Script
#
# Purpose: Automatically concatenate exported 2D video chunks from Insta360
#          and rename according to naming convention
#
# Usage: ./concat_360_chunks.sh <participant> <instrument> <date> <camera> <model>
#
# Example: ./concat_360_chunks.sh "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1"
#
# Output: 2D_from_360_Camera_1-Quan-Violin-20250725-X5_1.mp4
################################################################################

set -e  # Exit on error

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg is not installed. Please install it first:"
    echo "  brew install ffmpeg"
    exit 1
fi

# Parse arguments
PARTICIPANT="$1"
INSTRUMENT="$2"
DATE="$3"
CAMERA="$4"
MODEL="$5"

if [ -z "$PARTICIPANT" ] || [ -z "$INSTRUMENT" ] || [ -z "$DATE" ] || [ -z "$CAMERA" ] || [ -z "$MODEL" ]; then
    echo "Usage: $0 <participant> <instrument> <date> <camera> <model>"
    echo ""
    echo "Example:"
    echo "  $0 \"Quan\" \"Violin\" \"2025-07-25\" \"Camera 1\" \"X5 1\""
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

# Find all MP4 files (excluding already concatenated files)
MP4_FILES=($(ls -1 *.mp4 2>/dev/null | grep -v "^2D_from_360" | sort))

if [ ${#MP4_FILES[@]} -eq 0 ]; then
    echo "ERROR: No MP4 chunks found in:"
    echo "  $FULL_PATH"
    echo ""
    echo "Make sure you've exported chunks from Insta360 Studio first."
    exit 1
fi

echo "========================================="
echo "360° to 2D Concatenation Script"
echo "========================================="
echo "Participant: $PARTICIPANT"
echo "Instrument:  $INSTRUMENT"
echo "Date:        $DATE"
echo "Camera:      $CAMERA"
echo "Model:       $MODEL"
echo ""
echo "Working Directory:"
echo "  $FULL_PATH"
echo ""
echo "Found ${#MP4_FILES[@]} chunk(s):"
for file in "${MP4_FILES[@]}"; do
    size=$(du -h "$file" | cut -f1)
    echo "  - $file ($size)"
done
echo ""

# If only one file, just rename it
if [ ${#MP4_FILES[@]} -eq 1 ]; then
    OUTPUT_NAME="2D_from_360_${CAMERA_NUM}-${PARTICIPANT}-${INSTRUMENT}-${DATE_FORMATTED}-${MODEL_FORMATTED}.mp4"
    echo "Only one chunk found. Renaming to:"
    echo "  $OUTPUT_NAME"
    mv "${MP4_FILES[0]}" "$OUTPUT_NAME"

    echo ""
    echo "✓ Done!"
    echo ""
    echo "Output file:"
    ls -lh "$OUTPUT_NAME"
    echo ""

    # Get duration
    echo "Verifying video duration..."
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_NAME" | awk '{printf "Duration: %.2f minutes (%.2f seconds)\n", $1/60, $1}'

    exit 0
fi

# Create concat list file
CONCAT_FILE="concat_list.txt"
rm -f "$CONCAT_FILE"

for file in "${MP4_FILES[@]}"; do
    echo "file '$file'" >> "$CONCAT_FILE"
done

echo "Created concatenation list:"
cat "$CONCAT_FILE"
echo ""

# Output filename
OUTPUT_NAME="2D_from_360_${CAMERA_NUM}-${PARTICIPANT}-${INSTRUMENT}-${DATE_FORMATTED}-${MODEL_FORMATTED}.mp4"

echo "Concatenating chunks..."
echo "Output: $OUTPUT_NAME"
echo ""

# Concatenate using ffmpeg
# Using -c copy for lossless concatenation (no re-encoding)
ffmpeg -f concat -safe 0 -i "$CONCAT_FILE" -c copy "$OUTPUT_NAME" -y

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Concatenation successful!"
    echo ""

    # Clean up
    rm -f "$CONCAT_FILE"

    # Create backup folder for chunks
    BACKUP_DIR="_chunks_backup"
    mkdir -p "$BACKUP_DIR"

    echo "Moving original chunks to backup folder..."
    for file in "${MP4_FILES[@]}"; do
        mv "$file" "$BACKUP_DIR/"
        echo "  ✓ Moved: $file"
    done

    echo ""
    echo "========================================="
    echo "CONVERSION COMPLETE!"
    echo "========================================="
    echo ""
    echo "Output file:"
    ls -lh "$OUTPUT_NAME"
    echo ""
    echo "Backup chunks:"
    ls -lh "$BACKUP_DIR"
    echo ""

    # Verify duration
    echo "Verifying concatenated video..."
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_NAME")
    MINUTES=$(echo "$DURATION / 60" | bc -l | xargs printf "%.2f")
    echo "Total Duration: $MINUTES minutes"
    echo ""

    # Get video info
    echo "Video Information:"
    ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,codec_name -of default=noprint_wrappers=1 "$OUTPUT_NAME"
    echo ""

    echo "✓ All done! You can safely delete the chunks in _chunks_backup/ if everything looks good."
    echo ""
else
    echo ""
    echo "✗ ERROR: Concatenation failed!"
    echo ""
    echo "This might happen if the chunks have different codecs or resolutions."
    echo "Check that all chunks were exported with identical settings from Insta360 Studio."
    exit 1
fi
