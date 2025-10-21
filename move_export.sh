#!/bin/bash

################################################################################
# Move Exported File from Phan Dissertation Data to Correct Folder
#
# Purpose: Move exported file from export location to correct camera folder
#
# Usage: ./move_export.sh <exported_file> <participant> <instrument> <date> <camera> <model> <chunk>
#
# Example: ./move_export.sh "/Volumes/MAML 26TB 1/Phan Dissertation Data/VID_20250725_142824_00_020.mp4" "Quan" "Violin" "2025-07-25" "Camera 1" "X5 1" "020"
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
EXPORTED_FILE="$1"
PARTICIPANT="$2"
INSTRUMENT="$3"
DATE="$4"
CAMERA="$5"
MODEL="$6"
CHUNK="$7"

if [ -z "$EXPORTED_FILE" ] || [ -z "$PARTICIPANT" ] || [ -z "$INSTRUMENT" ] || [ -z "$DATE" ] || [ -z "$CAMERA" ] || [ -z "$MODEL" ] || [ -z "$CHUNK" ]; then
    echo "Usage: $0 <exported_file> <participant> <instrument> <date> <camera> <model> <chunk>"
    echo ""
    echo "Example:"
    echo "  $0 \"/Volumes/MAML 26TB 1/Phan Dissertation Data/VID_20250725_142824_00_020.mp4\" \"Quan\" \"Violin\" \"2025-07-25\" \"Camera 1\" \"X5 1\" \"020\""
    echo ""
    exit 1
fi

# Check if file exists
if [ ! -f "$EXPORTED_FILE" ]; then
    echo -e "${RED}ERROR: Exported file not found: $EXPORTED_FILE${NC}"
    exit 1
fi

# Determine which drive the file is on
if [[ "$EXPORTED_FILE" == *"/Volumes/MAML 26TB 1/"* ]]; then
    BASE_PATH="/Volumes/MAML 26TB 1/Phan Dissertation Data"
    DRIVE_NAME="MAML 26TB 1"
elif [[ "$EXPORTED_FILE" == *"/Volumes/MAML 26TB 2/"* ]]; then
    BASE_PATH="/Volumes/MAML 26TB 2/Phan Dissertation Data"
    DRIVE_NAME="MAML 26TB 2"
elif [[ "$EXPORTED_FILE" == *"/Volumes/X10 Pro 1/"* ]]; then
    BASE_PATH="/Volumes/X10 Pro 1/Phan Dissertation Data"
    DRIVE_NAME="X10 Pro 1"
else
    echo -e "${RED}ERROR: File is not on a recognized drive${NC}"
    exit 1
fi

# Format date (remove hyphens: 2025-07-25 → 20250725)
DATE_FORMATTED="${DATE//-/}"

# Format camera number (Camera 1 → Camera_1)
CAMERA_NUM="${CAMERA// /_}"

# Format model (X5 1 → X5_1)
MODEL_FORMATTED="${MODEL// /_}"

# Construct paths
PARTICIPANT_FOLDER="${PARTICIPANT} - MultiCam Data - ${INSTRUMENT} - ${DATE}"
CAMERA_FOLDER="2D Converted - 360 ${CAMERA} - ${PARTICIPANT} - ${INSTRUMENT} - ${DATE} ${MODEL}"
DEST_FOLDER="${BASE_PATH}/${PARTICIPANT_FOLDER}/${CAMERA_FOLDER}"

# Check if destination folder exists
if [ ! -d "$DEST_FOLDER" ]; then
    echo -e "${YELLOW}Destination folder doesn't exist. Creating it...${NC}"
    mkdir -p "$DEST_FOLDER"
    echo -e "${GREEN}Created: $DEST_FOLDER${NC}"
fi

# Construct new filename
NEW_FILENAME="2D_from_360_${CAMERA_NUM}-${PARTICIPANT}-${INSTRUMENT}-${DATE_FORMATTED}-${MODEL_FORMATTED}-chunk_${CHUNK}.mp4"
DEST_PATH="${DEST_FOLDER}/${NEW_FILENAME}"

echo ""
echo "========================================="
echo "Moving and Renaming Exported File"
echo "========================================="
echo -e "${BLUE}Drive: $DRIVE_NAME${NC}"
echo -e "${BLUE}Participant: $PARTICIPANT${NC}"
echo -e "${BLUE}Instrument: $INSTRUMENT${NC}"
echo -e "${BLUE}Date: $DATE${NC}"
echo -e "${BLUE}Camera: $CAMERA${NC}"
echo -e "${BLUE}Chunk: $CHUNK${NC}"
echo ""
echo "From:"
echo "  $EXPORTED_FILE"
echo ""
echo "To:"
echo "  $DEST_PATH"
echo ""

# Move and rename
mv "$EXPORTED_FILE" "$DEST_PATH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ File moved and renamed successfully!${NC}"
    echo ""

    # Show file info
    ls -lh "$DEST_PATH"
    echo ""

    # Verify file with ffprobe
    if command -v ffprobe &> /dev/null; then
        echo "Verifying file..."

        # Get resolution
        RESOLUTION=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$DEST_PATH")
        WIDTH=$(echo "$RESOLUTION" | cut -d',' -f1)
        HEIGHT=$(echo "$RESOLUTION" | cut -d',' -f2)

        # Get codec
        CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$DEST_PATH")

        # Get bitrate
        BITRATE=$(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 "$DEST_PATH" 2>/dev/null)
        if [ -n "$BITRATE" ] && [ "$BITRATE" != "N/A" ]; then
            BITRATE_MBPS=$(echo "scale=1; $BITRATE / 1000000" | bc -l)
        else
            BITRATE_MBPS="N/A"
        fi

        # Get FPS
        FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$DEST_PATH")
        FPS_DECIMAL=$(echo "scale=2; $FPS" | bc -l 2>/dev/null || echo "$FPS")

        # Get duration
        DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$DEST_PATH")
        DURATION_MIN=$(echo "scale=2; $DURATION / 60" | bc -l)

        echo ""
        echo "File Properties:"
        echo "  Resolution: ${WIDTH}x${HEIGHT}"
        echo "  Codec: $CODEC"
        echo "  Bitrate: ${BITRATE_MBPS} Mbps"
        echo "  Frame Rate: ${FPS_DECIMAL} fps"
        echo "  Duration: ${DURATION_MIN} minutes"
        echo ""

        # Check against requirements
        echo "Checking Requirements:"

        # Check resolution
        if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
            echo -e "  ${GREEN}✓ Resolution: 4K (3840x2160)${NC}"
        else
            echo -e "  ${YELLOW}⚠ Resolution: ${WIDTH}x${HEIGHT} (expected 3840x2160)${NC}"
        fi

        # Check codec
        if [ "$CODEC" = "h264" ]; then
            echo -e "  ${GREEN}✓ Codec: H.264${NC}"
        elif [ "$CODEC" = "hevc" ]; then
            echo -e "  ${YELLOW}⚠ Codec: H.265/HEVC (H.264 preferred for compatibility)${NC}"
        else
            echo -e "  ${RED}✗ Codec: $CODEC (expected H.264)${NC}"
        fi

        # Check bitrate
        if [ "$BITRATE_MBPS" != "N/A" ]; then
            if (( $(echo "$BITRATE_MBPS >= 70" | bc -l) )) && (( $(echo "$BITRATE_MBPS <= 100" | bc -l) )); then
                echo -e "  ${GREEN}✓ Bitrate: ${BITRATE_MBPS} Mbps (target: 80 Mbps)${NC}"
            elif (( $(echo "$BITRATE_MBPS >= 60" | bc -l) )); then
                echo -e "  ${YELLOW}⚠ Bitrate: ${BITRATE_MBPS} Mbps (acceptable, but 80 Mbps recommended)${NC}"
            else
                echo -e "  ${RED}✗ Bitrate: ${BITRATE_MBPS} Mbps (too low, should be 80 Mbps)${NC}"
            fi
        else
            echo -e "  ${YELLOW}⚠ Bitrate: N/A (variable bitrate)${NC}"
        fi

        echo ""
    fi

    echo -e "${GREEN}All done!${NC}"
    echo ""
else
    echo -e "${RED}✗ ERROR: Failed to move file${NC}"
    exit 1
fi
