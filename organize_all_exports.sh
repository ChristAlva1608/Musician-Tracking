#!/bin/bash

################################################################################
# Organize All Exported 360° Files
#
# Purpose:
# 1. Find all exported MP4 files in Phan Dissertation Data root
# 2. Match them to original .insv files
# 3. Verify duration is proportional (exported ~= original × compression ratio)
# 4. Move to correct folder with proper naming
# 5. Document all movements in a log file
#
# Usage: ./organize_all_exports.sh <participant> <instrument> <date> <drive>
#
# Example: ./organize_all_exports.sh "Quan" "Violin" "2025-07-25" "MAML 26TB 1"
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
PARTICIPANT="$1"
INSTRUMENT="$2"
DATE="$3"
DRIVE="$4"

if [ -z "$PARTICIPANT" ] || [ -z "$INSTRUMENT" ] || [ -z "$DATE" ] || [ -z "$DRIVE" ]; then
    echo "Usage: $0 <participant> <instrument> <date> <drive>"
    echo ""
    echo "Example:"
    echo "  $0 \"Quan\" \"Violin\" \"2025-07-25\" \"MAML 26TB 1\""
    echo ""
    echo "Available drives: \"MAML 26TB 1\", \"MAML 26TB 2\", \"X10 Pro 1\""
    exit 1
fi

# Check if ffprobe is available
if ! command -v ffprobe &> /dev/null; then
    echo -e "${RED}ERROR: ffprobe is not installed.${NC}"
    echo "Please install it: brew install ffmpeg"
    exit 1
fi

# Set base paths
BASE_PATH="/Volumes/${DRIVE}/Phan Dissertation Data"
PARTICIPANT_FOLDER="${PARTICIPANT} - MultiCam Data - ${INSTRUMENT} - ${DATE}"
PARTICIPANT_PATH="${BASE_PATH}/${PARTICIPANT_FOLDER}"

# Check if participant folder exists
if [ ! -d "$PARTICIPANT_PATH" ]; then
    echo -e "${RED}ERROR: Participant folder not found:${NC}"
    echo "  $PARTICIPANT_PATH"
    exit 1
fi

# Log file
LOG_FILE="${PARTICIPANT_PATH}/EXPORT_ORGANIZATION_LOG_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "========================================="
echo "360° Export Organization"
echo "========================================="
echo -e "${BLUE}Participant: $PARTICIPANT${NC}"
echo -e "${BLUE}Instrument: $INSTRUMENT${NC}"
echo -e "${BLUE}Date: $DATE${NC}"
echo -e "${BLUE}Drive: $DRIVE${NC}"
echo ""
echo "Working Directory:"
echo "  $PARTICIPANT_PATH"
echo ""
echo "Log file:"
echo "  $LOG_FILE"
echo ""

# Initialize log
cat > "$LOG_FILE" <<EOF
360° Export Organization Log
========================================
Participant: $PARTICIPANT
Instrument: $INSTRUMENT
Date: $DATE
Drive: $DRIVE
Timestamp: $(date)

========================================

EOF

# Find all MP4 files in root of Phan Dissertation Data
echo "Searching for exported MP4 files in root of Phan Dissertation Data..."
MP4_FILES=($(find "$BASE_PATH" -maxdepth 1 -name "*.mp4" -o -name "*.MP4" 2>/dev/null))

if [ ${#MP4_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}No MP4 files found in root of Phan Dissertation Data.${NC}"
    echo "Either files are already organized, or export hasn't finished yet."
    exit 0
fi

echo -e "${GREEN}Found ${#MP4_FILES[@]} exported file(s):${NC}"
for file in "${MP4_FILES[@]}"; do
    echo "  - $(basename "$file")"
done
echo ""

# Counter for processed files
PROCESSED=0
MOVED=0
SKIPPED=0

# Process each MP4 file
for EXPORTED_FILE in "${MP4_FILES[@]}"; do
    EXPORTED_BASENAME=$(basename "$EXPORTED_FILE")
    echo "========================================="
    echo -e "${CYAN}Processing: $EXPORTED_BASENAME${NC}"
    echo "========================================="

    # Try to match the filename to original .insv naming pattern
    # Expected pattern: VID_YYYYMMDD_HHMMSS_LL_CCC.mp4
    if [[ "$EXPORTED_BASENAME" =~ VID_([0-9]{8})_([0-9]{6})_([0-9]{2})_([0-9]{3})\.(mp4|MP4) ]]; then
        DATE_STR="${BASH_REMATCH[1]}"
        TIME_STR="${BASH_REMATCH[2]}"
        LENS_ID="${BASH_REMATCH[3]}"
        CHUNK_NUM="${BASH_REMATCH[4]}"

        echo "Parsed filename:"
        echo "  Date: $DATE_STR"
        echo "  Time: $TIME_STR"
        echo "  Lens: $LENS_ID"
        echo "  Chunk: $CHUNK_NUM"
        echo ""

        # Find matching original .insv file in participant folders
        echo "Looking for original .insv file..."
        ORIGINAL_INSV=$(find "$PARTICIPANT_PATH" -name "VID_${DATE_STR}_${TIME_STR}_${LENS_ID}_${CHUNK_NUM}.insv" 2>/dev/null | head -1)

        if [ -z "$ORIGINAL_INSV" ]; then
            echo -e "${YELLOW}⚠ Warning: Could not find original .insv file${NC}"
            echo "Searched for: VID_${DATE_STR}_${TIME_STR}_${LENS_ID}_${CHUNK_NUM}.insv"
            echo ""

            # Still try to process without verification
            ORIGINAL_DURATION="Unknown"
            CAMERA_FOLDER="Unknown"
        else
            echo -e "${GREEN}✓ Found original: $(basename "$ORIGINAL_INSV")${NC}"
            ORIGINAL_DIR=$(dirname "$ORIGINAL_INSV")
            CAMERA_FOLDER=$(basename "$ORIGINAL_DIR")
            echo "  Location: $CAMERA_FOLDER"

            # Get original file size to estimate duration
            ORIGINAL_SIZE=$(stat -f%z "$ORIGINAL_INSV" 2>/dev/null || stat -c%s "$ORIGINAL_INSV" 2>/dev/null)
            ORIGINAL_SIZE_GB=$(echo "scale=2; $ORIGINAL_SIZE / 1073741824" | bc -l)
            ORIGINAL_DURATION_MIN=$(echo "scale=2; ($ORIGINAL_SIZE_GB / 20) * 30" | bc -l)
            ORIGINAL_DURATION_SEC=$(echo "scale=2; $ORIGINAL_DURATION_MIN * 60" | bc -l)

            echo "  Original size: ${ORIGINAL_SIZE_GB} GB"
            echo "  Estimated duration: ${ORIGINAL_DURATION_MIN} minutes"
            echo ""
        fi

        # Get exported file duration
        echo "Checking exported file properties..."
        EXPORTED_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$EXPORTED_FILE" 2>/dev/null)

        if [ -z "$EXPORTED_DURATION" ]; then
            echo -e "${RED}✗ Error: Could not read exported file duration${NC}"
            SKIPPED=$((SKIPPED + 1))
            echo "" | tee -a "$LOG_FILE"
            echo "SKIPPED: $EXPORTED_BASENAME - Could not read duration" >> "$LOG_FILE"
            echo "" >> "$LOG_FILE"
            continue
        fi

        EXPORTED_DURATION_MIN=$(echo "scale=2; $EXPORTED_DURATION / 60" | bc -l)
        EXPORTED_SIZE=$(stat -f%z "$EXPORTED_FILE" 2>/dev/null || stat -c%s "$EXPORTED_FILE" 2>/dev/null)
        EXPORTED_SIZE_GB=$(echo "scale=2; $EXPORTED_SIZE / 1073741824" | bc -l)

        # Get video properties
        RESOLUTION=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$EXPORTED_FILE" 2>/dev/null)
        CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$EXPORTED_FILE" 2>/dev/null)
        BITRATE=$(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 "$EXPORTED_FILE" 2>/dev/null)

        if [ -n "$BITRATE" ] && [ "$BITRATE" != "N/A" ]; then
            BITRATE_MBPS=$(echo "scale=1; $BITRATE / 1000000" | bc -l)
        else
            BITRATE_MBPS="N/A"
        fi

        echo "Exported file properties:"
        echo "  Duration: ${EXPORTED_DURATION_MIN} minutes"
        echo "  Size: ${EXPORTED_SIZE_GB} GB"
        echo "  Resolution: $RESOLUTION"
        echo "  Codec: $CODEC"
        echo "  Bitrate: ${BITRATE_MBPS} Mbps"
        echo ""

        # Verify duration is proportional (allow 5% tolerance)
        if [ "$ORIGINAL_DURATION_SEC" != "Unknown" ]; then
            DURATION_DIFF=$(echo "scale=2; ($EXPORTED_DURATION - $ORIGINAL_DURATION_SEC) / $ORIGINAL_DURATION_SEC * 100" | bc -l)
            DURATION_DIFF_ABS=${DURATION_DIFF#-}  # Absolute value

            echo "Duration verification:"
            echo "  Original (estimated): ${ORIGINAL_DURATION_MIN} min"
            echo "  Exported: ${EXPORTED_DURATION_MIN} min"
            echo "  Difference: ${DURATION_DIFF}%"

            if (( $(echo "$DURATION_DIFF_ABS < 5" | bc -l) )); then
                echo -e "  ${GREEN}✓ Duration match: Within 5% tolerance${NC}"
            else
                echo -e "  ${YELLOW}⚠ Duration mismatch: More than 5% difference${NC}"
                echo -e "  ${YELLOW}  This might indicate incomplete export or wrong file${NC}"
            fi
            echo ""
        fi

        # Determine destination based on camera folder or chunk pattern
        if [ "$CAMERA_FOLDER" != "Unknown" ]; then
            # Extract camera info from folder name
            # Expected: "360 Camera X - Participant - Instrument - Date Model"
            if [[ "$CAMERA_FOLDER" =~ 360\ Camera\ ([0-9]+).*([A-Z0-9]+\ [0-9]+)$ ]]; then
                CAMERA_NUM="${BASH_REMATCH[1]}"
                MODEL="${BASH_REMATCH[2]}"

                DEST_FOLDER="${PARTICIPANT_PATH}/2D Converted - 360 Camera ${CAMERA_NUM} - ${PARTICIPANT} - ${INSTRUMENT} - ${DATE} ${MODEL}"
            else
                echo -e "${YELLOW}⚠ Warning: Could not parse camera folder name${NC}"
                DEST_FOLDER="${PARTICIPANT_PATH}/2D Converted - Unknown Camera"
            fi
        else
            # Guess based on lens ID
            # Typically: 00 = main lens, 01 = LRV (low-res video)
            if [ "$LENS_ID" = "00" ]; then
                DEST_FOLDER="${PARTICIPANT_PATH}/2D Converted - 360 Camera (Auto-detected)"
            else
                echo -e "${YELLOW}⚠ Warning: Unexpected lens ID: $LENS_ID${NC}"
                DEST_FOLDER="${PARTICIPANT_PATH}/2D Converted - Unknown Camera"
            fi
        fi

        # Create destination folder if it doesn't exist
        if [ ! -d "$DEST_FOLDER" ]; then
            echo "Creating destination folder..."
            mkdir -p "$DEST_FOLDER"
            echo -e "${GREEN}✓ Created: $(basename "$DEST_FOLDER")${NC}"
        fi

        # Generate new filename
        DATE_FORMATTED="${DATE//-/}"
        if [ "$CAMERA_FOLDER" != "Unknown" ]; then
            CAMERA_NUM_CLEAN=$(echo "$CAMERA_FOLDER" | grep -oE "Camera [0-9]+" | grep -oE "[0-9]+")
            MODEL_CLEAN=$(echo "$MODEL" | tr ' ' '_')
            NEW_FILENAME="2D_from_360_Camera_${CAMERA_NUM_CLEAN}-${PARTICIPANT}-${INSTRUMENT}-${DATE_FORMATTED}-${MODEL_CLEAN}-chunk_${CHUNK_NUM}.mp4"
        else
            NEW_FILENAME="2D_from_360-${PARTICIPANT}-${INSTRUMENT}-${DATE_FORMATTED}-chunk_${CHUNK_NUM}.mp4"
        fi

        DEST_PATH="${DEST_FOLDER}/${NEW_FILENAME}"

        echo "Moving and renaming file..."
        echo "  To: $NEW_FILENAME"
        echo ""

        # Move the file
        mv "$EXPORTED_FILE" "$DEST_PATH"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Successfully moved and renamed!${NC}"
            MOVED=$((MOVED + 1))

            # Log the movement
            cat >> "$LOG_FILE" <<EOF
FILE MOVED:
  Original name: $EXPORTED_BASENAME
  New name: $NEW_FILENAME
  Destination: $(basename "$DEST_FOLDER")

  Original .insv: $(basename "$ORIGINAL_INSV" 2>/dev/null || echo "Not found")
  Original duration (est): ${ORIGINAL_DURATION_MIN} min
  Exported duration: ${EXPORTED_DURATION_MIN} min
  Duration match: ${DURATION_DIFF}% difference

  Resolution: $RESOLUTION
  Codec: $CODEC
  Bitrate: ${BITRATE_MBPS} Mbps
  File size: ${EXPORTED_SIZE_GB} GB

  Status: ✓ SUCCESS

========================================

EOF
        else
            echo -e "${RED}✗ Error: Failed to move file${NC}"
            SKIPPED=$((SKIPPED + 1))

            # Log the error
            cat >> "$LOG_FILE" <<EOF
FILE SKIPPED:
  Original name: $EXPORTED_BASENAME
  Reason: Failed to move file
  Status: ✗ FAILED

========================================

EOF
        fi

    else
        echo -e "${YELLOW}⚠ Warning: Filename doesn't match expected pattern${NC}"
        echo "  Expected: VID_YYYYMMDD_HHMMSS_LL_CCC.mp4"
        echo "  Got: $EXPORTED_BASENAME"
        echo ""
        SKIPPED=$((SKIPPED + 1))

        # Log the skip
        cat >> "$LOG_FILE" <<EOF
FILE SKIPPED:
  Original name: $EXPORTED_BASENAME
  Reason: Filename doesn't match expected pattern
  Status: ✗ SKIPPED

========================================

EOF
    fi

    PROCESSED=$((PROCESSED + 1))
    echo ""
done

# Final summary
echo ""
echo "========================================="
echo "ORGANIZATION COMPLETE"
echo "========================================="
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "  Files processed: $PROCESSED"
echo -e "  ${GREEN}Files moved: $MOVED${NC}"
echo -e "  ${YELLOW}Files skipped: $SKIPPED${NC}"
echo ""
echo "Log file saved to:"
echo "  $LOG_FILE"
echo ""

# Append summary to log
cat >> "$LOG_FILE" <<EOF

========================================
SUMMARY
========================================
Files processed: $PROCESSED
Files moved: $MOVED
Files skipped: $SKIPPED

Completed: $(date)
EOF

if [ $MOVED -gt 0 ]; then
    echo -e "${GREEN}✓ All exported files have been organized!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review the log file for any warnings"
    echo "2. Verify a few exported files play correctly"
    echo "3. If all looks good, you can delete the original .insv files to save space"
else
    echo -e "${YELLOW}No files were moved. Check if exports are complete.${NC}"
fi

echo ""
