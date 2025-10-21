#!/bin/bash

################################################################################
# Export Quality Verification Script
#
# Purpose: Verify exported 2D video meets detection requirements
#
# Usage: ./verify_export_quality.sh <video_file>
#
# Example: ./verify_export_quality.sh "2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_020.mp4"
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if ffprobe is installed
if ! command -v ffprobe &> /dev/null; then
    echo -e "${RED}ERROR: ffprobe is not installed.${NC}"
    echo "Please install it: brew install ffmpeg"
    exit 1
fi

# Parse arguments
VIDEO_FILE="$1"

if [ -z "$VIDEO_FILE" ]; then
    echo "Usage: $0 <video_file>"
    echo ""
    echo "Example:"
    echo "  $0 \"2D_from_360_Camera_1-Quan-Violin-20250725-X5_1-chunk_020.mp4\""
    echo ""
    exit 1
fi

# Check if file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${RED}ERROR: File not found: $VIDEO_FILE${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "Export Quality Verification"
echo "========================================="
echo -e "${BLUE}File: $(basename "$VIDEO_FILE")${NC}"
echo ""

# Initialize pass/fail counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# Get video properties
echo "Analyzing video properties..."
echo ""

# Resolution
RESOLUTION=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$VIDEO_FILE")
WIDTH=$(echo "$RESOLUTION" | cut -d',' -f1)
HEIGHT=$(echo "$RESOLUTION" | cut -d',' -f2)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. RESOLUTION CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Resolution: ${WIDTH}x${HEIGHT}"

if [ "$WIDTH" -ge 3840 ] && [ "$HEIGHT" -ge 2160 ]; then
    echo -e "   ${GREEN}✓ PASS${NC} - 4K resolution (optimal for detection)"
    PASS_COUNT=$((PASS_COUNT + 1))
elif [ "$WIDTH" -ge 1920 ] && [ "$HEIGHT" -ge 1080 ]; then
    echo -e "   ${YELLOW}⚠ WARN${NC} - 1080p resolution (acceptable, but 4K preferred)"
    WARN_COUNT=$((WARN_COUNT + 1))
else
    echo -e "   ${RED}✗ FAIL${NC} - Resolution too low (minimum 1920x1080)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi
echo ""

# Codec
CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$VIDEO_FILE")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. CODEC CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Codec: $CODEC"

if [ "$CODEC" = "h264" ]; then
    echo -e "   ${GREEN}✓ PASS${NC} - H.264 (best compatibility for OpenCV/MediaPipe)"
    PASS_COUNT=$((PASS_COUNT + 1))
elif [ "$CODEC" = "hevc" ] || [ "$CODEC" = "h265" ]; then
    echo -e "   ${YELLOW}⚠ WARN${NC} - H.265/HEVC (smaller files, but may have compatibility issues)"
    echo "   Recommendation: Use H.264 for primary detection camera"
    WARN_COUNT=$((WARN_COUNT + 1))
else
    echo -e "   ${RED}✗ FAIL${NC} - Unexpected codec: $CODEC"
    echo "   Use H.264 or H.265"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi
echo ""

# Frame Rate
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$VIDEO_FILE")
# Convert fraction to decimal
FPS_DECIMAL=$(echo "scale=2; $FPS" | bc -l 2>/dev/null || echo "$FPS")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. FRAME RATE CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Frame Rate: $FPS_DECIMAL fps"

# Check if FPS is close to 30 or 60
if (( $(echo "$FPS_DECIMAL >= 29" | bc -l) )) && (( $(echo "$FPS_DECIMAL <= 31" | bc -l) )); then
    echo -e "   ${GREEN}✓ PASS${NC} - 30fps (standard for detection)"
    PASS_COUNT=$((PASS_COUNT + 1))
elif (( $(echo "$FPS_DECIMAL >= 59" | bc -l) )) && (( $(echo "$FPS_DECIMAL <= 61" | bc -l) )); then
    echo -e "   ${GREEN}✓ PASS${NC} - 60fps (excellent for fast motion)"
    PASS_COUNT=$((PASS_COUNT + 1))
elif (( $(echo "$FPS_DECIMAL >= 23" | bc -l) )) && (( $(echo "$FPS_DECIMAL <= 26" | bc -l) )); then
    echo -e "   ${YELLOW}⚠ WARN${NC} - 24-25fps (cinematic, but not ideal for detection)"
    WARN_COUNT=$((WARN_COUNT + 1))
else
    echo -e "   ${RED}✗ FAIL${NC} - Unusual frame rate"
    echo "   Recommended: 30fps or 60fps"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi
echo ""

# Bitrate
BITRATE=$(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 "$VIDEO_FILE" 2>/dev/null)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. BITRATE CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -n "$BITRATE" ] && [ "$BITRATE" != "N/A" ]; then
    BITRATE_MBPS=$(echo "scale=1; $BITRATE / 1000000" | bc -l)
    echo "   Bitrate: ${BITRATE_MBPS} Mbps"

    # Check bitrate based on resolution
    if [ "$WIDTH" -ge 3840 ]; then
        # 4K
        if (( $(echo "$BITRATE_MBPS >= 50" | bc -l) )); then
            echo -e "   ${GREEN}✓ PASS${NC} - Sufficient bitrate for 4K"
            PASS_COUNT=$((PASS_COUNT + 1))
        elif (( $(echo "$BITRATE_MBPS >= 30" | bc -l) )); then
            echo -e "   ${YELLOW}⚠ WARN${NC} - Bitrate acceptable but could be higher (50-100 Mbps recommended)"
            WARN_COUNT=$((WARN_COUNT + 1))
        else
            echo -e "   ${RED}✗ FAIL${NC} - Bitrate too low for 4K (minimum 30 Mbps)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        # 1080p or lower
        if (( $(echo "$BITRATE_MBPS >= 20" | bc -l) )); then
            echo -e "   ${GREEN}✓ PASS${NC} - Sufficient bitrate for 1080p"
            PASS_COUNT=$((PASS_COUNT + 1))
        elif (( $(echo "$BITRATE_MBPS >= 10" | bc -l) )); then
            echo -e "   ${YELLOW}⚠ WARN${NC} - Bitrate acceptable but could be higher (25-40 Mbps recommended)"
            WARN_COUNT=$((WARN_COUNT + 1))
        else
            echo -e "   ${RED}✗ FAIL${NC} - Bitrate too low for 1080p (minimum 10 Mbps)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
else
    echo "   Bitrate: N/A (variable bitrate or not reported)"
    echo -e "   ${YELLOW}⚠ WARN${NC} - Cannot verify bitrate"
    WARN_COUNT=$((WARN_COUNT + 1))
fi
echo ""

# Duration
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO_FILE")
DURATION_MIN=$(echo "scale=2; $DURATION / 60" | bc -l)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. FILE INFO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Duration: ${DURATION_MIN} minutes"

FILE_SIZE=$(du -h "$VIDEO_FILE" | cut -f1)
echo "   File Size: $FILE_SIZE"
echo ""

# File integrity check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. FILE INTEGRITY CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Checking for corruption..."

if ffmpeg -v error -i "$VIDEO_FILE" -f null - 2>&1 | grep -q "error"; then
    echo -e "   ${RED}✗ FAIL${NC} - File may be corrupted"
    FAIL_COUNT=$((FAIL_COUNT + 1))
else
    echo -e "   ${GREEN}✓ PASS${NC} - No corruption detected"
    PASS_COUNT=$((PASS_COUNT + 1))
fi
echo ""

# Summary
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo ""
echo -e "${GREEN}Passed:${NC}  $PASS_COUNT"
echo -e "${YELLOW}Warnings:${NC} $WARN_COUNT"
echo -e "${RED}Failed:${NC}  $FAIL_COUNT"
echo ""

# Overall result
if [ $FAIL_COUNT -eq 0 ]; then
    if [ $WARN_COUNT -eq 0 ]; then
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ EXCELLENT - File meets all requirements${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "This file is optimal for detection work."
        exit 0
    else
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}⚠ ACCEPTABLE - File has minor issues${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "File should work for detection, but consider the warnings above."
        exit 0
    fi
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ NOT RECOMMENDED - File has issues${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Consider re-exporting with corrected settings."
    exit 1
fi
