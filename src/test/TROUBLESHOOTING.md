# Test Troubleshooting Guide

## Video Corruption Issues

If you encounter massive MPEG4 decoder errors like:
```
[mpeg4 @ 0x...] header damaged
[mpeg4 @ 0x...] Error at MB: ...
```

This indicates corrupted video files. Here's how to fix it:

### Solutions:

1. **Use Different Video Files**
   - Try with fresh, uncorrupted video files
   - Use videos from a different source/directory
   - Example: Use original video files instead of detection output videos

2. **Check Video File Integrity**
   ```bash
   # Test if video is readable
   ffmpeg -v error -i your_video.mp4 -f null -
   ```

3. **Re-encode Corrupted Videos**
   ```bash
   # Re-encode with FFmpeg
   ffmpeg -i corrupted_video.mp4 -c:v libx264 -crf 23 -c:a aac fixed_video.mp4
   ```

4. **Use Test Script with Validation**
   The updated test scripts now include video validation:
   ```bash
   python test/test_video_stacking.py
   python test/test_unified_video.py
   ```

## Common Test Issues

### 1. Configuration Errors
- **Error**: "expected str, bytes or os.PathLike object, not dict"
- **Solution**: Fixed in updated test scripts with proper config handling

### 2. Import Errors
- **Error**: ModuleNotFoundError
- **Solution**: Run from project root directory:
  ```bash
  cd "/Volumes/Extreme_Pro/Mitou Project/Musician Tracking"
  python test/test_video_stacking.py
  ```

### 3. Video Dimension Issues
- **Problem**: Mixed video dimensions causing stacking issues
- **Solution**: Test scripts now filter out unified videos and validate dimensions

### 4. Database Connection Issues
- **Solution**: Test scripts disable database by default for isolation

## Recommended Test Videos

For best results, use:
- Clean, uncorrupted MP4 files
- Similar resolutions (e.g., all 1920x1080)
- Reasonable duration (5-30 seconds)
- Consistent frame rates (e.g., all 30fps)

## Test Directory Structure

```
test/
├── test_video_stacking.py      # Direct video stacking test
├── test_unified_video.py       # Full unified video workflow test
└── output/                     # Test output directory
    ├── test_stacked_video.mp4
    └── test_unified_output.mp4
```

## Debugging Tips

1. **Enable Verbose Logging**
   - The test scripts now provide detailed validation output
   - Check each step of video processing

2. **Test with Minimal Data**
   - Start with 2-3 clean video files
   - Gradually increase complexity

3. **Check Output**
   - Verify output videos are created successfully
   - Use video players to check final results

4. **Monitor Resource Usage**
   - Large video processing can consume significant memory
   - Consider limiting video duration for testing