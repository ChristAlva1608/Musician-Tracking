"""
Consolidated test script for audio transcription with video output
Extracts transcript from video and creates output video with embedded transcript
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import whisper
        print("✅ OpenAI Whisper is available")
    except ImportError:
        missing_deps.append("openai-whisper")
        print("❌ OpenAI Whisper not available")
    
    try:
        import cv2
        print("✅ OpenCV is available")
    except ImportError:
        missing_deps.append("opencv-python")
        print("❌ OpenCV not available")
    
    if missing_deps:
        print(f"\n📦 Missing dependencies: {', '.join(missing_deps)}")
        print("\n💡 Install missing dependencies with:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        
        if "pyaudio" in missing_deps:
            print("\n⚠️  For PyAudio on macOS, you may also need:")
            print("   brew install portaudio")
        
        return False
    
    return True

def create_transcript_video(input_video: str, output_video: str, model_size: str = "base", language: str = "en"):
    """
    Create video with embedded transcript
    
    Args:
        input_video: Path to input video file
        output_video: Path to save output video with transcript
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (en, es, fr, etc. or "auto")
        
    Returns:
        True if successful, False otherwise
    """
    if not check_dependencies():
        return False
    
    from src.models.transcript.whisper_realtime import WhisperRealtimeTranscriber
    
    print("🎬 Creating Video with Embedded Transcript")
    print("=" * 50)
    print(f"📹 Input: {input_video}")
    print(f"💾 Output: {output_video}")
    print(f"🤖 Model: {model_size}")
    print(f"🌍 Language: {language}")
    
    # Check input file
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return False
    
    # Create output directory
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    
    try:
        # Initialize transcriber
        print(f"\n🔄 Loading Whisper model ({model_size})...")
        transcriber = WhisperRealtimeTranscriber(model_size=model_size, language=language)
        
        # Progress callback
        def progress_callback(transcript: str, timestamp: float):
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            display_text = transcript[:60] + "..." if len(transcript) > 60 else transcript
            print(f"📝 [{mins:02d}:{secs:02d}] {display_text}")
        
        transcriber.set_transcript_callback(progress_callback)
        
        # Process video
        print("\n🚀 Starting transcription and video creation...")
        start_time = time.time()
        
        results = transcriber.transcribe_video_file(
            video_path=input_video,
            chunk_duration=15.0,
            output_path=output_video
        )
        
        processing_time = time.time() - start_time
        
        if results and os.path.exists(output_video):
            print(f"\n✅ Successfully created video with transcript!")
            print(f"⏱️  Processing time: {processing_time:.1f} seconds")
            print(f"📊 Transcribed {len(results)} chunks")
            
            # File sizes
            input_size = os.path.getsize(input_video) / (1024 * 1024)
            output_size = os.path.getsize(output_video) / (1024 * 1024)
            print(f"📏 Input size: {input_size:.1f} MB")
            print(f"📏 Output size: {output_size:.1f} MB")
            print(f"💾 Output saved to: {output_video}")
            
            return True
        else:
            print("❌ Failed to create video with transcript")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Create video with embedded transcript')
    parser.add_argument('input_video', nargs='?', 
                       default='/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/video/multi-cam video/vid_shot1/original_video/cam_1.mp4',
                       help='Path to input video file')
    parser.add_argument('output_video', nargs='?',
                       default='/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/output/cam_1_with_transcript.mp4',
                       help='Path to save output video with transcript')
    parser.add_argument('--model-size', default='tiny',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: tiny for faster processing)')
    parser.add_argument('--language', default='en',
                       help='Language code (default: en, use "auto" for auto-detection)')
    
    args = parser.parse_args()
    
    print("🎵 Musician Tracking - Transcript Video Creator")
    print("=" * 50)
    
    # Show usage examples
    print("\n📚 Usage Examples:")
    print("   python test/test_transcript.py")
    print("   python test/test_transcript.py input.mp4 output.mp4")
    print("   python test/test_transcript.py input.mp4 output.mp4 --model-size base --language en")
    
    print("\n🎬 Features:")
    print("   - Extracts audio from video")
    print("   - Transcribes speech using OpenAI Whisper")
    print("   - Embeds transcript text in video frames")
    print("   - Semi-transparent overlay at bottom of video")
    print("   - Synchronized timestamps")
    print("   - Automatic text wrapping")
    
    # Create transcript video
    success = create_transcript_video(
        input_video=args.input_video,
        output_video=args.output_video,
        model_size=args.model_size,
        language=args.language
    )
    
    if success:
        print(f"\n🎉 Success! Your video with transcript is ready!")
        print(f"🎬 Play the video to see the embedded transcript:")
        print(f"   {args.output_video}")
    else:
        print(f"\n❌ Failed to create video with transcript")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)