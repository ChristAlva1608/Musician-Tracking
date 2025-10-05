"""
Real-time audio transcription using OpenAI's Whisper model
"""

import numpy as np
import cv2
import threading
import queue
import time
from typing import Optional, Callable, Dict, Any
import logging

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Whisper not available. Install with: pip install openai-whisper")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("❌ PyAudio not available. Install with: pip install pyaudio")


class WhisperRealtimeTranscriber:
    """
    Real-time audio transcription using OpenAI's Whisper model
    Supports both live audio input and video file audio extraction
    """
    
    def __init__(self, model_size: str = "base", language: str = "en"):
        """
        Initialize the Whisper transcriber
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            language: Language code for transcription ("en", "es", "fr", etc.)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper is not available. Install with: pip install openai-whisper")
        
        self.model_size = model_size
        self.language = language
        self.model = None
        self.is_running = False
        self.transcript_queue = queue.Queue()
        self.audio_buffer = queue.Queue()
        
        # Audio settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_duration = 5.0  # Process audio in 5-second chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Threading
        self.transcription_thread = None
        self.audio_thread = None
        
        # Callback for real-time transcript updates
        self.transcript_callback: Optional[Callable[[str, float], None]] = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """Load the Whisper model"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            self.logger.info("✅ Whisper model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def set_transcript_callback(self, callback: Callable[[str, float], None]):
        """
        Set callback function for real-time transcript updates
        
        Args:
            callback: Function that takes (transcript_text, timestamp) as arguments
        """
        self.transcript_callback = callback
    
    def extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio data as numpy array or None if extraction fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"❌ Could not open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            self.logger.info(f"📹 Video duration: {duration:.2f}s, FPS: {fps}")
            
            # Release video capture (we'll use ffmpeg for audio extraction)
            cap.release()
            
            # Use ffmpeg to extract audio (more reliable than OpenCV for audio)
            import subprocess
            import tempfile
            import os
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ar', str(self.sample_rate),  # Set sample rate
                '-ac', '1',  # Mono channel
                '-y',  # Overwrite output
                temp_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"❌ FFmpeg failed: {result.stderr}")
                os.unlink(temp_audio_path)
                return None
            
            # Load audio using Whisper's audio loading function
            audio_data = whisper.load_audio(temp_audio_path)
            
            # Clean up temporary file
            os.unlink(temp_audio_path)
            
            self.logger.info(f"✅ Extracted audio: {len(audio_data) / self.sample_rate:.2f}s")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting audio: {e}")
            return None
    
    def transcribe_audio_chunk(self, audio_data: np.ndarray, start_time: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Transcribe a chunk of audio data
        
        Args:
            audio_data: Audio data as numpy array
            start_time: Start time of this chunk in seconds
            
        Returns:
            Transcription result dictionary or None
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_data,
                language=self.language if self.language != "auto" else None,
                word_timestamps=True
            )
            
            # Add absolute timestamps
            if "segments" in result:
                for segment in result["segments"]:
                    segment["start"] += start_time
                    segment["end"] += start_time
                    if "words" in segment:
                        for word in segment["words"]:
                            word["start"] += start_time
                            word["end"] += start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return None
    
    def transcribe_video_file(self, video_path: str, chunk_duration: float = 30.0, output_path: Optional[str] = None,
                             start_time_offset: float = 0.0, end_time_offset: Optional[float] = None) -> list:
        """
        Transcribe video file in chunks for better performance

        Args:
            video_path: Path to video file
            chunk_duration: Duration of each processing chunk in seconds
            output_path: Optional path to save video with embedded transcript
            start_time_offset: Start time offset in seconds (for video alignment)
            end_time_offset: End time offset in seconds (None for entire video)

        Returns:
            List of transcription results
        """
        # Extract audio from video
        audio_data = self.extract_audio_from_video(video_path)
        if audio_data is None:
            return []

        # Calculate sample offsets based on time offsets
        start_sample = int(start_time_offset * self.sample_rate)
        total_samples = len(audio_data)

        if end_time_offset is not None:
            end_sample = min(int(end_time_offset * self.sample_rate), total_samples)
        else:
            end_sample = total_samples

        # Trim audio data to the specified range
        if start_sample >= total_samples:
            self.logger.warning(f"⚠️ Start offset {start_time_offset:.1f}s exceeds audio duration")
            return []

        audio_data = audio_data[start_sample:end_sample]
        effective_samples = len(audio_data)

        self.logger.info(f"🔄 Processing audio from {start_time_offset:.1f}s to {end_time_offset or 'end'}s in {chunk_duration}s chunks...")

        # Process audio in chunks
        chunk_samples = int(self.sample_rate * chunk_duration)
        results = []

        for i in range(0, effective_samples, chunk_samples):
            # Calculate actual start time including the offset
            chunk_start_time = start_time_offset + (i / self.sample_rate)
            end_sample_idx = min(i + chunk_samples, effective_samples)
            chunk = audio_data[i:end_sample_idx]

            chunk_end_time = start_time_offset + (end_sample_idx / self.sample_rate)
            self.logger.info(f"📝 Transcribing chunk {chunk_start_time:.1f}s - {chunk_end_time:.1f}s")

            result = self.transcribe_audio_chunk(chunk, chunk_start_time)
            if result:
                results.append(result)

                # Call callback if provided
                if self.transcript_callback and result.get("text"):
                    self.transcript_callback(result["text"].strip(), chunk_start_time)
        
        # Create output video with embedded transcript if requested
        if output_path and results:
            self.logger.info(f"🎬 Creating output video with embedded transcript: {output_path}")
            success = self.create_video_with_transcript(video_path, results, output_path)
            if success:
                self.logger.info(f"✅ Video with transcript saved: {output_path}")
            else:
                self.logger.error(f"❌ Failed to create output video: {output_path}")
        
        return results
    
    def start_realtime_transcription(self, audio_source: str = "microphone"):
        """
        Start real-time transcription from microphone or video file
        
        Args:
            audio_source: "microphone" or path to video file
        """
        if not PYAUDIO_AVAILABLE and audio_source == "microphone":
            self.logger.error("❌ PyAudio not available for microphone input")
            return False
        
        if self.is_running:
            self.logger.warning("⚠️ Transcription already running")
            return False
        
        if not self.load_model():
            return False
        
        self.is_running = True
        
        if audio_source == "microphone":
            self._start_microphone_capture()
        else:
            self._start_video_transcription(audio_source)
        
        # Start transcription processing thread
        self.transcription_thread = threading.Thread(target=self._transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        return True
    
    def _start_microphone_capture(self):
        """Start capturing audio from microphone"""
        def audio_worker():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            audio_buffer = []
            
            while self.is_running:
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    audio_buffer.extend(np.frombuffer(data, dtype=np.float32))
                    
                    # Process when we have enough audio
                    if len(audio_buffer) >= self.chunk_size:
                        chunk = np.array(audio_buffer[:self.chunk_size])
                        self.audio_buffer.put((chunk, time.time()))
                        audio_buffer = audio_buffer[self.chunk_size:]
                        
                except Exception as e:
                    self.logger.error(f"❌ Audio capture error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
        
        self.audio_thread = threading.Thread(target=audio_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def _start_video_transcription(self, video_path: str):
        """Start transcribing video file in real-time simulation"""
        def video_worker():
            audio_data = self.extract_audio_from_video(video_path)
            if audio_data is None:
                return
            
            # Simulate real-time by sending chunks at regular intervals
            start_time = time.time()
            
            for i in range(0, len(audio_data), self.chunk_size):
                if not self.is_running:
                    break
                
                chunk = audio_data[i:i + self.chunk_size]
                timestamp = i / self.sample_rate
                
                self.audio_buffer.put((chunk, timestamp))
                
                # Wait for real-time simulation
                expected_time = start_time + timestamp + self.chunk_duration
                current_time = time.time()
                
                if expected_time > current_time:
                    time.sleep(expected_time - current_time)
        
        self.audio_thread = threading.Thread(target=video_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def _transcription_worker(self):
        """Worker thread for processing transcription"""
        while self.is_running:
            try:
                # Get audio chunk with timeout
                audio_chunk, timestamp = self.audio_buffer.get(timeout=1.0)
                
                # Transcribe the chunk
                result = self.transcribe_audio_chunk(audio_chunk, timestamp)
                
                if result and result.get("text"):
                    transcript = result["text"].strip()
                    if transcript:  # Only process non-empty transcripts
                        self.transcript_queue.put((transcript, timestamp))
                        
                        # Call callback if provided
                        if self.transcript_callback:
                            self.transcript_callback(transcript, timestamp)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Transcription worker error: {e}")
    
    def get_latest_transcript(self) -> Optional[tuple]:
        """
        Get the latest transcript from the queue
        
        Returns:
            Tuple of (transcript_text, timestamp) or None
        """
        try:
            return self.transcript_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop_transcription(self):
        """Stop real-time transcription"""
        self.is_running = False
        
        if self.transcription_thread:
            self.transcription_thread.join(timeout=2.0)
        
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        
        self.logger.info("🛑 Transcription stopped")
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        # Common Whisper supported languages
        return [
            "auto", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", 
            "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", 
            "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", 
            "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", 
            "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", 
            "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", 
            "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", 
            "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", 
            "ha", "ba", "jw", "su"
        ]
    
    def create_video_with_transcript(self, video_path: str, transcription_results: list, output_path: str) -> bool:
        """
        Create a new video with embedded transcript text
        
        Args:
            video_path: Path to original video file
            transcription_results: List of transcription results from transcribe_video_file
            output_path: Path to save the output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"❌ Could not open video: {video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"📹 Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
            
            # Create transcript timeline from results
            transcript_timeline = self._build_transcript_timeline(transcription_results)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.logger.error(f"❌ Could not create output video writer: {output_path}")
                cap.release()
                return False
            
            self.logger.info(f"🎬 Processing {frame_count} frames...")
            
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current timestamp
                current_time = frame_number / fps
                
                # Get current transcript text for this timestamp
                current_transcript = self._get_transcript_at_time(transcript_timeline, current_time)
                
                # Add transcript overlay to frame
                if current_transcript:
                    frame = self._add_transcript_overlay(frame, current_transcript, current_time)
                
                # Write frame to output video
                out.write(frame)
                
                frame_number += 1
                
                # Progress indicator
                if frame_number % int(fps * 10) == 0:  # Every 10 seconds
                    progress = (frame_number / frame_count) * 100
                    self.logger.info(f"🔄 Progress: {progress:.1f}% ({frame_number}/{frame_count} frames)")
            
            # Cleanup
            cap.release()
            out.release()
            
            self.logger.info(f"✅ Successfully created video with transcript: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating video with transcript: {e}")
            return False
    
    def _build_transcript_timeline(self, transcription_results: list) -> list:
        """
        Build a timeline of transcript segments with timestamps
        
        Args:
            transcription_results: List of transcription results
            
        Returns:
            List of transcript segments with timing information
        """
        timeline = []
        
        for result in transcription_results:
            if "segments" in result:
                for segment in result["segments"]:
                    timeline.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                        "words": segment.get("words", [])
                    })
            else:
                # Fallback for results without segments
                if result.get("text"):
                    timeline.append({
                        "start": 0,
                        "end": float("inf"),
                        "text": result["text"].strip(),
                        "words": []
                    })
        
        # Sort by start time
        timeline.sort(key=lambda x: x["start"])
        return timeline
    
    def _get_transcript_at_time(self, timeline: list, timestamp: float) -> Optional[str]:
        """
        Get the transcript text that should be displayed at a given timestamp
        
        Args:
            timeline: List of transcript segments
            timestamp: Current timestamp in seconds
            
        Returns:
            Transcript text to display or None
        """
        current_segments = []
        
        for segment in timeline:
            if segment["start"] <= timestamp <= segment["end"]:
                current_segments.append(segment["text"])
        
        if current_segments:
            return " ".join(current_segments)
        
        return None
    
    def _add_transcript_overlay(self, frame: np.ndarray, transcript: str, timestamp: float) -> np.ndarray:
        """
        Add transcript text overlay to video frame
        
        Args:
            frame: OpenCV video frame
            transcript: Text to overlay
            timestamp: Current timestamp
            
        Returns:
            Frame with transcript overlay
        """
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay area
        overlay_height = int(height * 0.25)  # Use bottom 25% of frame
        overlay_y = height - overlay_height
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, overlay_y), (width, height), (0, 0, 0), -1)
        
        # Blend overlay with original frame
        alpha = 0.7  # Transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Format timestamp
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        time_str = f"[{mins:02d}:{secs:02d}]"
        
        # Prepare text
        text_lines = self._wrap_text(transcript, width - 40)  # Leave margin
        
        # Text styling
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)  # White text
        thickness = 2
        line_height = 30
        
        # Add timestamp
        cv2.putText(frame, time_str, (20, overlay_y + 30), font, font_scale * 0.8, (100, 255, 100), thickness - 1)
        
        # Add transcript text
        y_offset = overlay_y + 60
        for line in text_lines:
            if y_offset < height - 10:
                cv2.putText(frame, line, (20, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
        
        return frame
    
    def _wrap_text(self, text: str, max_width: int, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2) -> list:
        """
        Wrap text to fit within specified width
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            font: OpenCV font
            font_scale: Font scale
            thickness: Font thickness
            
        Returns:
            List of text lines
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width > max_width:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, add it anyway
                    lines.append(word)
                    current_line = ""
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        return lines