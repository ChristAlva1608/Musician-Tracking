#!/usr/bin/env python3
"""
FER MTCNN Video Emotion Analysis
Uses FER (Facial Emotion Recognition) with MTCNN for video emotion analysis with chart generation
"""

import cv2
import numpy as np
import os
import time
from datetime import timedelta
from collections import defaultdict

# Try to import FER
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("FER not available. Install with: pip install fer")

# Import chart functions from video_emotion_analysis
from video_emotion_analysis import create_emotion_chart

# Emotion classes (FER standard)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion colors for visualization
EMOTION_COLORS = {
    'angry': '#FF0000',      # Red
    'disgust': '#8B4513',    # Brown
    'fear': '#800080',       # Purple
    'happy': '#00FF00',      # Green
    'sad': '#0000FF',        # Blue
    'surprise': '#FFFF00',   # Yellow
    'neutral': '#808080'     # Gray
}

class FERMTCNNVideoAnalyzer:
    """FER MTCNN video emotion analyzer"""
    
    def __init__(self, model_name='fer2013'):
        """
        Initialize FER MTCNN video analyzer
        
        Args:
            model_name: Emotion model to use ('fer2013', 'emotion', etc.)
        """
        if not FER_AVAILABLE:
            raise ImportError("FER is not available. Install with: pip install fer")
        
        self.model_name = model_name
        self.emotion_classes = EMOTIONS
        
        # Initialize FER with MTCNN detector
        self.detector = FER(mtcnn=True)
        
        print(f"✅ FER MTCNN video analyzer initialized")
        print(f"  Model: {model_name}")
        print(f"  Detector: MTCNN")
    
    def predict_emotion_from_array(self, image_array):
        """
        Predict emotion from numpy array
        
        Args:
            image_array: Numpy array of image
            
        Returns:
            List of emotion predictions
        """
        try:
            # Detect emotions using FER
            emotions = self.detector.detect_emotions(image_array)
            
            if not emotions:
                return []
            
            # Convert FER format to our format
            results = []
            for emotion_data in emotions:
                # Get bounding box
                bbox = emotion_data['box']
                x, y, w, h = bbox
                
                # Get emotions dictionary
                emotion_scores = emotion_data['emotions']
                
                # Find dominant emotion
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                
                results.append({
                    'bbox': (x, y, w, h),
                    'emotions': emotion_scores,
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence
                })
            
            return results
                
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return []
    
    def analyze_video(self, video_path, interval_seconds=30, sample_rate=1):
        """
        Analyze emotions in video
        
        Args:
            video_path: Path to video file
            interval_seconds: Time interval for chart (default: 30 seconds)
            sample_rate: Process every Nth frame (default: 1 = every frame)
        
        Returns:
            Dictionary with time series data
        """
        print(f"Analyzing video with FER MTCNN: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video properties:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds ({timedelta(seconds=int(duration))})")
        print(f"  Sample rate: Every {sample_rate} frame(s)")
        
        # Initialize data structures
        emotion_data = defaultdict(list)  # emotion -> list of (timestamp, confidence)
        frame_count = 0
        processed_frames = 0
        
        # Process frames
        print("\nProcessing video frames...")
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame based on sample rate
            if frame_count % sample_rate == 0:
                # Calculate timestamp
                timestamp = frame_count / fps
                
                try:
                    # Predict emotions
                    emotions = self.predict_emotion_from_array(frame)
                    
                    # Record emotions (use the most confident prediction if multiple faces)
                    if emotions:
                        # Sort by confidence and take the highest
                        emotions.sort(key=lambda x: x['confidence'], reverse=True)
                        best_emotion_data = emotions[0]
                        
                        dominant_emotion = best_emotion_data['dominant_emotion']
                        confidence = best_emotion_data['confidence']
                        
                        emotion_data[dominant_emotion].append((timestamp, confidence))
                    else:
                        # No face detected - record as neutral with low confidence
                        emotion_data['neutral'].append((timestamp, 0.1))
                    
                    processed_frames += 1
                    
                    # Progress update
                    if processed_frames % 10 == 0:
                        elapsed = time.time() - start_time
                        progress = (frame_count / total_frames) * 100
                        print(f"  Progress: {progress:.1f}% ({processed_frames} frames processed in {elapsed:.1f}s)")
                
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
        
        cap.release()
        
        total_time = time.time() - start_time
        print(f"\n✅ Video analysis completed in {total_time:.1f} seconds")
        print(f"  Processed {processed_frames} frames")
        print(f"  Average processing speed: {processed_frames/total_time:.1f} FPS")
        
        return {
            'emotion_data': dict(emotion_data),
            'video_duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': processed_frames
        }

def analyze_video_file(video_path, interval_seconds=30, sample_rate=1, output_path=None):
    """Complete FER MTCNN video analysis workflow"""
    print("FER MTCNN Video Emotion Analysis")
    print("=" * 50)
    
    if not FER_AVAILABLE:
        print("❌ FER not available. Install with: pip install fer")
        return
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Initialize analyzer
    try:
        analyzer = FERMTCNNVideoAnalyzer(
            model_name='fer2013'
        )
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Analyze video
    analysis_result = analyzer.analyze_video(
        video_path=video_path,
        interval_seconds=interval_seconds,
        sample_rate=sample_rate
    )
    
    if analysis_result is None:
        return
    
    # Generate chart
    if output_path is None:
        output_path = f"fer_mtcnn_emotion_analysis_{os.path.splitext(os.path.basename(video_path))[0]}.png"
    
    print(f"\nGenerating emotion chart...")
    create_emotion_chart(analysis_result, interval_seconds, output_path, 
                        emotion_classes=EMOTIONS, emotion_colors=EMOTION_COLORS, 
                        title_prefix="FER MTCNN")
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"  Video duration: {analysis_result['video_duration']:.1f} seconds")
    print(f"  Total detections: {sum(len(timestamps) for timestamps in analysis_result['emotion_data'].values())}")
    print(f"  Chart saved to: {output_path}")
    
    # Print emotion breakdown
    emotion_counts = {emotion: len(timestamps) for emotion, timestamps in analysis_result['emotion_data'].items()}
    total_detections = sum(emotion_counts.values())
    
    if total_detections > 0:
        print(f"\nEmotion breakdown:")
        for emotion in EMOTIONS:
            if emotion in emotion_counts:
                percentage = (emotion_counts[emotion] / total_detections) * 100
                print(f"  {emotion}: {emotion_counts[emotion]} ({percentage:.1f}%)")

if __name__ == "__main__":
    print("FER MTCNN Video Emotion Analysis")
    print("=" * 50)
    
    if not FER_AVAILABLE:
        print("❌ FER not available")
        print("Install with: pip install fer")
        exit(1)
    
    print("✓ FER is available")
    
    # Get video path
    video_path = input("\nEnter video file path: ").strip()
    
    if not video_path:
        print("No video path provided. Exiting.")
        exit(1)
    
    # Get analysis parameters
    try:
        interval_seconds = int(input("Enter time interval in seconds (default: 30): ") or "30")
        sample_rate = int(input("Enter sample rate (process every Nth frame, default: 1): ") or "1")
    except ValueError:
        print("Invalid input. Using defaults.")
        interval_seconds = 30
        sample_rate = 1
    
    # Get output path
    output_path = input("Enter output chart path (press Enter for auto): ").strip()
    if not output_path:
        output_path = None
    
    # Run analysis
    analyze_video_file(
        video_path=video_path,
        interval_seconds=interval_seconds,
        sample_rate=sample_rate,
        output_path=output_path
    ) 