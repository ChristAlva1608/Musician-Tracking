#!/usr/bin/env python3
"""
GhostFaceNetsV2 Video Emotion Analysis
Uses GhostFaceNetsV2 models for video emotion analysis with chart generation
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import torch.nn.functional as F
from ellzaf_ml.models import GhostFaceNetsV2
from ellzaf_ml.tools.load_pretrained import load_pretrained
import time
from datetime import timedelta
from collections import defaultdict

# Import chart functions from video_emotion_analysis
from video_emotion_analysis import create_emotion_chart, get_available_checkpoints

# Emotion classes
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion colors for visualization
EMOTION_COLORS = {
    'Angry': '#FF0000',      # Red
    'Disgust': '#8B4513',    # Brown
    'Fear': '#800080',       # Purple
    'Happy': '#00FF00',      # Green
    'Sad': '#0000FF',        # Blue
    'Surprise': '#FFFF00',   # Yellow
    'Neutral': '#808080'     # Gray
}

class GhostFaceNetsV2VideoAnalyzer:
    """GhostFaceNetsV2 video emotion analyzer"""
    
    def __init__(self, checkpoint_path, image_size=224, device='cpu'):
        """
        Initialize GhostFaceNetsV2 video analyzer
        
        Args:
            checkpoint_path: Path to pretrained model checkpoint
            image_size: Input image size
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.image_size = image_size
        self.emotion_classes = EMOTIONS
        
        # Create GhostFaceNetsV2 model
        self.model = GhostFaceNetsV2(
            image_size=image_size,
            num_classes=len(self.emotion_classes),
            width=1.0,
            dropout=0.2
        )
        
        # Load pretrained weights
        if os.path.exists(checkpoint_path):
            print(f"Loading GhostFaceNetsV2 weights from {checkpoint_path}")
            try:
                load_pretrained(self.model, checkpoint_path, not_vit=True)
                print("✅ Weights loaded successfully")
            except Exception as e:
                print(f"❌ Error loading weights: {e}")
                print("Trying alternative loading method...")
                try:
                    load_pretrained(self.model, checkpoint_path, not_vit=False)
                    print("✅ Weights loaded with alternative method")
                except Exception as e2:
                    print(f"❌ Alternative loading also failed: {e2}")
                    raise Exception(f"Could not load checkpoint: {e}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
        # Move model to device and set evaluation mode
        self.model.to(device)
        self.model.eval()
        
        # Initialize face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        print(f"✅ GhostFaceNetsV2 video analyzer initialized (device: {device})")
    
    def preprocess_face_image(self, face_roi):
        """Preprocess face image for GhostFaceNetsV2"""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        face_resized = cv2.resize(face_rgb, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        face_tensor = torch.from_numpy(face_resized).float() / 255.0
        face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(self.device)
    
    def predict_emotion(self, face_roi):
        """Predict emotion from face image"""
        try:
            # Preprocess image
            face_tensor = self.preprocess_face_image(face_roi)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                emotion_label = self.emotion_classes[predicted_idx.item()]
                confidence_score = confidence.item()
                all_predictions = probabilities[0].cpu().numpy()
                
                return emotion_label, confidence_score, all_predictions
                
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "Unknown", 0.0, np.zeros(len(self.emotion_classes))
    
    def detect_faces_and_emotions(self, image):
        """Detect faces and predict emotions"""
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        face_emotions = []
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                # Extract face region
                face_roi = image[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # Predict emotion
                    emotion, confidence, all_predictions = self.predict_emotion(face_roi)
                    face_emotions.append(((x, y, w, h), emotion, confidence, all_predictions))
        
        return face_emotions
    
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
        print(f"Analyzing video with GhostFaceNetsV2: {video_path}")
        
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
                
                # Detect faces and emotions
                face_emotions = self.detect_faces_and_emotions(frame)
                
                # Record emotions (use the most confident prediction if multiple faces)
                if face_emotions:
                    # Sort by confidence and take the highest
                    face_emotions.sort(key=lambda x: x[2], reverse=True)
                    bbox, best_emotion, best_confidence, _ = face_emotions[0]
                    
                    emotion_data[best_emotion].append((timestamp, best_confidence))
                else:
                    # No face detected - record as neutral with low confidence
                    emotion_data['Neutral'].append((timestamp, 0.1))
                
                processed_frames += 1
                
                # Progress update
                if processed_frames % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    print(f"  Progress: {progress:.1f}% ({processed_frames} frames processed in {elapsed:.1f}s)")
        
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
    """Complete GhostFaceNetsV2 video analysis workflow"""
    print("GhostFaceNetsV2 Video Emotion Analysis")
    print("=" * 50)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Get available checkpoints
    checkpoints = get_available_checkpoints()
    
    if not checkpoints:
        print("❌ No GhostFaceNetsV2 checkpoints found")
        return
    
    # Use first checkpoint
    checkpoint_path = checkpoints[0]
    print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize analyzer
    try:
        analyzer = GhostFaceNetsV2VideoAnalyzer(
            checkpoint_path=checkpoint_path,
            image_size=224,
            device=device
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
        output_path = f"ghostfacenetsv2_emotion_analysis_{os.path.splitext(os.path.basename(video_path))[0]}.png"
    
    print(f"\nGenerating emotion chart...")
    create_emotion_chart(analysis_result, interval_seconds, output_path, 
                        emotion_classes=EMOTIONS, emotion_colors=EMOTION_COLORS, 
                        title_prefix="GhostFaceNetsV2")
    
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
    print("GhostFaceNetsV2 Video Emotion Analysis")
    print("=" * 50)
    
    # Show available checkpoints
    checkpoints = get_available_checkpoints()
    if checkpoints:
        print("✓ Found GhostFaceNetsV2 checkpoints:")
        for checkpoint in checkpoints:
            print(f"  - {os.path.basename(checkpoint)}")
    else:
        print("❌ No GhostFaceNetsV2 checkpoints found")
        exit(1)
    
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