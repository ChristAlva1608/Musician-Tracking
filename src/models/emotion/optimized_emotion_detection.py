#!/usr/bin/env python3
"""
Optimized emotion detection for real-time applications
"""

import cv2
import torch
import numpy as np
import time
from ellzaf_ml.models import GhostFaceNetsV2
from ellzaf_ml.tools.load_pretrained import load_pretrained
import torch.nn.functional as F
import os
import mediapipe as mp

class OptimizedEmotionDetector:
    """
    Optimized emotion detector with performance improvements
    """
    def __init__(self, checkpoint_path, image_size=112, device='cpu', skip_frames=3):
        """
        Initialize optimized emotion detector
        
        Args:
            checkpoint_path: Path to pretrained model
            image_size: Input image size (smaller = faster)
            device: Device to use ('cpu' or 'cuda')
            skip_frames: Process every Nth frame (higher = faster)
        """
        self.device = device
        self.image_size = image_size
        self.skip_frames = skip_frames
        self.frame_count = 0
        
        # Emotion classes
        self.emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Initialize smaller model for speed
        self.model = GhostFaceNetsV2(
            image_size=image_size,
            num_classes=len(self.emotion_classes),
            width=0.5,  # Smaller model
            dropout=0.1
        )
        
        # Load pretrained weights
        if os.path.exists(checkpoint_path):
            print(f"Loading optimized model from {checkpoint_path}")
            load_pretrained(self.model, checkpoint_path, not_vit=True)
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize face detection with lower resolution
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # Use short-range model (faster)
            min_detection_confidence=0.3  # Lower threshold for speed
        )
        
        # Performance tracking
        self.fps_history = []
        self.last_fps_time = time.time()
        
        print(f"✅ Optimized emotion detector initialized (image_size={image_size}, skip_frames={skip_frames})")
    
    def preprocess_image_fast(self, face_roi):
        """
        Fast image preprocessing
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Resize to smaller size for speed
        face_resized = cv2.resize(face_rgb, (self.image_size, self.image_size))
        
        # Fast normalization
        face_tensor = torch.from_numpy(face_resized).float() / 255.0
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    def predict_emotion_fast(self, face_roi):
        """
        Fast emotion prediction
        """
        try:
            face_tensor = self.preprocess_image_fast(face_roi)
            
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                emotion_label = self.emotion_classes[predicted_idx.item()]
                confidence_score = confidence.item()
                
                return emotion_label, confidence_score
                
        except Exception as e:
            return "Unknown", 0.0
    
    def detect_faces_and_emotions_fast(self, img):
        """
        Fast face and emotion detection with frame skipping
        """
        self.frame_count += 1
        
        # Skip frames for better performance
        if self.frame_count % self.skip_frames != 0:
            return []
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        
        face_emotions = []
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    face_roi = img[y:y+height, x:x+width]
                    emotion, confidence = self.predict_emotion_fast(face_roi)
                    face_emotions.append(((x, y, width, height), emotion, confidence))
        
        return face_emotions
    
    def get_fps(self):
        """
        Calculate current FPS
        """
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_fps_time)
        self.last_fps_time = current_time
        
        # Keep last 10 FPS measurements
        self.fps_history.append(fps)
        if len(self.fps_history) > 10:
            self.fps_history.pop(0)
        
        return np.mean(self.fps_history)
    
    def get_emotion_color(self, emotion):
        """
        Get color for emotion visualization
        """
        emotion_colors = {
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Angry': (0, 0, 255),      # Red
            'Fear': (128, 0, 128),     # Purple
            'Surprise': (0, 255, 255), # Yellow
            'Disgust': (0, 128, 0),    # Dark Green
            'Neutral': (128, 128, 128), # Gray
            'Unknown': (64, 64, 64)    # Dark Gray
        }
        return emotion_colors.get(emotion, (128, 128, 128))

def test_optimized_emotion_detection():
    """
    Test optimized emotion detection
    """
    print("Testing Optimized GhostFaceNetsV2 emotion detection...")
    
    # Check available checkpoints
    checkpoint_dir = "checkpoints"
    available_checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.h5') and file.startswith('GN_'):
                available_checkpoints.append(os.path.join(checkpoint_dir, file))
    
    if not available_checkpoints:
        print("❌ No checkpoints found")
        return
    
    checkpoint_path = available_checkpoints[0]
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize optimized detector
    try:
        detector = OptimizedEmotionDetector(
            checkpoint_path=checkpoint_path,
            image_size=112,  # Smaller for speed
            device=device,
            skip_frames=2    # Process every 2nd frame
        )
        
        print("✅ Optimized detector initialized!")
        
        # Test with webcam
        print("Opening webcam for testing...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("Press 'q' to quit, 's' to save image, '1-5' to change skip frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # Detect emotions
            face_emotions = detector.detect_faces_and_emotions_fast(frame)
            
            # Draw results
            for (x, y, w, h), emotion, confidence in face_emotions:
                color = detector.get_emotion_color(emotion)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(display_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display FPS and settings
            fps = detector.get_fps()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Skip Frames: {detector.skip_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Image Size: {detector.image_size}x{detector.image_size}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Optimized Emotion Detection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("optimized_test_frame.jpg", frame)
                print("✅ Test image saved")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                detector.skip_frames = int(chr(key))
                print(f"Skip frames set to {detector.skip_frames}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def benchmark_performance():
    """
    Benchmark different configurations
    """
    print("Benchmarking emotion detection performance...")
    
    checkpoint_path = "checkpoints/GN_W0.5_S2_ArcFace_epoch16.h5"  # Use smaller model
    
    if not os.path.exists(checkpoint_path):
        print("❌ Checkpoint not found")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different configurations
    configs = [
        {"image_size": 224, "skip_frames": 1, "width": 1.0, "name": "Original"},
        {"image_size": 112, "skip_frames": 1, "width": 1.0, "name": "Small Image"},
        {"image_size": 112, "skip_frames": 2, "width": 1.0, "name": "Skip Frames"},
        {"image_size": 112, "skip_frames": 2, "width": 0.5, "name": "Small Model"},
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        
        try:
            detector = OptimizedEmotionDetector(
                checkpoint_path=checkpoint_path,
                image_size=config['image_size'],
                device=device,
                skip_frames=config['skip_frames']
            )
            
            # Test with dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                detector.detect_faces_and_emotions_fast(dummy_image)
            
            # Benchmark
            start_time = time.time()
            for _ in range(30):
                detector.detect_faces_and_emotions_fast(dummy_image)
            end_time = time.time()
            
            fps = 30 / (end_time - start_time)
            print(f"  FPS: {fps:.1f}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_performance()
    else:
        test_optimized_emotion_detection() 