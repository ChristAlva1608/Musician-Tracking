#!/usr/bin/env python3
"""
GhostFaceNetsV2 Emotion Recognition Test
Uses pretrained GhostFaceNetsV2 models for emotion detection
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import torch.nn.functional as F
from ellzaf_ml.models import GhostFaceNetsV2
from ellzaf_ml.tools.load_pretrained import load_pretrained

# Emotion classes
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class GhostFaceNetsV2EmotionDetector:
    """GhostFaceNetsV2 emotion detector"""
    
    def __init__(self, checkpoint_path, image_size=224, device='cpu'):
        """
        Initialize GhostFaceNetsV2 emotion detector
        
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
            load_pretrained(self.model, checkpoint_path, not_vit=True)
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
        
        print(f"✅ GhostFaceNetsV2 emotion detector initialized (device: {device})")
    
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
            
            # Make prediction with timing
            import time
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            inference_time = time.time() - start_time
                
            emotion_label = self.emotion_classes[predicted_idx.item()]
            confidence_score = confidence.item()
            all_predictions = probabilities[0].cpu().numpy()
            
            return emotion_label, confidence_score, all_predictions, inference_time
                
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "Unknown", 0.0, np.zeros(len(self.emotion_classes)), 0.0
    
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
                    emotion, confidence, all_predictions, inference_time = self.predict_emotion(face_roi)
                    face_emotions.append(((x, y, w, h), emotion, confidence, all_predictions, inference_time))
        
        return face_emotions

def get_available_checkpoints():
    """Get list of available GhostFaceNetsV2 checkpoints"""
    checkpoint_dir = "checkpoints"
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if (file.endswith('.h5') and 
                ('GN_' in file or 'GhostFaceNet' in file) and 
                not file.startswith('._')):  # Filter out macOS metadata files
                checkpoints.append(os.path.join(checkpoint_dir, file))
    
    return checkpoints

def draw_emotion_results(image, face_emotions):
    """Draw emotion detection results on image"""
    emotion_colors = {
        'Happy': (0, 255, 0),      # Green
        'Sad': (255, 0, 0),        # Blue
        'Angry': (0, 0, 255),      # Red
        'Fear': (128, 0, 128),     # Purple
        'Surprise': (0, 255, 255), # Yellow
        'Disgust': (0, 128, 0),    # Dark Green
        'Neutral': (128, 128, 128) # Gray
    }
    
    for (x, y, w, h), emotion, confidence, all_predictions, inference_time in face_emotions:
        # Get color for emotion
        color = emotion_colors.get(emotion, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Draw emotion label, confidence and inference time
        text = f"{emotion}: {confidence:.2f} ({inference_time*1000:.0f}ms)"
        cv2.putText(image, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw confidence bar
        bar_width = int(w * confidence)
        cv2.rectangle(image, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
        cv2.rectangle(image, (x, y+h+5), (x+w, y+h+15), (255, 255, 255), 1)

def test_with_video(video_path):
    """Test GhostFaceNetsV2 with video file"""
    print("GhostFaceNetsV2 Emotion Recognition - Video Test")
    print("=" * 50)
    print(f"Testing with video: {video_path}")
    
    # Get available checkpoints
    checkpoints = get_available_checkpoints()
    
    if not checkpoints:
        print("❌ No GhostFaceNetsV2 checkpoints found in src/checkpoints/ directory")
        print("Available files:")
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                print(f"  - {file}")
        return
    
    # Choose checkpoint
    print("Available GhostFaceNetsV2 checkpoints:")
    for i, checkpoint in enumerate(checkpoints):
        print(f"  {i+1}. {os.path.basename(checkpoint)}")
    
    choice = input(f"\nChoose checkpoint (1-{len(checkpoints)}): ").strip()
    
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(checkpoints):
        print("Invalid choice. Using first checkpoint.")
        checkpoint_path = checkpoints[0]
    else:
        checkpoint_path = checkpoints[int(choice) - 1]
    
    print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize detector
    try:
        detector = GhostFaceNetsV2EmotionDetector(
            checkpoint_path=checkpoint_path,
            image_size=224,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Start video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
    print("\nStarting video test...")
    print("Press 'q' to quit, 's' to save image")
    
    frame_count = 0
    total_inference_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces and emotions
        face_emotions = detector.detect_faces_and_emotions(frame)
        
        # Calculate total inference time for this frame
        frame_inference_time = sum(face_data[4] for face_data in face_emotions) if face_emotions else 0
        total_inference_time += frame_inference_time
        
        # Print periodic stats
        if frame_count % 30 == 0 or len(face_emotions) > 0:
            avg_inference = (total_inference_time / frame_count) * 1000 if frame_count > 0 else 0
            print(f"Frame {frame_count}: {len(face_emotions)} faces, {frame_inference_time*1000:.1f}ms inference, avg: {avg_inference:.1f}ms")
        
        # Draw results
        draw_emotion_results(frame, face_emotions)
        
        # Add frame counter and timing info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Inference: {frame_inference_time*1000:.0f}ms", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('GhostFaceNetsV2 Emotion Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"emotion_test_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")
    
    # Print final statistics
    avg_inference = (total_inference_time / frame_count) * 1000 if frame_count > 0 else 0
    print(f"\n📊 Test Results:")
    print(f"Processed {frame_count} frames")
    print(f"Average inference time: {avg_inference:.1f}ms per frame")
    print(f"Total inference time: {total_inference_time:.1f}s")
    
    cap.release()
    cv2.destroyAllWindows()

def test_with_image(image_path):
    """Test GhostFaceNetsV2 with a single image"""
    print(f"GhostFaceNetsV2 Emotion Recognition - Image Test")
    print("=" * 50)
    print(f"Testing with image: {image_path}")
    
    # Get available checkpoints
    checkpoints = get_available_checkpoints()
    
    if not checkpoints:
        print("❌ No GhostFaceNetsV2 checkpoints found")
        return
    
    # Use first checkpoint for image test
    checkpoint_path = checkpoints[0]
    print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize detector
    try:
        detector = GhostFaceNetsV2EmotionDetector(
            checkpoint_path=checkpoint_path,
            image_size=224,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # Detect faces and emotions
    face_emotions = detector.detect_faces_and_emotions(image)
    
    if not face_emotions:
        print("No faces detected in the image")
        return
    
    # Print results
    print(f"\nDetected {len(face_emotions)} face(s):")
    for i, ((x, y, w, h), emotion, confidence, all_predictions, inference_time) in enumerate(face_emotions):
        print(f"\nFace {i+1} (at {x},{y}):")
        print(f"  Emotion: {emotion}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Inference time: {inference_time*1000:.1f}ms")
        print("  All predictions:")
        for emotion_name, pred in zip(EMOTIONS, all_predictions):
            print(f"    {emotion_name}: {pred:.3f}")
    
    # Draw results
    draw_emotion_results(image, face_emotions)
    
    # Display result
    cv2.imshow('GhostFaceNetsV2 Emotion Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def benchmark_performance():
    """Benchmark GhostFaceNetsV2 performance"""
    print("GhostFaceNetsV2 Performance Benchmark")
    print("=" * 40)
    
    checkpoints = get_available_checkpoints()
    if not checkpoints:
        print("❌ No checkpoints found")
        return
    
    checkpoint_path = checkpoints[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Testing with: {os.path.basename(checkpoint_path)} on {device}")
    
    try:
        detector = GhostFaceNetsV2EmotionDetector(
            checkpoint_path=checkpoint_path,
            image_size=224,
            device=device
        )
        
        # Create dummy face image
        dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Warm up
        print("Warming up...")
        for _ in range(5):
            detector.predict_emotion(dummy_face)
        
        # Benchmark
        import time
        num_tests = 50
        start_time = time.time()
        
        for _ in range(num_tests):
            detector.predict_emotion(dummy_face)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_tests
        fps = 1.0 / avg_time
        
        print(f"\nPerformance Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per prediction: {avg_time*1000:.1f}ms")
        print(f"  FPS: {fps:.1f}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    print("GhostFaceNetsV2 Emotion Recognition Test")
    print("=" * 50)
    
    # Show available checkpoints
    checkpoints = get_available_checkpoints()
    if checkpoints:
        print("✓ Found GhostFaceNetsV2 checkpoints:")
        for checkpoint in checkpoints:
            print(f"  - {os.path.basename(checkpoint)}")
    else:
        print("❌ No GhostFaceNetsV2 checkpoints found")
        print("Make sure you have .h5 files with 'GN_' or 'GhostFaceNet' in the name")
    
    print("\nRunning video test automatically...")
    video_path = "video/moonlight_sonata/short_video.mp4"
    print(f"Using default video: {video_path}")
    test_with_video(video_path) 