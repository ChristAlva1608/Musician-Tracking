#!/usr/bin/env python3
"""
Lightweight emotion detection for real-time applications
Uses simple heuristics and facial landmarks for speed
"""

import cv2
import numpy as np
import mediapipe as mp
import time

class LightweightEmotionDetector:
    """
    Lightweight emotion detector using facial landmarks
    Much faster than deep learning models
    """
    def __init__(self):
        """
        Initialize lightweight emotion detector
        """
        # Initialize MediaPipe Face Mesh for facial landmarks
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Emotion classes
        self.emotion_classes = ['Happy', 'Sad', 'Angry', 'Surprise', 'Neutral']
        
        # Performance tracking
        self.fps_history = []
        self.last_fps_time = time.time()
        
        print("✅ Lightweight emotion detector initialized")
    
    def analyze_facial_expression(self, landmarks):
        """
        Analyze facial expression using landmark positions
        """
        if not landmarks:
            return "Neutral", 0.5
        
        # Get key facial landmarks
        # Mouth corners (61, 291)
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Eye corners (33, 133, 362, 263)
        left_eye_left = landmarks[33]
        left_eye_right = landmarks[133]
        right_eye_left = landmarks[362]
        right_eye_right = landmarks[263]
        
        # Eyebrow positions (70, 300, 336, 69)
        left_eyebrow = landmarks[70]
        right_eyebrow = landmarks[300]
        
        # Calculate features
        mouth_width = abs(right_mouth.x - left_mouth.x)
        mouth_height = abs(right_mouth.y - left_mouth.y)
        
        left_eye_width = abs(left_eye_right.x - left_eye_left.x)
        right_eye_width = abs(right_eye_right.x - right_eye_left.x)
        
        eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2
        
        # Simple emotion classification based on facial features
        emotions = []
        confidences = []
        
        # Happy: wide mouth, raised eyebrows
        if mouth_width > 0.3 and eyebrow_height < 0.3:
            emotions.append("Happy")
            confidences.append(0.8)
        
        # Sad: drooping mouth, lowered eyebrows
        if mouth_width < 0.2 and eyebrow_height > 0.4:
            emotions.append("Sad")
            confidences.append(0.7)
        
        # Angry: narrowed eyes, lowered eyebrows
        if (left_eye_width < 0.15 and right_eye_width < 0.15) and eyebrow_height > 0.4:
            emotions.append("Angry")
            confidences.append(0.6)
        
        # Surprise: wide eyes, raised eyebrows
        if (left_eye_width > 0.25 and right_eye_width > 0.25) and eyebrow_height < 0.2:
            emotions.append("Surprise")
            confidences.append(0.7)
        
        # Default to neutral
        if not emotions:
            emotions.append("Neutral")
            confidences.append(0.5)
        
        # Return the most confident emotion
        best_idx = np.argmax(confidences)
        return emotions[best_idx], confidences[best_idx]
    
    def detect_emotions(self, img):
        """
        Detect emotions in image using facial landmarks
        """
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        face_emotions = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to list
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append(landmark)
                
                # Analyze expression
                emotion, confidence = self.analyze_facial_expression(landmarks)
                
                # Get face bounding box
                h, w, _ = img.shape
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                face_emotions.append(((x_min, y_min, x_max - x_min, y_max - y_min), emotion, confidence))
        
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
            'Surprise': (0, 255, 255), # Yellow
            'Neutral': (128, 128, 128), # Gray
        }
        return emotion_colors.get(emotion, (128, 128, 128))

def test_lightweight_emotion_detection():
    """
    Test lightweight emotion detection
    """
    print("Testing Lightweight Emotion Detection...")
    
    # Initialize detector
    detector = LightweightEmotionDetector()
    
    # Test with webcam
    print("Opening webcam for testing...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("Press 'q' to quit, 's' to save image")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # Detect emotions
        face_emotions = detector.detect_emotions(frame)
        
        # Draw results
        for (x, y, w, h), emotion, confidence in face_emotions:
            color = detector.get_emotion_color(emotion)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(display_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display FPS
        fps = detector.get_fps()
        cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Lightweight Detection", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Lightweight Emotion Detection", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("lightweight_test_frame.jpg", frame)
            print("✅ Test image saved")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_lightweight_emotion_detection() 