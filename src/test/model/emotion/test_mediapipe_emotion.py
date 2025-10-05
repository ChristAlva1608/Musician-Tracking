#!/usr/bin/env python3
"""
Test script for MediaPipe-based emotion detection
"""

import cv2
import numpy as np
import mediapipe as mp
import time

def analyze_facial_expression_mediapipe(landmarks, img_height, img_width):
    """
    Analyze facial expression using MediaPipe Face Mesh landmarks
    
    Args:
        landmarks: List of MediaPipe face landmarks
        img_height: Image height
        img_width: Image width
    
    Returns:
        Dictionary of emotion scores
    """
    # Convert landmarks to pixel coordinates
    points = []
    for landmark in landmarks:
        x = landmark.x * img_width
        y = landmark.y * img_height
        points.append((x, y))
    
    # Define key facial landmark indices for MediaPipe Face Mesh
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    MOUTH = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    
    # Calculate facial features
    emotions = {
        'happy': 0.0,
        'sad': 0.0,
        'angry': 0.0,
        'surprise': 0.0,
        'fear': 0.0,
        'disgust': 0.0,
        'neutral': 0.0
    }
    
    try:
        # Calculate mouth openness (for surprise, happy)
        mouth_height = calculate_mouth_height(points, MOUTH)
        mouth_width = calculate_mouth_width(points, MOUTH)
        
        # Calculate eyebrow position (for surprise, angry, sad)
        left_eyebrow_height = calculate_eyebrow_height(points, LEFT_EYEBROW)
        right_eyebrow_height = calculate_eyebrow_height(points, RIGHT_EYEBROW)
        
        # Calculate eye openness (for fear, surprise)
        left_eye_openness = calculate_eye_openness(points, LEFT_EYE)
        right_eye_openness = calculate_eye_openness(points, RIGHT_EYE)
        
        # Analyze expressions based on facial features
        
        # Happy: wide mouth, raised cheeks
        if mouth_width > 0.3:  # Wide mouth
            emotions['happy'] = min(1.0, mouth_width * 2)
        
        # Surprise: raised eyebrows, wide eyes, open mouth
        if left_eyebrow_height > 0.1 and right_eyebrow_height > 0.1 and mouth_height > 0.1:
            surprise_score = (left_eyebrow_height + right_eyebrow_height + mouth_height) / 3
            emotions['surprise'] = min(1.0, surprise_score * 3)
        
        # Sad: lowered eyebrows, downturned mouth
        if left_eyebrow_height < -0.05 and right_eyebrow_height < -0.05:
            emotions['sad'] = min(1.0, abs(left_eyebrow_height + right_eyebrow_height) * 2)
        
        # Angry: lowered eyebrows, narrowed eyes
        if left_eyebrow_height < -0.08 and right_eyebrow_height < -0.08 and left_eye_openness < 0.3:
            emotions['angry'] = min(1.0, abs(left_eyebrow_height + right_eyebrow_height) * 2.5)
        
        # Fear: wide eyes, raised eyebrows
        if left_eye_openness > 0.8 and right_eye_openness > 0.8 and left_eyebrow_height > 0.08:
            emotions['fear'] = min(1.0, (left_eye_openness + right_eye_openness) * 0.5)
        
        # Disgust: wrinkled nose, narrowed eyes
        if left_eye_openness < 0.4 and right_eye_openness < 0.4 and mouth_width < 0.2:
            emotions['disgust'] = min(1.0, (1 - left_eye_openness + 1 - right_eye_openness) * 0.5)
        
        # Neutral: balanced features
        if all(score < 0.3 for emotion, score in emotions.items() if emotion != 'neutral'):
            emotions['neutral'] = 0.8
        
        # Normalize scores
        total_score = sum(emotions.values())
        if total_score > 0:
            for emotion in emotions:
                emotions[emotion] /= total_score
        
    except Exception as e:
        # If analysis fails, default to neutral
        emotions['neutral'] = 1.0
    
    return emotions

def calculate_mouth_height(points, mouth_indices):
    """Calculate mouth height using landmark points"""
    try:
        mouth_points = [points[i] for i in mouth_indices if i < len(points)]
        if len(mouth_points) >= 2:
            heights = []
            for i in range(0, len(mouth_points), 2):
                if i + 1 < len(mouth_points):
                    height = abs(mouth_points[i][1] - mouth_points[i+1][1])
                    heights.append(height)
            return np.mean(heights) / 100.0  # Normalize
    except:
        pass
    return 0.0

def calculate_mouth_width(points, mouth_indices):
    """Calculate mouth width using landmark points"""
    try:
        mouth_points = [points[i] for i in mouth_indices if i < len(points)]
        if len(mouth_points) >= 2:
            x_coords = [p[0] for p in mouth_points]
            width = max(x_coords) - min(x_coords)
            return width / 200.0  # Normalize
    except:
        pass
    return 0.0

def calculate_eyebrow_height(points, eyebrow_indices):
    """Calculate eyebrow height relative to eyes"""
    try:
        eyebrow_points = [points[i] for i in eyebrow_indices if i < len(points)]
        if len(eyebrow_points) >= 2:
            # Calculate average eyebrow position
            avg_y = np.mean([p[1] for p in eyebrow_points])
            # Compare with eye position (approximate)
            eye_y = avg_y + 20  # Approximate eye position
            return (eye_y - avg_y) / 100.0  # Normalize
    except:
        pass
    return 0.0

def calculate_eye_openness(points, eye_indices):
    """Calculate eye openness using landmark points"""
    try:
        eye_points = [points[i] for i in eye_indices if i < len(points)]
        if len(eye_points) >= 4:
            # Calculate eye area or height
            y_coords = [p[1] for p in eye_points]
            height = max(y_coords) - min(y_coords)
            return height / 50.0  # Normalize
    except:
        pass
    return 0.5  # Default to half-open

def detect_emotions_mediapipe(img, face_mesh_model):
    """
    Detect emotions using MediaPipe Face Mesh landmarks
    
    Args:
        img: Input image (BGR format)
        face_mesh_model: MediaPipe Face Mesh model
    
    Returns:
        List of emotion detection results with bounding boxes and emotion scores
    """
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh_model.process(img_rgb)
    
    emotions_in_frame = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get face bounding box
            h, w, _ = img.shape
            x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Analyze facial expression using landmarks
            emotions = analyze_facial_expression_mediapipe(face_landmarks.landmark, h, w)
            
            emotions_in_frame.append({
                "box": (x_min, y_min, x_max - x_min, y_max - y_min),
                "emotions": emotions
            })
    
    return emotions_in_frame

def draw_emotion_results(img, emotions_in_frame):
    """
    Draw emotion detection results on the image
    
    Args:
        img: Input image
        emotions_in_frame: List of emotion detection results from MediaPipe
    """
    for emotion_info in emotions_in_frame:
        (x, y, w, h) = emotion_info["box"]
        top_emotion = max(emotion_info["emotions"], key=emotion_info["emotions"].get)
        emotion_score = emotion_info["emotions"][top_emotion]
        
        # Draw green box around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text with the top emotion and its score
        text = f"{top_emotion}: {emotion_score:.2f}"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display all emotions for this face
        y_offset = 20
        for emotion, score in emotion_info["emotions"].items():
            if score > 0.1:  # Only show emotions with >10% confidence
                emotion_text = f"{emotion}: {score:.2f}"
                cv2.putText(img, emotion_text, (x, y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 15

def main():
    """
    Test MediaPipe emotion detection with webcam
    """
    print("Initializing MediaPipe Face Mesh...")
    
    # Initialize MediaPipe Face Mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✅ MediaPipe Face Mesh initialized")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("✅ Webcam opened successfully")
    print("Press 'q' to quit, 's' to save test image")
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0  # Initialize fps variable
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Detect emotions using MediaPipe
        emotions_in_frame = detect_emotions_mediapipe(frame, face_mesh)
        
        # Draw results
        draw_emotion_results(frame, emotions_in_frame)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - start_time)
            start_time = current_time
            print(f"FPS: {fps:.1f}")
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("MediaPipe Emotion Detection", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save test image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_test_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Test image saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completed")

if __name__ == "__main__":
    main() 