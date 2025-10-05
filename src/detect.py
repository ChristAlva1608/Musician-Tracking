import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
import json
from datetime import datetime
import torch
import yaml
# from fer import FER
from database.setup import MusicianDatabase
from models.emotion import DeepFaceDetector, GhostFaceNetDetector, FERDetector

def convert_landmarks_to_dict(landmarks):
    """
    Convert MediaPipe landmarks to dictionary format for database storage
    
    Args:
        landmarks: MediaPipe landmark list
    
    Returns:
        List of dictionaries with x, y, z, confidence
    """
    if not landmarks:
        return None
    
    landmark_data = []
    for landmark in landmarks.landmark:
        landmark_data.append({
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z) if hasattr(landmark, 'z') else 0.0,
            "confidence": float(landmark.visibility) if hasattr(landmark, 'visibility') else 1.0
        })
    
    return landmark_data

def convert_hand_landmarks_to_dict(multi_hand_landmarks):
    """
    Convert MediaPipe hand landmarks to database format
    
    Args:
        multi_hand_landmarks: MediaPipe multi-hand landmarks
    
    Returns:
        Tuple of (left_hand_landmarks, right_hand_landmarks)
    """
    left_hand = None
    right_hand = None
    
    if multi_hand_landmarks:
        for i, hand_landmarks in enumerate(multi_hand_landmarks):
            hand_data = convert_landmarks_to_dict(hand_landmarks)
            # For simplicity, assume first hand is left, second is right
            # In production, you'd use handedness detection
            if i == 0:
                left_hand = hand_data
            elif i == 1:
                right_hand = hand_data
    
    return left_hand, right_hand

def convert_pose_landmarks_to_dict(pose_results):
    """
    Convert pose landmarks to database format (supports both YOLO and MediaPipe)
    
    Args:
        pose_results: YOLO results (list) or MediaPipe pose results (object)
    
    Returns:
        List of pose landmark dictionaries
    """
    # Handle YOLO results (list format)
    if isinstance(pose_results, list):
        if not pose_results or pose_results[0].keypoints is None:
            return None
        
        # Convert YOLO keypoints to our format
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()
        if len(keypoints) == 0:
            return None
            
        # Get the first person's keypoints
        person_keypoints = keypoints[0]
        landmark_data = []
        
        for i, (x, y) in enumerate(person_keypoints):
            # YOLO provides x,y coordinates, we add z=0 and confidence=1 for consistency
            landmark_data.append({
                "x": float(x) if x > 0 else 0.0,
                "y": float(y) if y > 0 else 0.0, 
                "z": 0.0,  # YOLO doesn't provide z-coordinate
                "confidence": 1.0 if x > 0 and y > 0 else 0.0  # Valid if both x,y > 0
            })
        
        return landmark_data
    
    # Handle MediaPipe results (object format)
    else:
        if not pose_results or not pose_results.pose_landmarks:
            return None
        
        return convert_landmarks_to_dict(pose_results.pose_landmarks)

def get_hand_landmarks(hand_model, img):
    """
    Detect hand landmarks using MediaPipe Hands model
    """
    return hand_model.process(img).multi_hand_landmarks

def get_mp_pose_landmarks(mp_pose, img):
    """
    Detect pose landmarks using MediaPipe Pose model
    
    Args:
        mp_pose: MediaPipe Pose model
        img: Input image (RGB format)
    
    Returns:
        MediaPipe pose results or None
    """
    return mp_pose.process(img)

def get_face_landmarks(face_mesh_model, img):
    """
    Detect face landmarks using MediaPipe Face Mesh model
    """
    return face_mesh_model.process(img).multi_face_landmarks

def extract_face_keypoints_from_yolo(pose_results):
    """
    Extract face keypoints from YOLO pose results
    
    YOLO pose keypoints include face landmarks:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    
    Args:
        pose_results: YOLO pose detection results
    
    Returns:
        List of face landmarks in MediaPipe format or None
    """
    if not pose_results or pose_results[0].keypoints is None:
        return None
    
    keypoints = pose_results[0].keypoints.xy.cpu().numpy()
    if len(keypoints) == 0:
        return None
    
    # Get the first person's keypoints
    person_keypoints = keypoints[0]
    
    # Face keypoint indices in YOLO COCO format
    face_indices = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear
    
    face_landmarks = []
    for idx in face_indices:
        if idx < len(person_keypoints):
            x, y = person_keypoints[idx]
            if x > 0 and y > 0:  # Valid keypoint
                # Convert to normalized coordinates (similar to MediaPipe format)
                # Note: This is a simplified conversion - you may need to adjust based on image size
                face_landmarks.append({
                    "x": float(x / 640),  # Assuming image width of 640 (adjust as needed)
                    "y": float(y / 480),  # Assuming image height of 480 (adjust as needed)  
                    "z": 0.0,
                    "confidence": 1.0
                })
    
    # Return in a format similar to MediaPipe multi_face_landmarks
    return [face_landmarks] if face_landmarks else None

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
    # These are approximate indices - you may need to adjust based on your specific needs
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

def draw_mp_hand(img, multi_hand_landmarks):
    """
    Draw hand landmarks and connections on the image
    """
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, 
                hand_landmarks, 
                mp.solutions.hands.HAND_CONNECTIONS
            )

def draw_mp_face_mesh(img, multi_face_landmarks):
    """
    Draw face landmarks and mesh on the image
    """
    if multi_face_landmarks:
        for face_landmarks in multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, 
                face_landmarks, 
                mp.solutions.face_mesh.FACEMESH_TESSELATION
            )

def draw_facelandmarker_landmarks(img, detection_result):
    """
    Draw face landmarks from FaceLandmarker API on the image
    
    Args:
        img: Input image (BGR format)
        detection_result: FaceLandmarker detection result
    """
    if not detection_result.face_landmarks:
        return
    
    # Get image dimensions
    image_rows, image_cols, _ = img.shape
    
    # Draw landmarks for each detected face
    for face_landmarks in detection_result.face_landmarks:
        # Draw landmarks as small circles
        for landmark in face_landmarks:
            x = int(landmark.x * image_cols)
            y = int(landmark.y * image_rows)
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

def draw_yolo_face_keypoints(img, face_landmarks_list):
    """
    Draw YOLO face keypoints on the image
    
    Args:
        img: Input image
        face_landmarks_list: List of face landmark dictionaries from YOLO
    """
    if not face_landmarks_list:
        return
        
    # Get image dimensions for coordinate conversion
    h, w, _ = img.shape
    
    # Colors for different face keypoints
    keypoint_colors = [
        (0, 255, 0),    # nose - green
        (255, 0, 0),    # left_eye - blue
        (255, 0, 0),    # right_eye - blue  
        (0, 255, 255),  # left_ear - yellow
        (0, 255, 255)   # right_ear - yellow
    ]
    
    for face_landmarks in face_landmarks_list:
        for i, landmark in enumerate(face_landmarks):
            if isinstance(landmark, dict):
                x = int(landmark['x'] * w)
                y = int(landmark['y'] * h)
                color = keypoint_colors[i] if i < len(keypoint_colors) else (255, 255, 255)
                
                # Draw keypoint
                cv2.circle(img, (x, y), 3, color, -1)
                
        # Draw connections between face keypoints
        if len(face_landmarks) >= 5:
            # Connect eyes to ears
            for i in range(min(4, len(face_landmarks))):
                if i + 1 < len(face_landmarks):
                    landmark1 = face_landmarks[i]
                    landmark2 = face_landmarks[i + 1]
                    if isinstance(landmark1, dict) and isinstance(landmark2, dict):
                        x1, y1 = int(landmark1['x'] * w), int(landmark1['y'] * h)
                        x2, y2 = int(landmark2['x'] * w), int(landmark2['y'] * h)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

def draw_yolo_pose(img, pose_results):
    """
    Draw YOLO pose detection results
    """
    if pose_results and pose_results[0].keypoints is not None:
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()
        
        # Define COCO pose connections (17 keypoints)
        pose_connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (3, 5), (4, 6)   # Neck: ears to shoulders
        ]
        for person_keypoints in keypoints:
            # Draw keypoints
            for i, (x, y) in enumerate(person_keypoints):
                if x > 0 and y > 0:  # Valid keypoint
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Draw connections
            for connection in pose_connections:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(person_keypoints) and pt2_idx < len(person_keypoints)):
                    pt1 = person_keypoints[pt1_idx]
                    pt2 = person_keypoints[pt2_idx]
                    if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0):
                        cv2.line(img, (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)

def draw_mp_pose(img, mp_pose_results):
    """
    Draw MediaPipe pose detection results
    
    Args:
        img: Input image
        mp_pose_results: MediaPipe pose results
    """
    if mp_pose_results.pose_landmarks:
        # Draw pose landmarks and connections
        mp.solutions.drawing_utils.draw_landmarks(
            img, 
            mp_pose_results.pose_landmarks, 
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

def draw_emotion_results(img, emotions_in_frame):
    """
    Draw emotion detection results on the image
    
    Args:
        img: Input image
        emotions_in_frame: List of emotion detection results
    """
    for emotion_info in emotions_in_frame:
        # Handle different formats (DeepFace uses 'bbox', older code used 'box')
        if 'bbox' in emotion_info:
            (x, y, w, h) = emotion_info["bbox"]
        elif 'box' in emotion_info:
            (x, y, w, h) = emotion_info["box"]
        else:
            continue  # Skip if no bounding box information
            
        # Get top emotion and score
        if 'dominant_emotion' in emotion_info and 'confidence' in emotion_info:
            # DeepFace format
            top_emotion = emotion_info['dominant_emotion']
            emotion_score = emotion_info['confidence']
        else:
            # FER format
            top_emotion = max(emotion_info["emotions"], key=emotion_info["emotions"].get)
            emotion_score = emotion_info["emotions"][top_emotion]
        
        # Draw green box around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text with the top emotion and its score
        text = f"{top_emotion}: {emotion_score:.2f}"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

####### Functions to detect bad gestures #######

#--------- Turtle neck -----------#
def extract_unified_pose_landmarks(pose_results, img_width, img_height):
    """
    Extract pose landmarks in unified format from both YOLO and MediaPipe results
    Returns: dict with landmark positions as pixel coordinates or None if no valid pose
    """
    # Handle YOLO results (list format)
    if isinstance(pose_results, list) and pose_results and pose_results[0].keypoints is not None:
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()
        if len(keypoints) == 0:
            return None
            
        person_keypoints = keypoints[0]
        # YOLO COCO keypoint mapping: 0=nose, 3=left_ear, 4=right_ear, 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
        if len(person_keypoints) >= 13:
            return {
                'nose': np.array([person_keypoints[0][0], person_keypoints[0][1]]),
                'left_ear': np.array([person_keypoints[3][0], person_keypoints[3][1]]),
                'right_ear': np.array([person_keypoints[4][0], person_keypoints[4][1]]),
                'left_shoulder': np.array([person_keypoints[5][0], person_keypoints[5][1]]),
                'right_shoulder': np.array([person_keypoints[6][0], person_keypoints[6][1]]),
                'left_hip': np.array([person_keypoints[11][0], person_keypoints[11][1]]),
                'right_hip': np.array([person_keypoints[12][0], person_keypoints[12][1]])
            }
    
    # Handle MediaPipe results (object format)
    elif pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        return {
            'nose': np.array([landmarks[0].x * img_width, landmarks[0].y * img_height]),
            'left_ear': np.array([landmarks[7].x * img_width, landmarks[7].y * img_height]),
            'right_ear': np.array([landmarks[8].x * img_width, landmarks[8].y * img_height]),
            'left_shoulder': np.array([landmarks[11].x * img_width, landmarks[11].y * img_height]),
            'right_shoulder': np.array([landmarks[12].x * img_width, landmarks[12].y * img_height]),
            'left_hip': np.array([landmarks[23].x * img_width, landmarks[23].y * img_height]),
            'right_hip': np.array([landmarks[24].x * img_width, landmarks[24].y * img_height])
        }
    
    return None

def turtle_neck(img, pose_results):
    """
    Detect turtle neck posture using pose landmarks (supports both YOLO and MediaPipe)
    """
    detected = False
    h, w, _ = img.shape
    
    # Extract landmarks in unified format
    landmarks = extract_unified_pose_landmarks(pose_results, w, h)
    if not landmarks:
        return detected
        
    # Get landmark positions
    nose = landmarks['nose']
    left_ear = landmarks['left_ear']
    right_ear = landmarks['right_ear']
    left_shoulder = landmarks['left_shoulder']
    right_shoulder = landmarks['right_shoulder']
    
    # Check if all keypoints are valid (for YOLO compatibility)
    if not all(pt[0] > 0 and pt[1] > 0 for pt in [nose, left_ear, right_ear, left_shoulder, right_shoulder]):
        return detected
        
    # Calculate ear center and shoulder center
    ear_center = (left_ear + right_ear) / 2
    shoulder_center = (left_shoulder + right_shoulder) / 2
        
    # Calculate neck length (distance from ear center to shoulder center)
    neck_vector = ear_center - shoulder_center
    neck_length = np.linalg.norm(neck_vector)
        
    # Calculate head forward position (nose position relative to shoulder line)
    # Project nose onto shoulder line
    shoulder_vector = right_shoulder - left_shoulder
    shoulder_length = np.linalg.norm(shoulder_vector)
        
    if shoulder_length > 0:
        # Calculate how far forward the nose is from the shoulder line
        nose_to_shoulder = nose - left_shoulder
        shoulder_unit = shoulder_vector / shoulder_length
            
        # Project nose onto shoulder line
        projection_length = np.dot(nose_to_shoulder, shoulder_unit)
        projected_point = left_shoulder + projection_length * shoulder_unit
            
        # Calculate forward distance
        forward_distance = np.linalg.norm(nose - projected_point)
            
        # Turtle neck detection criteria
        forward_threshold = shoulder_length * 0.25  # Head forward by 25% of shoulder width
        neck_extension_threshold = neck_length * 1.5  # Neck extended beyond normal length
            
        if forward_distance > forward_threshold or neck_length > neck_extension_threshold:
            detected = True
                
            # Draw detection visualization
            cv2.circle(img, (int(nose[0]), int(nose[1])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(ear_center[0]), int(ear_center[1])), 5, (0, 255, 0), -1)
            cv2.circle(img, (int(shoulder_center[0]), int(shoulder_center[1])), 5, (255, 0, 0), -1)
                
            # Draw neck line
            cv2.line(img, (int(ear_center[0]), int(ear_center[1])), 
                    (int(shoulder_center[0]), int(shoulder_center[1])), (0, 0, 255), 2)
                
            # Draw forward projection line
            cv2.line(img, (int(nose[0]), int(nose[1])), 
                    (int(projected_point[0]), int(projected_point[1])), (255, 0, 255), 2)
                
            # Display detection text
            cv2.putText(img, "TURTLE NECK", 
                       (int(nose[0] - 60), int(nose[1] - 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            # Display measurements
            cv2.putText(img, f"Forward: {forward_distance:.1f}", 
                       (int(nose[0] - 60), int(nose[1] + 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(img, f"Neck Length: {neck_length:.1f}", 
                       (int(nose[0] - 60), int(nose[1] + 35)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return detected

#--------- Hunched back -----------#
def hunched_back(img, pose_results):
    """
    Detect hunched back posture using pose landmarks (supports both YOLO and MediaPipe)
    
    Args:
        img: Input image
        pose_results: Pose detection results (YOLO or MediaPipe)
    
    Returns:
        bool: True if hunched back is detected (angle between CA and CB < 160 degrees)
    """
    detected = False
    h, w, _ = img.shape
    
    # Extract landmarks in unified format
    landmarks = extract_unified_pose_landmarks(pose_results, w, h)
    if not landmarks:
        return detected
        
    # Get landmark positions
    left_shoulder = landmarks['left_shoulder']
    right_shoulder = landmarks['right_shoulder']
    left_hip = landmarks['left_hip']
    right_hip = landmarks['right_hip']
    
    # Check if all keypoints are valid (for YOLO compatibility)
    if not all(pt[0] > 0 and pt[1] > 0 for pt in [left_shoulder, right_shoulder, left_hip, right_hip]):
        return detected
        
    # Calculate the three key points:
    # A: middle_shoulder point
    middle_shoulder = np.array([
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    ])
        
    # B: middle_hip point
    middle_hip = np.array([
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    ])
        
    # C: middle_back point (intersection of 2 diagonal lines)
    # Diagonal 1: from left shoulder to right hip
    # Diagonal 2: from right shoulder to left hip
    # Intersection point C
    middle_back = np.array([
        (left_shoulder[0] + right_hip[0]) / 2,
        (left_shoulder[1] + right_hip[1]) / 2
    ])
        
    # Calculate vectors CA and CB
    vector_CA = middle_shoulder - middle_back  # Vector from C to A
    vector_CB = middle_hip - middle_back       # Vector from C to B
        
    # Calculate the angle between CA and CB
    dot_product = np.dot(vector_CA, vector_CB)
    magnitude_CA = np.linalg.norm(vector_CA)
    magnitude_CB = np.linalg.norm(vector_CB)
        
    if magnitude_CA > 0 and magnitude_CB > 0:
        cos_angle = dot_product / (magnitude_CA * magnitude_CB)
        # Clamp cos_angle to [-1, 1] to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_degrees = np.degrees(np.arccos(cos_angle))
            
        # Detection logic: if angle between CA and CB < 160 degrees, then hunched back
        if angle_degrees < 160:
            detected = True
                
            # Visualization
            # Draw the three key points
            cv2.circle(img, (int(middle_shoulder[0]), int(middle_shoulder[1])), 5, (0, 255, 0), -1)  # Green for A
            cv2.circle(img, (int(middle_hip[0]), int(middle_hip[1])), 5, (255, 0, 0), -1)           # Blue for B
            cv2.circle(img, (int(middle_back[0]), int(middle_back[1])), 5, (0, 0, 255), -1)        # Red for C
                
            # Draw lines CA and CB
            cv2.line(img, (int(middle_back[0]), int(middle_back[1])), 
                    (int(middle_shoulder[0]), int(middle_shoulder[1])), (0, 255, 0), 2)  # CA line
            cv2.line(img, (int(middle_back[0]), int(middle_back[1])), 
                    (int(middle_hip[0]), int(middle_hip[1])), (255, 0, 0), 2)            # CB line
                
            # Display detection text and angle
            cv2.putText(img, f"HUNCHED BACK", 
                       (int(middle_back[0] - 80), int(middle_back[1] - 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            cv2.putText(img, f"Angle: {angle_degrees:.1f}°", 
                       (int(middle_back[0] - 80), int(middle_back[1] + 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
            cv2.putText(img, f"Threshold: 160°", 
                       (int(middle_back[0] - 80), int(middle_back[1] + 35)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return detected

#--------- Raised shoulders -----------#
def raised_shoulders(img, hand_landmarker_result):
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            for landmark in hand_landmarks:
                if landmark.y > 0.5:
                    return True
    return False

#--------- Low wrists -----------#
def low_wrists(img, multi_hand_landmarks):
    detected = False
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            # Check if wrist (landmark 0) is lower than knuckle (landmark 9) 
            wrist = hand_landmarks.landmark[0]
            middle_knuckle = hand_landmarks.landmark[9]
            if wrist.y > middle_knuckle.y:  # Higher y value means lower position
                detected = True
                # Convert normalized coordinates to pixel coordinates
                wrist_x = int(wrist.x * img.shape[1])
                wrist_y = int(wrist.y * img.shape[0])
                
                # Display alarm text at wrist location
                cv2.putText(img, "LOW-WRIST", 
                            (wrist_x - 40, wrist_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Add warning background rectangle around text
                cv2.rectangle(img, (wrist_x - 45, wrist_y - 25), 
                                (wrist_x + 55, wrist_y + 5), (0, 0, 255), 1)
    return detected

#--------- Fingers pointing up to the sky -----------#
def fingers_pointing_up_to_the_sky(img, multi_hand_landmarks):
    detected = False
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            pinky_tip = hand_landmarks.landmark[20]
            pinky_mcp = hand_landmarks.landmark[17]
            if pinky_tip.y < pinky_mcp.y:
                detected = True
                cv2.putText(img, "FINGERS POINTING UP TO THE SKY", 
                            (int(pinky_tip.x * img.shape[1]), int(pinky_tip.y * img.shape[0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(img, (int(pinky_tip.x * img.shape[1] - 20), int(pinky_tip.y * img.shape[0] - 20)), 
                             (int(pinky_tip.x * img.shape[1] + 20), int(pinky_tip.y * img.shape[0] + 20)), (0, 0, 255), 1)
    return detected


class LandmarkHeatmapGenerator:
    def __init__(self, frame_width, frame_height):
        """
        Initialize heatmap generator
        Args:
            frame_width: Width of video frames
            frame_height: Height of video frames
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.hand_heatmap_data = defaultdict(list)  # {landmark_index: [(x, y), ...]}
        self.pose_heatmap_data = defaultdict(list)   # {landmark_index: [(x, y), ...]}
        # self.face_heatmap_data = defaultdict(list)   # {landmark_index: [(x, y), ...]}
        
    def add_hand_landmark_data(self, multi_hand_landmarks, landmark_index):
        """
        Add hand landmark position to heatmap data
        Args:
            multi_hand_landmarks: MediaPipe hand landmarks
            landmark_index: Index of the landmark to track (0-20)
        """
        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                if landmark_index < len(hand_landmarks.landmark):
                    landmark = hand_landmarks.landmark[landmark_index]
                    x = int(landmark.x * self.frame_width)
                    y = int(landmark.y * self.frame_height)
                    self.hand_heatmap_data[landmark_index].append((x, y))
    
    def add_pose_landmark_data(self, pose_results, landmark_index):
        """
        Add pose landmark position to heatmap data
        Args:
            pose_results: YOLO pose detection results
            landmark_index: Index of the landmark to track (0-16 for COCO)
        """
        if pose_results and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()
            for person_keypoints in keypoints:
                if landmark_index < len(person_keypoints):
                    x, y = person_keypoints[landmark_index]
                    if x > 0 and y > 0:  # Valid keypoint
                        self.pose_heatmap_data[landmark_index].append((int(x), int(y)))
    
    # def add_face_landmark_data(self, multi_face_landmarks, landmark_index):
    #     """
    #     Add face landmark position to heatmap data
    #     Args:
    #         multi_face_landmarks: MediaPipe face landmarks
    #         landmark_index: Index of the landmark to track (0-467 for face mesh)
    #     """
    #     if multi_face_landmarks:
    #         for face_landmarks in multi_face_landmarks:
    #             if landmark_index < len(face_landmarks.landmark):
    #                 landmark = face_landmarks.landmark[landmark_index]
    #                 x = int(landmark.x * self.frame_width)
    #                 y = int(landmark.y * self.frame_height)
    #                 self.face_heatmap_data[landmark_index].append((x, y))
    
    def generate_heatmap(self, landmark_type, landmark_index, save_path=None):
        """
        Generate and display heatmap for a specific landmark
        Args:
            landmark_type: 'hand', 'pose', or 'face'
            landmark_index: Index of the landmark
            save_path: Optional path to save the heatmap image
        """
        # Select appropriate data
        if landmark_type == 'hand':
            data = self.hand_heatmap_data[landmark_index]
            title = f"Hand Landmark {landmark_index} Heatmap"
        elif landmark_type == 'pose':
            data = self.pose_heatmap_data[landmark_index]
            title = f"Pose Landmark {landmark_index} Heatmap"
        # elif landmark_type == 'face':
        #     data = self.face_heatmap_data[landmark_index]
        #     title = f"Face Landmark {landmark_index} Heatmap"
        else:
            raise ValueError("landmark_type must be 'hand', 'pose', or 'face'")
        
        if not data:
            print(f"No data collected for {landmark_type} landmark {landmark_index}")
            return None
        
        # Create heatmap
        heatmap = np.zeros((self.frame_height, self.frame_width))
        
        # Add Gaussian blur for each point
        for x, y in data:
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                # Create small gaussian around each point
                y_min = max(0, y - 10)
                y_max = min(self.frame_height, y + 11)
                x_min = max(0, x - 10)
                x_max = min(self.frame_width, x + 11)
                
                for dy in range(y_min, y_max):
                    for dx in range(x_min, x_max):
                        distance = np.sqrt((dx - x)**2 + (dy - y)**2)
                        if distance <= 10:
                            intensity = np.exp(-(distance**2) / (2 * 5**2))
                            heatmap[dy, dx] += intensity
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='bilinear', origin='upper')
        plt.colorbar(label='Intensity')
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()
        return heatmap
    
    def get_heatmap_overlay(self, landmark_type, landmark_index, alpha=0.6):
        """
        Get heatmap as overlay for video frames
        Args:
            landmark_type: 'hand', 'pose', or 'face'
            landmark_index: Index of the landmark
            alpha: Transparency of the overlay
        Returns:
            Colored heatmap overlay as BGR image
        """
        # Select appropriate data
        if landmark_type == 'hand':
            data = self.hand_heatmap_data[landmark_index]
        elif landmark_type == 'pose':
            data = self.pose_heatmap_data[landmark_index]
        # elif landmark_type == 'face':
        #     data = self.face_heatmap_data[landmark_index]
        else:
            return None
        
        if not data:
            return None
        
        # Create heatmap
        heatmap = np.zeros((self.frame_height, self.frame_width))
        
        for x, y in data:
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                y_min = max(0, y - 10)
                y_max = min(self.frame_height, y + 11)
                x_min = max(0, x - 10)
                x_max = min(self.frame_width, x + 11)
                
                for dy in range(y_min, y_max):
                    for dx in range(x_min, x_max):
                        distance = np.sqrt((dx - x)**2 + (dy - y)**2)
                        if distance <= 10:
                            intensity = np.exp(-(distance**2) / (2 * 5**2))
                            heatmap[dy, dx] += intensity
        
        # Normalize and apply colormap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to color using OpenCV colormap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap_color

class GestureTimer:
    def __init__(self, report_time_threshold=300):  # Default 300 seconds = 5 minutes
        """Initialize gesture timer for tracking bad posture durations"""
        self.gesture_start_times = {
            'low_wrists': None,
            'turtle_neck': None,
            'hunched_back': None,
            'fingers_pointing_up': None
        }
        self.gesture_durations = {
            'low_wrists': 0,
            'turtle_neck': 0,
            'hunched_back': 0,
            'fingers_pointing_up': 0
        }
        
        # Emotion tracking
        self.emotion_counts = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0
        }
        self.emotion_history = []
        self.total_frames = 0
        self.session_start_time = time.time()
        self.last_report_time = time.time()
        self.report_time_threshold = report_time_threshold  # Configurable report interval
        
        # Frame-by-frame tracking for histogram
        self.frame_gesture_data = {
            'low_wrists': [],
            'turtle_neck': [],
            'hunched_back': [],
            'fingers_pointing_up': []
        }
        
        # Frame-by-frame emotion tracking
        self.frame_emotion_data = {
            'angry': [],
            'disgust': [],
            'fear': [],
            'happy': [],
            'sad': [],
            'surprise': [],
            'neutral': []
        }
        

        
        self.fps = 30  # Default FPS, will be updated
        self.session_start_frame = 0
        
    def start_gesture(self, gesture_type):
        """Start timing a bad gesture"""
        if self.gesture_start_times[gesture_type] is None:
            self.gesture_start_times[gesture_type] = time.time()
    
    def stop_gesture(self, gesture_type):
        """Stop timing a bad gesture and add to total duration"""
        if self.gesture_start_times[gesture_type] is not None:
            duration = time.time() - self.gesture_start_times[gesture_type]
            self.gesture_durations[gesture_type] += duration
            self.gesture_start_times[gesture_type] = None
    
    def update_frame(self):
        """Update frame count and check if report is due"""
        self.total_frames += 1
        current_time = time.time()
        
        # Check if report threshold has been reached
        if current_time - self.last_report_time >= self.report_time_threshold:
            self.generate_analysis_report()
            self.last_report_time = current_time
    
    def set_fps(self, fps):
        """Set FPS for accurate time calculations"""
        self.fps = fps
    
    def record_gesture_frame(self, gesture_type, detected):
        """Record gesture detection for current frame"""
        self.frame_gesture_data[gesture_type].append(1 if detected else 0)
    
    def record_emotion(self, emotions_in_frame):
        """Record emotions detected in current frame"""
        # Initialize frame emotion data
        frame_emotions = {emotion: 0 for emotion in self.emotion_counts.keys()}
        
        if emotions_in_frame:
            for emotion_info in emotions_in_frame:
                top_emotion = max(emotion_info["emotions"], key=emotion_info["emotions"].get)
                emotion_score = emotion_info["emotions"][top_emotion]
                
                # Only count emotions with high confidence (>0.5)
                if emotion_score > 0.5:
                    frame_emotions[top_emotion] += 1
                    self.emotion_counts[top_emotion] += 1
        
        # Record frame data for histogram
        for emotion in self.frame_emotion_data.keys():
            self.frame_emotion_data[emotion].append(frame_emotions[emotion])
        
        # Store detailed emotion data for this frame
        self.emotion_history.append({
            'frame': self.total_frames,
            'emotions': frame_emotions,
            'raw_data': emotions_in_frame
        })
    
    def get_emotion_statistics(self):
        """Get emotion statistics"""
        total_emotions = sum(self.emotion_counts.values())
        if total_emotions == 0:
            return {emotion: 0.0 for emotion in self.emotion_counts.keys()}
        
        return {emotion: (count / total_emotions) * 100 
                for emotion, count in self.emotion_counts.items()}
    

    

    
    def get_minute_bins(self):
        """Get gesture counts per 1-minute intervals in seconds"""
        frames_per_minute = self.fps * 60  # 1 minute worth of frames
        
        # Calculate how many complete 1-minute intervals we have
        total_minutes = int(len(self.frame_gesture_data['low_wrists']) // frames_per_minute)
        
        minute_data = {
            'low_wrists': [],
            'turtle_neck': [],
            'fingers_pointing_up': []
        }
        
        for minute in range(total_minutes):
            start_frame = minute * frames_per_minute
            end_frame = (minute + 1) * frames_per_minute
            
            for gesture_type in minute_data.keys():
                if len(self.frame_gesture_data[gesture_type]) >= end_frame:
                    frame_count = sum(self.frame_gesture_data[gesture_type][start_frame:end_frame])
                    # Convert frames to seconds
                    seconds = frame_count / self.fps
                    minute_data[gesture_type].append(seconds)
        
        return minute_data
    
    def get_current_percentages(self):
        """Get current percentages of bad gesture time"""
        total_time = time.time() - self.session_start_time
        if total_time == 0:
            return {gesture: 0.0 for gesture in self.gesture_durations.keys()}
        
        percentages = {}
        for gesture, duration in self.gesture_durations.items():
            percentages[gesture] = (duration / total_time) * 100
        
        return percentages
    
    def generate_analysis_report(self):
        """Generate and save analysis report"""
        # Create analysis_report directory if it doesn't exist
        os.makedirs('analysis_report', exist_ok=True)
        
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        # filename = f"analysis_report/posture_analysis_{timestamp}.json"
        
        # Calculate statistics
        total_time = time.time() - self.session_start_time
        percentages = self.get_current_percentages()
        
        # report_data = {
        #     "timestamp": current_time.isoformat(),
        #     "session_duration_seconds": total_time,
        #     "total_frames_processed": self.total_frames,
        #     "gesture_durations_seconds": self.gesture_durations.copy(),
        #     "gesture_percentages": percentages,
        #     "average_bad_posture_percentage": sum(percentages.values()),
        #     "recommendations": self.generate_recommendations(percentages)
        # }
        
        # # Save report
        # with open(filename, 'w') as f:
        #     json.dump(report_data, f, indent=2)
        
        # Also save a human-readable summary
        summary_filename = f"analysis_report/posture_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write("POSTURE & EMOTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session Duration: {total_time/60:.1f} minutes\n")
            f.write(f"Total Frames: {self.total_frames}\n\n")
            
            f.write("BAD POSTURE BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            for gesture, percentage in percentages.items():
                duration = self.gesture_durations[gesture]
                f.write(f"{gesture.replace('_', ' ').title()}: {percentage:.1f}% ({duration:.1f}s)\n")
            
            f.write(f"\nOverall Bad Posture: {sum(percentages.values()):.1f}%\n\n")
            
    
            
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            for rec in self.generate_recommendations(percentages):
                f.write(f"• {rec}\n")
        
        # print(f"Analysis report generated: {filename}")
        print(f"Summary saved: {summary_filename}")
    
    def generate_histogram(self, timestamp, actual_fps):
        """Generate multi-bars histogram showing gesture frequency per minute"""
        # Calculate frames per minute based on actual FPS
        frames_per_minute = int(actual_fps * 60)  # 1 minute worth of frames
        
        # Calculate how many complete 1-minute intervals we have
        total_frames = len(self.frame_gesture_data['low_wrists'])
        total_minutes = total_frames // frames_per_minute
        
        if total_minutes == 0:  # No complete minutes
            print("No complete minute data available for histogram")
            return
        
        # Prepare data for plotting
        minute_data = {
            'low_wrists': [],
            'turtle_neck': [],
            'hunched_back': [],
            'fingers_pointing_up': []
        }
        
        intervals = list(range(1, total_minutes + 1))
        
        for minute in range(total_minutes):
            start_frame = minute * frames_per_minute
            end_frame = (minute + 1) * frames_per_minute
            
            for gesture_type in minute_data.keys():
                if len(self.frame_gesture_data[gesture_type]) >= end_frame:
                    # Count frames with bad gesture in this minute
                    gesture_frames = sum(self.frame_gesture_data[gesture_type][start_frame:end_frame])
                    # Convert frames to seconds using actual FPS
                    seconds = gesture_frames / actual_fps
                    minute_data[gesture_type].append(seconds)
        
        # Create the histogram
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(intervals))
        width = 0.25
        
        # Plot bars for each gesture type
        bars1 = ax.bar(x - width*1.5, minute_data['low_wrists'], width, 
                      label='Low Wrists', color='red', alpha=0.8)
        bars2 = ax.bar(x - width*0.5, minute_data['turtle_neck'], width, 
                      label='Turtle Neck', color='blue', alpha=0.8)
        bars3 = ax.bar(x + width*0.5, minute_data['hunched_back'], width, 
                      label='Hunched Back', color='orange', alpha=0.8)
        bars4 = ax.bar(x + width*1.5, minute_data['fingers_pointing_up'], width, 
                      label='Fingers Pointing Up', color='green', alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Time (Minutes)', fontsize=12)
        ax.set_ylabel('Time with Bad Gesture (Seconds)', fontsize=12)
        ax.set_title(f'Bad Gesture Duration per Minute (Actual FPS: {actual_fps:.1f})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(intervals)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save the histogram
        histogram_filename = f"analysis_report/gesture_histogram_{timestamp}.png"
        plt.savefig(histogram_filename, dpi=300, bbox_inches='tight')
        print(f"Histogram saved: {histogram_filename}")
        
        # # Also save as PDF for better quality
        # pdf_filename = f"analysis_report/gesture_histogram_{timestamp}.pdf"
        # plt.savefig(pdf_filename, bbox_inches='tight')
        # print(f"Histogram PDF saved: {pdf_filename}")
        
        plt.close()  # Close the plot to free memory
    
    def generate_recommendations(self, percentages):
        """Generate recommendations based on posture percentages"""
        recommendations = []
        
        if percentages['low_wrists'] > 20:
            recommendations.append("Keep wrists elevated - consider wrist support or ergonomic setup")
        
        if percentages['turtle_neck'] > 15:
            recommendations.append("Improve neck posture - keep head aligned with spine")
        
        if percentages['hunched_back'] > 15:
            recommendations.append("Straighten your back - avoid hunching forward")
        
        if percentages['fingers_pointing_up'] > 10:
            recommendations.append("Maintain natural hand position - avoid upward finger pointing")
        
        if sum(percentages.values()) > 30:
            recommendations.append("Overall posture needs improvement - take regular breaks and stretch")
        
        if not recommendations:
            recommendations.append("Great posture! Keep up the good work")
        
        return recommendations


def create_landmark_heatmap_video(video_path, landmark_type, landmark_index, output_path=None):
    # Configuration: Set to True for MediaPipe, False for YOLO
    USE_MEDIAPIPE_POSE = False
    """
    Example function to create a heatmap for a specific landmark across a whole video
    
    Args:
        video_path: Path to input video file
        landmark_type: 'hand', 'pose', or 'face'
        landmark_index: Index of the landmark to track
        output_path: Optional path to save heatmap image
    
    Example usage:
        # Track wrist movement (hand landmark 0)
        create_landmark_heatmap_video("video.mp4", "hand", 0, "wrist_heatmap.png")
        
        # Track nose movement (pose landmark 0)
        create_landmark_heatmap_video("video.mp4", "pose", 0, "nose_heatmap.png")
        
        # Track specific face point (face landmark 1)
        create_landmark_heatmap_video("video.mp4", "face", 1, "face_heatmap.png")
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize models
    hand_model = mp.solutions.hands.Hands()
    face_mesh_model = mp.solutions.face_mesh.FaceMesh()
    
    # Initialize pose detection based on configuration
    if USE_MEDIAPIPE_POSE:
        pose_model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        pose_model = YOLO('checkpoints/yolo11n-pose.pt')
    
    # Initialize heatmap generator
    heatmap_gen = LandmarkHeatmapGenerator(frame_width, frame_height)
    
    print(f"Processing video to track {landmark_type} landmark {landmark_index}...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get detections based on landmark type
        if landmark_type == 'hand':
            multi_hand_landmarks = hand_model.process(img_rgb).multi_hand_landmarks
            heatmap_gen.add_hand_landmark_data(multi_hand_landmarks, landmark_index)
        elif landmark_type == 'pose':
            if USE_MEDIAPIPE_POSE:
                pose_results = pose_model.process(img_rgb)  # MediaPipe pose detection
            else:
                pose_results = pose_model(frame, verbose=False)  # YOLO pose detection
            heatmap_gen.add_pose_landmark_data(pose_results, landmark_index)
        # elif landmark_type == 'face':
        #     multi_face_landmarks = face_mesh_model.process(img_rgb).multi_face_landmarks
        #     heatmap_gen.add_face_landmark_data(multi_face_landmarks, landmark_index)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    print(f"Finished processing {frame_count} frames.")
    
    # Generate and save heatmap
    heatmap_gen.generate_heatmap(landmark_type, landmark_index, output_path)
    
    return heatmap_gen

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️ Config file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return get_default_config()

def get_default_config():
    """Get default configuration if config file is not available"""
    return {
        'database': {
            'enabled': True,
            'batch_size': 50,        # Number of frames to batch before inserting
            'batch_timeout': 5.0     # Maximum time to wait before forcing insert (seconds)
        },
        'video': {
            'source_path': '/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/video/koto_instrument_short.mp4',
            'use_webcam': False,
            'skip_frames': 1, 
            'display_output': True,
            'alignment_directory': '/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/video/multi-cam video/vid_shot1/'
        },
        'detection': {
            'hand_model': 'mediapipe',
            'pose_model': 'yolo',
            'facemesh_model': 'mediapipe',  # 'mediapipe' or 'yolo'
            'emotion_model': 'deepface',
            'emotion_settings': {
                'deepface': {
                    'model_name': 'Facenet',
                    'detector_backend': 'retinaface',
                    'enforce_detection': False
                },
                'ghostfacenet': {
                    'model_version': 'v2',
                    'batch_size': 32
                },
                'fer': {
                    'use_mtcnn': True,
                    'min_face_size': 40
                }
            }
        },
        'bad_gestures': {
            'detect_low_wrists': True,
            'detect_turtle_neck': True,
            'detect_hunched_back': True,
            'detect_fingers_pointing_up': True
        }
    }

def create_emotion_detector(config):
    """
    Create emotion detector based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Emotion detector instance or None
    """
    emotion_model = config['detection'].get('emotion_model', 'none').lower()
    emotion_settings = config['detection'].get('emotion_settings', {})
    
    if emotion_model == 'deepface':
        return DeepFaceDetector(emotion_settings)
    elif emotion_model == 'ghostfacenet':
        return GhostFaceNetDetector(emotion_settings)
    elif emotion_model == 'fer':
        return FERDetector(emotion_settings)
    elif emotion_model == 'mediapipe':
        return None  # Use existing MediaPipe emotion detection
    elif emotion_model == 'none':
        return None
    else:
        print(f"⚠️ Unknown emotion model: {emotion_model}, using MediaPipe")
        return None

def main(skip_frames=None, config_path='config.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Configuration from config file
    USE_MEDIAPIPE_POSE = config['detection'].get('pose_model', 'yolo').lower() == 'mediapipe'
    USE_DATABASE = config['database'].get('enabled', True)
    
    # Override skip_frames from parameter if provided
    if skip_frames is None:
        skip_frames = config['video'].get('skip_frames', 1)
    
    # Video source configuration from config
    VIDEO_PATH = config['video'].get('source_path', '/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/video/koto_instrument_short.mp4')
    USE_WEBCAM = config['video'].get('use_webcam', False)
    
    # Open video file or webcam
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
        video_file = "webcam_session"
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        video_file = os.path.basename(VIDEO_PATH)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # Fallback for webcam
        fps = 30.0
    
    print(f"📹 Video: {video_file}")
    print(f"🎬 FPS: {fps}")
    
    # Initialize database connection
    db = None
    session_id = None
    if USE_DATABASE:
        try:
            db = MusicianDatabase()
            # Configure batch settings from config
            batch_config = config.get('database', {})
            if 'batch_size' in batch_config:
                db.batch_size = batch_config['batch_size']
            if 'batch_timeout' in batch_config:
                db.batch_timeout = batch_config['batch_timeout']
                
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"🗄️ Database connected. Session ID: {session_id}")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("⚠️ Continuing without database export...")
            USE_DATABASE = False
    
    # Get video alignment offset from database
    video_start_offset = 0.0  # Default to no offset
    if USE_DATABASE and db and not USE_WEBCAM:
        try:
            video_filename = os.path.basename(VIDEO_PATH)
            alignment_data = db.get_video_alignment(video_filename)
            if alignment_data:
                video_start_offset = alignment_data.get('start_time_offset', 0.0)
                print(f"📐 Video alignment offset found: {video_start_offset:.3f} seconds")
            else:
                print(f"📐 No alignment offset found for {video_filename}")
        except Exception as e:
            print(f"⚠️ Error retrieving video alignment: {e}")
    
    # Apply video start offset by seeking to the correct position
    if video_start_offset > 0 and not USE_WEBCAM:
        print(f"⏭️ Seeking to offset position: {video_start_offset:.3f} seconds")
        cap.set(cv2.CAP_PROP_POS_MSEC, video_start_offset * 1000)  # Seek to milliseconds
    
    # Set up skip_frames logic
    if skip_frames is None:
        skip_frames = 0  # Default: process every frame
    print(f"⚡ Skip frames setting: {skip_frames} (0 = process all frames)")
    
    # Initialize MediaPipe models
    hand_model = mp.solutions.hands.Hands()
    
    # Initialize face mesh based on configuration - using new FaceLandmarker API
    facemesh_model_name = config['detection'].get('facemesh_model', 'none').lower()
    USE_NEW_FACELANDMARKER_API = facemesh_model_name == 'mediapipe'
    face_mesh_model = None
    
    if facemesh_model_name == 'mediapipe':
        try:
            # Use new FaceLandmarker API
            base_options = python.BaseOptions(model_asset_path='checkpoints/face_landmarker.task')
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            face_mesh_model = vision.FaceLandmarker.create_from_options(options)
            print("✅ MediaPipe FaceLandmarker API enabled")
        except Exception as e:
            print(f"❌ FaceLandmarker initialization failed: {e}")
            face_mesh_model = None
            print("✅ Face detection disabled due to initialization failure")
    elif facemesh_model_name == 'yolo':
        # YOLO face detection/landmarks would use a different model
        # For now, we'll use the existing YOLO pose model which includes face keypoints
        face_mesh_model = "yolo_face"  # Placeholder to indicate YOLO face detection
        USE_NEW_FACELANDMARKER_API = False
        print("✅ YOLO Face detection enabled")
    else:
        print("✅ Face detection disabled")
    
    # Initialize emotion detector based on configuration
    emotion_detector = create_emotion_detector(config)
    if emotion_detector:
        if not emotion_detector.initialize_model():
            print("⚠️ Emotion detector initialization failed, disabling emotion detection")
            emotion_detector = None
    else:
        emotion_model_type = config['detection'].get('emotion_model', 'none')
        if emotion_model_type == 'mediapipe':
            print("✅ Using MediaPipe for emotion detection")
        else:
            print("✅ Emotion detection disabled")
    
    # Initialize pose detection based on configuration
    if USE_MEDIAPIPE_POSE:
        # MediaPipe pose detection
        pose_model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0, 1, or 2 (higher = more accurate but slower)
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        # YOLO pose detection
        pose_model = YOLO('checkpoints/yolo11n-pose.pt')
    

    
    # Initialize gesture timer with configurable report interval - COMMENTED OUT
    # report_interval_seconds = 30  # Change this to set report frequency (e.g., 60 for 1 minute, 300 for 5 minutes)
    # gesture_timer = GestureTimer(report_time_threshold=report_interval_seconds)
    # gesture_timer.set_fps(fps)  # Set FPS for accurate time calculations
    
    frame_count = 0
    processed_frames = 0
    start_time = time.time()
    
    while True: 
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply skip_frames logic
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            frame_count += 1
            continue  # Skip this frame
        
        frame_start_time = time.time()
        
        # Calculate current video time from video start offset
        current_video_time = video_start_offset + (frame_count / fps)
            
        # Convert to RGB for MediaPipe hands
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand detection with timing
        hand_start_time = time.time()
        multi_hand_landmarks = get_hand_landmarks(hand_model, img_rgb)
        hand_processing_time_ms = int((time.time() - hand_start_time) * 1000)

        # Pose detection with timing based on configuration (moved before face mesh)
        pose_start_time = time.time()
        if USE_MEDIAPIPE_POSE:
            pose_results = get_mp_pose_landmarks(pose_model, img_rgb)
        else:
            pose_results = pose_model(frame, verbose=False)
        pose_processing_time_ms = int((time.time() - pose_start_time) * 1000)

        # Face mesh detection with timing based on configuration
        face_detection_result = None
        multi_face_landmarks = None
        facemesh_processing_time_ms = 0
        if face_mesh_model:
            facemesh_start_time = time.time()
            if USE_NEW_FACELANDMARKER_API:
                # Use new FaceLandmarker API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                face_detection_result = face_mesh_model.detect(mp_image)
            else:  # YOLO face detection
                # For YOLO, we can extract face keypoints from pose results
                # YOLO pose includes face keypoints (nose, eyes, ears)
                multi_face_landmarks = extract_face_keypoints_from_yolo(pose_results)
            facemesh_processing_time_ms = int((time.time() - facemesh_start_time) * 1000)
        
        # Emotion detection with timing based on configuration
        emotion_start_time = time.time()
        emotions_in_frame = []
        emotion_scores = {'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0}
        
        if emotion_detector:
            emotions_in_frame = emotion_detector.detect_emotions(frame)
            # Convert to database format
            if emotions_in_frame:
                # Use the most confident detection
                best_detection = max(emotions_in_frame, key=lambda x: x.get('confidence', 0))
                emotion_scores = best_detection.get('emotions', emotion_scores)
        elif config['detection'].get('emotion_model', 'none') == 'mediapipe' and face_detection_result:
            # Use MediaPipe emotion detection with new FaceLandmarker API
            if face_detection_result.face_landmarks:
                for face_landmarks in face_detection_result.face_landmarks:
                    # Get face bounding box from landmarks
                    h, w, _ = frame.shape
                    x_coords = [landmark.x * w for landmark in face_landmarks]
                    y_coords = [landmark.y * h for landmark in face_landmarks]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Analyze facial expression using landmarks
                    emotions = analyze_facial_expression_mediapipe(face_landmarks, h, w)
                    
                    emotions_in_frame.append({
                        "box": (x_min, y_min, x_max - x_min, y_max - y_min),
                        "emotions": emotions
                    })
                
                if emotions_in_frame:
                    # Use the most confident detection
                    best_detection = max(emotions_in_frame, key=lambda x: max(x['emotions'].values()) if 'emotions' in x else 0)
                    if 'emotions' in best_detection:
                        emotion_scores = best_detection['emotions']
        
        emotion_processing_time_ms = int((time.time() - emotion_start_time) * 1000)

        # Convert landmarks to database format
        left_hand_landmarks, right_hand_landmarks = convert_hand_landmarks_to_dict(multi_hand_landmarks)
        pose_landmarks = convert_pose_landmarks_to_dict(pose_results)
        
        # Convert face mesh landmarks to database format
        facemesh_landmarks = None
        if face_detection_result and face_detection_result.face_landmarks:
            # New FaceLandmarker API format: convert to database format
            face_landmarks = face_detection_result.face_landmarks[0]  # Take first face
            facemesh_landmarks = []
            for landmark in face_landmarks:
                facemesh_landmarks.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z) if hasattr(landmark, 'z') else 0.0,
                    "confidence": float(landmark.visibility) if hasattr(landmark, 'visibility') else 1.0
                })
        elif multi_face_landmarks:
            # YOLO format: already converted to dictionary format
            facemesh_landmarks = multi_face_landmarks[0]
        
        # Check for bad gestures (for database storage)
        low_wrist_detected = low_wrists(frame, multi_hand_landmarks)
        turtle_neck_detected = turtle_neck(frame, pose_results)
        hunched_back_detected = hunched_back(frame, pose_results)
        fingers_up_detected = fingers_pointing_up_to_the_sky(frame, multi_hand_landmarks)
        
        # Calculate total processing time
        processing_time_ms = int((time.time() - frame_start_time) * 1000)
        
        # Save processed frames to database (using batch insert)
        if USE_DATABASE and db:
            try:
                success = db.add_frame_to_batch(
                    session_id=session_id,
                    frame_number=frame_count,
                    video_file=video_file,
                    original_time=(frame_count - 1) / fps,
                    synced_time=((frame_count - 1) / fps) + video_start_offset,
                    left_hand_landmarks=left_hand_landmarks,
                    right_hand_landmarks=right_hand_landmarks,
                    pose_landmarks=pose_landmarks,
                    facemesh_landmarks=facemesh_landmarks,
                    emotions=emotion_scores,
                    bad_gestures={
                        'low_wrists': low_wrist_detected,
                        'turtle_neck': turtle_neck_detected,
                        'hunched_back': hunched_back_detected,
                        'fingers_pointing_up': fingers_up_detected
                    },
                    processing_time_ms=processing_time_ms,
                    hand_processing_time_ms=hand_processing_time_ms,
                    pose_processing_time_ms=pose_processing_time_ms,
                    facemesh_processing_time_ms=facemesh_processing_time_ms,
                    emotion_processing_time_ms=emotion_processing_time_ms,
                    hand_model=config['detection'].get('hand_model', 'mediapipe'),
                    pose_model=config['detection'].get('pose_model', 'yolo'),
                    facemesh_model=config['detection'].get('facemesh_model', 'none'),
                    emotion_model=emotion_detector.model_name if emotion_detector else config['detection'].get('emotion_model', 'none')
                )
                
                if success:
                    # Batch was flushed automatically
                    pass
                    
            except Exception as e:
                print(f"❌ Database error on frame {frame_count}: {e}")
        
        # Draw all detections
        draw_mp_hand(frame, multi_hand_landmarks)
        if USE_MEDIAPIPE_POSE:
            draw_mp_pose(frame, pose_results)  # Draw MediaPipe pose
        else:
            draw_yolo_pose(frame, pose_results)  # Draw YOLO pose
        
        # Draw face mesh if available
        if face_detection_result and face_detection_result.face_landmarks:
            # Use new FaceLandmarker API drawing
            draw_facelandmarker_landmarks(frame, face_detection_result)
        elif multi_face_landmarks:
            # YOLO format
            draw_yolo_face_keypoints(frame, multi_face_landmarks)
            
        # Draw emotion results if available
        if emotions_in_frame:
            draw_emotion_results(frame, emotions_in_frame)

        # Face mesh is already drawn above with the new API

        # Display real-time information
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display frame info
        cv2.putText(frame, f"Frame: {frame_count} | Time: {current_video_time:.1f}s | Skip: {skip_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Processing FPS: {current_fps:.1f} | Video FPS: {fps:.1f}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Display database status
        if USE_DATABASE and session_id:
            cv2.putText(frame, f"DB: {session_id} (saving all processed frames)", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Display landmark counts and processing times
        hand_count = len(multi_hand_landmarks) if multi_hand_landmarks else 0
        
        # Check pose detection for both YOLO and MediaPipe formats
        if isinstance(pose_results, list):
            # YOLO format
            pose_detected = "✓" if pose_results and pose_results[0].keypoints is not None else "✗"
        else:
            # MediaPipe format
            pose_detected = "✓" if pose_results and pose_results.pose_landmarks else "✗"
            
        # Count faces from new FaceLandmarker API or YOLO
        if face_detection_result and face_detection_result.face_landmarks:
            facemesh_count = len(face_detection_result.face_landmarks)
        elif multi_face_landmarks:
            facemesh_count = len(multi_face_landmarks)
        else:
            facemesh_count = 0
        
        cv2.putText(frame, f"Hands: {hand_count} | Pose: {pose_detected} | Face: {facemesh_count}", 
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display processing times breakdown
        cv2.putText(frame, f"Hand: {hand_processing_time_ms}ms | Pose: {pose_processing_time_ms}ms", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"FaceMesh: {facemesh_processing_time_ms}ms | Emotion: {emotion_processing_time_ms}ms", 
                   (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Check for bad gestures and update timer - COMMENTED OUT
        # low_wrists_detected = low_wrists(frame, multi_hand_landmarks)
        # turtle_neck_detected = turtle_neck(frame, pose_results)
        # hunched_back_detected = hunched_back(frame, pose_results)
        # fingers_up_detected = fingers_pointing_up_to_the_sky(frame, multi_hand_landmarks)
        
        # Update gesture timer - COMMENTED OUT
        # if low_wrists_detected:
        #     gesture_timer.start_gesture('low_wrists')
        # else:
        #     gesture_timer.stop_gesture('low_wrists')
            
        # if turtle_neck_detected:
        #     gesture_timer.start_gesture('turtle_neck')
        # else:
        #     gesture_timer.stop_gesture('turtle_neck')
            
        # if hunched_back_detected:
        #     gesture_timer.start_gesture('hunched_back')
        # else:
        #     gesture_timer.stop_gesture('hunched_back')
            
        # if fingers_up_detected:
        #     gesture_timer.start_gesture('fingers_pointing_up')
        # else:
        #     gesture_timer.stop_gesture('fingers_pointing_up')
        
        # Record frame data for histogram - COMMENTED OUT
        # gesture_timer.record_gesture_frame('low_wrists', low_wrists_detected)
        # gesture_timer.record_gesture_frame('turtle_neck', turtle_neck_detected)
        # gesture_timer.record_gesture_frame('hunched_back', hunched_back_detected)
        # gesture_timer.record_gesture_frame('fingers_pointing_up', fingers_up_detected)
        
        # Record emotion data - COMMENTED OUT
        # gesture_timer.record_emotion(emotions_in_frame)
        

        
        # Update frame and check for report generation - COMMENTED OUT
        # gesture_timer.update_frame()
        # Display real-time timer and frame information - COMMENTED OUT
        # current_time = time.time()
        # session_duration = current_time - gesture_timer.session_start_time
        # time_until_next_report = gesture_timer.report_time_threshold - (current_time - gesture_timer.last_report_time)
        
        # Calculate actual FPS from frame count and time - COMMENTED OUT
        # actual_fps = frame_count / session_duration if session_duration > 0 else 0
        
        # Format time displays - COMMENTED OUT
        # session_minutes = int(session_duration // 60)
        # session_seconds = int(session_duration % 60)
        # countdown_minutes = int(time_until_next_report // 60)
        # countdown_seconds = int(time_until_next_report % 60)
        
        # Display session timer and frame info - COMMENTED OUT
        # cv2.putText(frame, f"Session: {session_minutes:02d}:{session_seconds:02d} | Frames: {frame_count}", 
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display actual FPS - COMMENTED OUT
        # cv2.putText(frame, f"Actual FPS: {actual_fps:.1f}", 
        #            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display countdown to next report - COMMENTED OUT
        # if time_until_next_report > 0:
        #     cv2.putText(frame, f"Next Report: {countdown_minutes:02d}:{countdown_seconds:02d}", 
        #                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # else:
        #     cv2.putText(frame, "Generating Report...", 
        #                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display current posture statistics - COMMENTED OUT
        # percentages = gesture_timer.get_current_percentages()
        # y_offset = 110  # Moved down to make room for timer and FPS info
        # cv2.putText(frame, f"Low Wrists: {percentages['low_wrists']:.1f}%", 
        #            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # cv2.putText(frame, f"Turtle Neck: {percentages['turtle_neck']:.1f}%", 
        #            (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # cv2.putText(frame, f"Hunched Back: {percentages['hunched_back']:.1f}%", 
        #            (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # cv2.putText(frame, f"Fingers Up: {percentages['fingers_pointing_up']:.1f}%", 
        #            (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # cv2.putText(frame, f"Total Bad Posture: {sum(percentages.values()):.1f}%", 
        #            (10, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Display emotion statistics - COMMENTED OUT
        # emotion_stats = gesture_timer.get_emotion_statistics()
        # y_offset_emotions = y_offset + 120  # Start emotions display below posture stats
        # cv2.putText(frame, f"Emotions:", 
        #            (10, y_offset_emotions), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Display top 3 emotions - COMMENTED OUT
        # sorted_emotions = sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True)
        # for i, (emotion, percentage) in enumerate(sorted_emotions[:3]):
        #     if percentage > 0:
        #         cv2.putText(frame, f"{emotion.capitalize()}: {percentage:.1f}%", 
        #                    (10, y_offset_emotions + 20 + i * 15), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        

        
        # Video display 
        cv2.imshow("Multi-Model Detection", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    video_duration = frame_count / fps if fps > 0 else 0
    
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"📹 Video: {video_file}")
    print(f"⏱️  Total Processing Time: {total_time:.1f} seconds")
    print(f"🎬 Video Duration: {video_duration:.1f} seconds")
    print(f"📈 Frames Processed: {frame_count}")
    print(f"⚡ Average Processing FPS: {avg_fps:.1f}")
    print(f"🎯 Video FPS: {fps:.1f}")
    
    # Database summary
    if USE_DATABASE and db and session_id:
        print(f"\n🗄️ DATABASE EXPORT:")
        print(f"📝 Session ID: {session_id}")
        
        try:
            # Get session data count
            session_data = db.get_session_data(session_id)
            if session_data:
                saved_frames = len(session_data)
                print(f"💾 Frames Saved: {saved_frames}")
                if skip_frames > 0:
                    print(f"📊 Data Reduction: Saved every {skip_frames + 1} frame(s) ({saved_frames}/{frame_count})")
                else:
                    print(f"📊 Data Storage: Saved all frames ({saved_frames}/{frame_count})")
                
                # Calculate total landmarks saved
                total_landmarks = 0
                for row in session_data:
                    if row.get('left_hand_landmarks'):
                        total_landmarks += len(row['left_hand_landmarks'])
                    if row.get('right_hand_landmarks'):
                        total_landmarks += len(row['right_hand_landmarks'])
                    if row.get('pose_landmarks'):
                        total_landmarks += len(row['pose_landmarks'])
                
                print(f"🎯 Total Landmarks Saved: {total_landmarks:,}")
                print(f"✅ Database export completed successfully!")
            else:
                print("❌ No data found in database")
                
        except Exception as e:
            print(f"❌ Error retrieving database summary: {e}")
    else:
        print(f"\n💡 Database export was disabled")
    
    # Cleanup database and flush remaining batch data
    if USE_DATABASE and db:
        try:
            db.close()
        except Exception as e:
            print(f"❌ Error closing database: {e}")
    
    # Cleanup emotion detector
    if emotion_detector:
        emotion_detector.cleanup()
        print("🧹 Emotion detector cleaned up")
    
    print("="*60)
    print("🎵 Session completed!")
    print("="*60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Musician Tracking Detection System')
    parser.add_argument('--config', '-c', default='config.yaml', 
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--skip-frames', type=int, 
                       help='Number of frames to skip (overrides config)')
    
    args = parser.parse_args()
    
    main(skip_frames=args.skip_frames, config_path=args.config)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  