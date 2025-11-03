#!/usr/bin/env python3
"""
Bad Gesture Detection System
Modular implementation for detecting bad postures in musical performance
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import defaultdict


class BadGestureDetector:
    """
    Comprehensive bad gesture detection system for musician tracking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize BadGestureDetector
        
        Args:
            config: Configuration dictionary with thresholds and detection flags
        """
        # Default configuration
        default_config = {
            'detect_low_wrists': True,
            'detect_turtle_neck': True,
            'detect_hunched_back': True,
            'detect_fingers_pointing_up': True,
            'thresholds': {
                'low_wrist_threshold': 0.1,
                'turtle_neck_angle': 30,
                'hunched_back_angle': 160,  # Angle below this is hunched
                'finger_pointing_threshold': 0.8
            }
        }
        
        # Merge with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            if 'thresholds' in config:
                self.config['thresholds'].update(config['thresholds'])
        
        # Initialize tracking
        self.gesture_timers = GestureTimer()
    
    def detect_all_gestures(self, frame: np.ndarray, 
                           hand_landmarks: Any = None, 
                           pose_results: Any = None) -> Dict[str, bool]:
        """
        Detect all bad gestures for a single frame
        
        Args:
            frame: Input image frame
            hand_landmarks: Hand detection results (MediaPipe format)
            pose_results: Pose detection results (YOLO or MediaPipe format)
            
        Returns:
            Dictionary with gesture detection results
        """
        results = {
            'low_wrists': False,
            'turtle_neck': False,
            'hunched_back': False,
            'fingers_pointing_up': False
        }
        
        # Detect low wrists
        if self.config['detect_low_wrists'] and hand_landmarks:
            results['low_wrists'] = self._detect_low_wrists(frame, hand_landmarks)
        
        # Detect turtle neck
        if self.config['detect_turtle_neck'] and pose_results:
            results['turtle_neck'] = self._detect_turtle_neck(frame, pose_results)
        
        # Detect hunched back
        if self.config['detect_hunched_back'] and pose_results:
            results['hunched_back'] = self._detect_hunched_back(frame, pose_results)
        
        # Detect fingers pointing up
        if self.config['detect_fingers_pointing_up'] and hand_landmarks:
            results['fingers_pointing_up'] = self._detect_fingers_pointing_up(frame, hand_landmarks)
        
        # Update gesture timers
        self.gesture_timers.update_frame_detections(results)
        
        return results
    
    def _extract_unified_pose_landmarks(self, pose_results: Any, img_width: int, img_height: int) -> Optional[Dict]:
        """
        Extract pose landmarks in unified format from both YOLO and MediaPipe results
        
        Returns:
            Dictionary with landmark positions as pixel coordinates or None if no valid pose
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
    
    def _detect_turtle_neck(self, frame: np.ndarray, pose_results: Any) -> bool:
        """
        Detect turtle neck posture using improved 2D analysis with perspective awareness
        """
        detected = False
        h, w, _ = frame.shape

        # Extract landmarks in unified format
        landmarks = self._extract_unified_pose_landmarks(pose_results, w, h)
        if not landmarks:
            return detected

        # Convert to array format for better calculations
        landmark_array = {}
        required_points = ['nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']

        for point in required_points:
            if point in landmarks:
                landmark_array[point] = np.array(landmarks[point])
            else:
                return detected

        # Check if all keypoints are valid
        if not all(pt[0] > 0 and pt[1] > 0 for pt in landmark_array.values()):
            return detected

        # Use improved turtle neck detection logic
        left_shoulder = landmark_array['left_shoulder']
        right_shoulder = landmark_array['right_shoulder']
        nose = landmark_array['nose']

        # Calculate shoulder line
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_length = np.linalg.norm(shoulder_vector)

        if shoulder_length > 0:
            # Project nose onto shoulder line to measure forward distance
            nose_to_shoulder = nose - left_shoulder
            shoulder_unit = shoulder_vector / shoulder_length
            projection_length = np.dot(nose_to_shoulder, shoulder_unit)
            projected_point = left_shoulder + projection_length * shoulder_unit

            # Calculate forward distance (perpendicular to shoulder line)
            forward_distance = np.linalg.norm(nose - projected_point)

            # Dynamic threshold based on shoulder width (accounts for camera distance)
            forward_threshold = shoulder_length * 0.35  # More conservative than 0.25

            # Additional check: head should be significantly forward
            head_y = (landmark_array['left_ear'][1] + landmark_array['right_ear'][1]) / 2
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

            # Head should be above shoulders for valid detection
            head_above_shoulders = head_y < shoulder_y

            detected = forward_distance > forward_threshold and head_above_shoulders

            # Debug visualization (optional)
            if detected:
                cv2.circle(frame, (int(nose[0]), int(nose[1])), 5, (0, 0, 255), -1)
                cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                        (int(right_shoulder[0]), int(right_shoulder[1])), (255, 0, 0), 2)
                cv2.line(frame, (int(nose[0]), int(nose[1])),
                        (int(projected_point[0]), int(projected_point[1])), (255, 0, 255), 2)
                cv2.putText(frame, "TURTLE NECK",
                           (int(nose[0] - 60), int(nose[1] - 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return detected
    
    def _detect_hunched_back(self, frame: np.ndarray, pose_results: Any) -> bool:
        """
        Detect hunched back posture using improved 2D analysis focusing on spine curvature and shoulder rounding
        """
        detected = False
        h, w, _ = frame.shape

        # Extract landmarks in unified format
        landmarks = self._extract_unified_pose_landmarks(pose_results, w, h)
        if not landmarks:
            return detected

        # Convert to array format and check required points
        required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_ear', 'right_ear']
        landmark_array = {}

        for point in required_points:
            if point in landmarks:
                landmark_array[point] = np.array(landmarks[point])
            else:
                return detected

        # Check if all keypoints are valid
        if not all(pt[0] > 0 and pt[1] > 0 for pt in landmark_array.values()):
            return detected

        # Calculate key centers
        left_shoulder = landmark_array['left_shoulder']
        right_shoulder = landmark_array['right_shoulder']
        left_hip = landmark_array['left_hip']
        right_hip = landmark_array['right_hip']
        left_ear = landmark_array['left_ear']
        right_ear = landmark_array['right_ear']

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        ear_center = (left_ear + right_ear) / 2

        # 1. SHOULDER ANGLE ANALYSIS (Primary indicator)
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        shoulder_angle_degrees = np.degrees(shoulder_angle)

        # Rounded shoulders: shoulders rotated forward significantly
        shoulder_rounded_threshold = 12  # degrees (more conservative)
        is_shoulders_rounded = abs(shoulder_angle_degrees) > shoulder_rounded_threshold

        # 2. SPINE CURVATURE ANALYSIS (Primary indicator)
        spine_vector = hip_center - shoulder_center
        spine_length = np.linalg.norm(spine_vector)

        if spine_length > 0:
            # Calculate spine angle relative to vertical
            spine_angle = np.arctan2(spine_vector[0], spine_vector[1])  # Relative to vertical
            spine_angle_degrees = np.degrees(spine_angle)

            # Forward spine curvature: spine bent forward
            spine_curved_threshold = 15  # degrees (more conservative for 2D)
            is_spine_curved = abs(spine_angle_degrees) > spine_curved_threshold

            # 3. HEAD FORWARD POSITION (Secondary indicator)
            head_to_spine_vector = ear_center - shoulder_center

            # Project head onto spine line
            spine_unit = spine_vector / spine_length
            projection_length = np.dot(head_to_spine_vector, spine_unit)
            projected_point = shoulder_center + projection_length * spine_unit

            # Calculate forward distance
            forward_vector = ear_center - projected_point
            forward_distance = np.linalg.norm(forward_vector)

            # Head forward threshold (adaptive based on spine length)
            head_forward_threshold = spine_length * 0.3  # More conservative
            is_head_forward = forward_distance > head_forward_threshold

            # 4. COMBINED DETECTION LOGIC
            # Primary: spine curvature OR shoulder rounding
            # Enhanced: both spine and shoulders, or head forward with one primary
            detected = (is_spine_curved and is_shoulders_rounded) or \
                      (is_head_forward and (is_spine_curved or is_shoulders_rounded))

            # Debug visualization (optional)
            if detected:
                # Draw spine line
                cv2.line(frame, (int(shoulder_center[0]), int(shoulder_center[1])),
                        (int(hip_center[0]), int(hip_center[1])), (0, 255, 0), 2)
                # Draw shoulder line
                cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                        (int(right_shoulder[0]), int(right_shoulder[1])), (255, 0, 0), 2)
                # Draw head position
                cv2.circle(frame, (int(ear_center[0]), int(ear_center[1])), 5, (0, 0, 255), -1)
                cv2.putText(frame, "HUNCHED BACK",
                           (int(shoulder_center[0] - 60), int(shoulder_center[1] - 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return detected
    
    def _detect_low_wrists(self, frame: np.ndarray, hand_landmarks: Any) -> bool:
        """
        Detect low wrist posture using hand landmarks
        """
        detected = False
        if hand_landmarks:
            for hand_landmarks_obj in hand_landmarks:
                # Check if wrist (landmark 0) is lower than knuckle (landmark 9) 
                wrist = hand_landmarks_obj.landmark[0]
                middle_knuckle = hand_landmarks_obj.landmark[9]
                if wrist.y > middle_knuckle.y:  # Higher y value means lower position
                    detected = True
                    # Convert normalized coordinates to pixel coordinates
                    wrist_x = int(wrist.x * frame.shape[1])
                    wrist_y = int(wrist.y * frame.shape[0])
                    
                    # Display alarm text at wrist location
                    cv2.putText(frame, "LOW-WRIST", 
                                (wrist_x - 40, wrist_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # Add warning background rectangle around text
                    cv2.rectangle(frame, (wrist_x - 45, wrist_y - 25), 
                                    (wrist_x + 55, wrist_y + 5), (0, 0, 255), 1)
        return detected
    
    def _detect_fingers_pointing_up(self, frame: np.ndarray, hand_landmarks: Any) -> bool:
        """
        Detect fingers pointing up to the sky using hand landmarks
        """
        detected = False
        if hand_landmarks:
            for hand_landmarks_obj in hand_landmarks:
                pinky_tip = hand_landmarks_obj.landmark[20]
                pinky_mcp = hand_landmarks_obj.landmark[17]
                if pinky_tip.y < pinky_mcp.y:
                    detected = True
                    cv2.putText(frame, "FINGERS POINTING UP TO THE SKY", 
                                (int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(pinky_tip.x * frame.shape[1] - 20), int(pinky_tip.y * frame.shape[0] - 20)), 
                                 (int(pinky_tip.x * frame.shape[1] + 20), int(pinky_tip.y * frame.shape[0] + 20)), (0, 0, 255), 1)
        return detected
    
    def get_gesture_statistics(self) -> Dict[str, float]:
        """Get current gesture detection statistics"""
        return self.gesture_timers.get_current_percentages()
    
    def get_gesture_counts(self) -> Dict[str, int]:
        """Get total gesture detection counts"""
        return self.gesture_timers.get_total_detections()
    
    def reset_statistics(self):
        """Reset all gesture tracking statistics"""
        self.gesture_timers.reset()

    def get_detailed_analysis(self) -> Dict:
        """
        Get detailed analysis of detected gestures

        Returns a simplified analysis for 2D detection (screen coordinates)
        Note: 2D detection has less detailed measurements than 3D
        """
        stats = self.get_gesture_statistics()
        counts = self.get_gesture_counts()

        analysis = {}
        gesture_names = ['turtle_neck', 'hunched_back', 'low_wrists', 'fingers_pointing_up']

        for gesture in gesture_names:
            detected = counts.get(gesture, 0) > 0
            percentage = stats.get(gesture, 0)

            analysis[gesture] = {
                'detected': detected,
                'severity': 'mild' if percentage < 30 else 'moderate' if percentage < 60 else 'severe',
                'confidence': min(1.0, percentage / 100),
                'measurements': {
                    'detection_percentage': percentage,
                    'total_frames': counts.get(gesture, 0)
                },
                'correction_needed': f"Address {gesture.replace('_', ' ')}" if detected else None
            }

        return analysis

    def get_posture_score(self) -> Dict:
        """
        Calculate overall posture score based on detection statistics

        Returns:
            Dictionary with score, grade, feedback, and deductions
        """
        stats = self.get_gesture_statistics()
        counts = self.get_gesture_counts()

        # Start with perfect score
        score = 100

        # Deduction amounts based on severity
        deductions = {
            'turtle_neck': {'mild': 10, 'moderate': 20, 'severe': 30},
            'hunched_back': {'mild': 15, 'moderate': 25, 'severe': 35},
            'low_wrists': {'mild': 8, 'moderate': 15, 'severe': 25},
            'fingers_pointing_up': {'mild': 5, 'moderate': 10, 'severe': 15}
        }

        detailed_deductions = []

        for gesture, percentage in stats.items():
            if counts.get(gesture, 0) > 0:  # Only deduct if gesture was detected
                # Determine severity based on percentage of time
                if percentage < 30:
                    severity = 'mild'
                elif percentage < 60:
                    severity = 'moderate'
                else:
                    severity = 'severe'

                deduction = deductions.get(gesture, {}).get(severity, 0)
                score -= deduction

                detailed_deductions.append({
                    'issue': gesture,
                    'severity': severity,
                    'points_lost': deduction,
                    'confidence': min(1.0, percentage / 100)
                })

        # Ensure score doesn't go below 0
        score = max(0, score)

        # Determine grade and feedback
        if score >= 90:
            grade = 'A'
            feedback = 'Excellent posture!'
        elif score >= 80:
            grade = 'B'
            feedback = 'Good posture with minor issues'
        elif score >= 70:
            grade = 'C'
            feedback = 'Fair posture, needs improvement'
        elif score >= 60:
            grade = 'D'
            feedback = 'Poor posture, multiple corrections needed'
        else:
            grade = 'F'
            feedback = 'Critical posture issues, immediate correction required'

        return {
            'score': score,
            'grade': grade,
            'feedback': feedback,
            'deductions': detailed_deductions
        }

    def cleanup(self):
        """Cleanup resources"""
        # Nothing to cleanup for this detector
        pass


class GestureTimer:
    """
    Timer and statistics tracker for bad gesture detection
    """
    
    def __init__(self):
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
        
        # Frame-by-frame tracking
        self.frame_gesture_data = {
            'low_wrists': [],
            'turtle_neck': [],
            'hunched_back': [],
            'fingers_pointing_up': []
        }
        
        self.total_frames = 0
        self.session_start_time = time.time()
    
    def update_frame_detections(self, detections: Dict[str, bool]):
        """Update frame-by-frame detection data"""
        self.total_frames += 1
        
        for gesture_type, detected in detections.items():
            # Update timing
            if detected:
                self.start_gesture(gesture_type)
            else:
                self.stop_gesture(gesture_type)
            
            # Record frame data
            self.frame_gesture_data[gesture_type].append(1 if detected else 0)
    
    def start_gesture(self, gesture_type: str):
        """Start timing a bad gesture"""
        if self.gesture_start_times[gesture_type] is None:
            self.gesture_start_times[gesture_type] = time.time()
    
    def stop_gesture(self, gesture_type: str):
        """Stop timing a bad gesture and add to total duration"""
        if self.gesture_start_times[gesture_type] is not None:
            duration = time.time() - self.gesture_start_times[gesture_type]
            self.gesture_durations[gesture_type] += duration
            self.gesture_start_times[gesture_type] = None
    
    def get_current_percentages(self) -> Dict[str, float]:
        """Get current percentages of bad gesture time"""
        total_time = time.time() - self.session_start_time
        if total_time == 0:
            return {gesture: 0.0 for gesture in self.gesture_durations.keys()}
        
        percentages = {}
        for gesture, duration in self.gesture_durations.items():
            percentages[gesture] = (duration / total_time) * 100
        
        return percentages
    
    def get_total_detections(self) -> Dict[str, int]:
        """Get total number of detections for each gesture"""
        return {
            gesture_type: sum(detections)
            for gesture_type, detections in self.frame_gesture_data.items()
        }
    
    def reset(self):
        """Reset all tracking data"""
        self.gesture_start_times = {key: None for key in self.gesture_start_times}
        self.gesture_durations = {key: 0 for key in self.gesture_durations}
        self.frame_gesture_data = {key: [] for key in self.frame_gesture_data}
        self.total_frames = 0
        self.session_start_time = time.time()