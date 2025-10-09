#!/usr/bin/env python3
"""
Enhanced 3D Bad Gesture Detection System
Uses MediaPipe's x, y, z coordinates for scale-invariant detection
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import defaultdict

# Import enhanced 3D detector
from src.enhanced_3d_bad_gesture_detector import EnhancedBadGestureDetector3D, GeometryUtils


class BadGestureDetector3D:
    """
    Enhanced 3D bad gesture detection using MediaPipe's x, y, z coordinates
    Drop-in replacement for BadGestureDetector with improved accuracy
    """

    def __init__(self, config: Dict[str, Any] = None, draw_on_frame: bool = True):
        """
        Initialize BadGestureDetector3D

        Args:
            config: Configuration dictionary with thresholds and detection flags
            draw_on_frame: Whether to draw detection visualizations on frame (default: True)
        """
        # Default configuration (compatible with original BadGestureDetector)
        default_config = {
            'detect_low_wrists': True,
            'detect_turtle_neck': True,
            'detect_hunched_back': True,
            'detect_fingers_pointing_up': True,
            'draw_on_frame': draw_on_frame,  # Control drawing behavior
            'thresholds': {
                # Enhanced 3D thresholds (scale-invariant)
                'head_forward_ratio': 0.20,
                'neck_forward_angle': 25,
                'spine_vertical_angle': 15,
                'upper_lower_spine_angle': 160,
                'wrist_drop_ratio': 0.08,
                'elbow_wrist_angle_min': 100,
                'finger_vertical_angle': 60,
            }
        }

        # Merge with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            if 'thresholds' in config:
                self.config['thresholds'].update(config['thresholds'])

        # Initialize enhanced 3D detector
        enhanced_config = {
            'thresholds': self.config['thresholds']
        }
        self.detector_3d = EnhancedBadGestureDetector3D(config=enhanced_config)

        # Initialize tracking (compatible with original)
        self.gesture_timers = GestureTimer()

        # Geometry utilities
        self.geom = GeometryUtils()

    def detect_all_gestures(self, frame: np.ndarray,
                           hand_landmarks: Any = None,
                           pose_landmarks: Any = None) -> Dict[str, bool]:
        """
        Detect all bad gestures using 3D coordinates from MediaPipe

        Args:
            frame: Input image frame
            hand_landmarks: Hand 3D world landmarks (multi_hand_world_landmarks)
            pose_landmarks: Pose 3D world landmarks (pose_world_landmarks)

        Returns:
            Dictionary with gesture detection results (compatible with original)
        """
        results = {
            'low_wrists': False,
            'turtle_neck': False,
            'hunched_back': False,
            'fingers_pointing_up': False
        }

        # Extract 3D landmarks from MediaPipe
        landmarks_3d = self._extract_3d_landmarks(frame, hand_landmarks, pose_landmarks)

        if not landmarks_3d:
            # Fallback: no valid 3D data
            self.gesture_timers.update_frame_detections(results)
            return results

        # Run enhanced 3D detection
        detection_results = self.detector_3d.detect_all_3d(landmarks_3d)

        # Convert to original format
        results['turtle_neck'] = detection_results['detections'].get('turtle_neck', False)
        results['hunched_back'] = detection_results['detections'].get('hunched_back', False)
        results['low_wrists'] = detection_results['detections'].get('low_wrists', False)
        results['fingers_pointing_up'] = detection_results['detections'].get('fingers_up', False)

        # Store detailed analysis for access
        self.last_detailed_analysis = detection_results['detailed_analysis']

        # Draw visualization on frame (optional, controlled by config)
        if self.config.get('draw_on_frame', True):
            self._draw_detections_on_frame(frame, results, detection_results)

        # Update gesture timers
        self.gesture_timers.update_frame_detections(results)

        return results

    def _extract_3d_landmarks(self, frame: np.ndarray, hand_landmarks: Any, pose_landmarks: Any) -> Dict[str, np.ndarray]:
        """
        Extract 3D landmarks from MediaPipe world coordinates

        Args:
            frame: Input image frame
            hand_landmarks: Hand 3D world landmarks (multi_hand_world_landmarks)
            pose_landmarks: Pose 3D world landmarks (pose_world_landmarks)

        MediaPipe world landmarks provide:
        - x, y, z: Real 3D coordinates in meters
        - Origin: Hip midpoint for pose

        Returns:
            Dictionary mapping landmark names to 3D numpy arrays
        """
        landmarks_3d = {}
        h, w, _ = frame.shape

        # Extract pose landmarks (MediaPipe world landmarks for real 3D coordinates)
        if pose_landmarks and hasattr(pose_landmarks, 'landmark'):
            landmarks = pose_landmarks.landmark

            # MediaPipe Pose landmark indices
            # 0: nose, 7: left_ear, 8: right_ear
            # 11: left_shoulder, 12: right_shoulder
            # 13: left_elbow, 14: right_elbow
            # 15: left_wrist, 16: right_wrist
            # 23: left_hip, 24: right_hip

            landmark_mapping = {
                'nose': 0,
                'left_ear': 7,
                'right_ear': 8,
                'left_shoulder': 11,
                'right_shoulder': 12,
                'left_elbow': 13,
                'right_elbow': 14,
                'left_wrist': 15,
                'right_wrist': 16,
                'left_hip': 23,
                'right_hip': 24,
            }

            for name, idx in landmark_mapping.items():
                lm = landmarks[idx]
                # Use actual 3D coordinates from MediaPipe
                # x, y are normalized [0, 1], z is depth
                # Convert to world coordinates (keep scale)
                landmarks_3d[name] = np.array([
                    lm.x,  # Normalized x
                    lm.y,  # Normalized y
                    lm.z   # Depth (relative to hip)
                ])

        # Extract hand landmarks if available
        if hand_landmarks:
            for hand_idx, hand_landmarks_obj in enumerate(hand_landmarks):
                # MediaPipe Hand landmark indices
                # 0: wrist, 4: thumb_tip, 5: index_mcp, 8: index_tip
                # 9: middle_mcp, 12: middle_tip, 13: ring_mcp, 16: ring_tip
                # 17: pinky_mcp, 20: pinky_tip

                hand_landmark_mapping = {
                    'wrist': 0,
                    'thumb_tip': 4,
                    'thumb_mcp': 2,
                    'index_mcp': 5,
                    'index_tip': 8,
                    'index_dip': 7,
                    'middle_mcp': 9,
                    'middle_tip': 12,
                    'middle_dip': 11,
                    'ring_mcp': 13,
                    'ring_tip': 16,
                    'ring_dip': 15,
                    'pinky_mcp': 17,
                    'pinky_tip': 20,
                    'pinky_dip': 19,
                }

                for name, idx in hand_landmark_mapping.items():
                    lm = hand_landmarks_obj.landmark[idx]
                    key = f'hand_{hand_idx}_{name}'
                    landmarks_3d[key] = np.array([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

        return landmarks_3d

    def _draw_detections_on_frame(self, frame: np.ndarray, results: Dict[str, bool],
                                   detection_results: Dict):
        """
        Draw detection visualizations on frame (compatible with original)
        """
        h, w, _ = frame.shape

        # Draw detection alerts with enhanced information
        y_offset = 30

        if results['turtle_neck']:
            analysis = detection_results['detailed_analysis']['turtle_neck']
            severity = analysis.get('severity', 'unknown')
            ratio = analysis.get('measurements', {}).get('head_forward_ratio', 0)

            text = f"TURTLE NECK ({severity.upper()})"
            detail = f"Forward: {ratio*100:.1f}% of shoulder width"

            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, detail, (10, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 60

        if results['hunched_back']:
            analysis = detection_results['detailed_analysis']['hunched_back']
            severity = analysis.get('severity', 'unknown')
            issues = analysis.get('issues', [])

            text = f"HUNCHED BACK ({severity.upper()})"
            detail = f"Issues: {', '.join(issues)}"

            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, detail, (10, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 60

        if results['low_wrists']:
            analysis = detection_results['detailed_analysis']['low_wrists']
            severity = analysis.get('severity', 'unknown')
            affected = analysis.get('affected_hands', [])

            text = f"LOW WRISTS ({severity.upper()})"
            detail = f"Affected: {', '.join(affected)}"

            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, detail, (10, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 60

        if results['fingers_pointing_up']:
            analysis = detection_results['detailed_analysis']['fingers_up']
            severity = analysis.get('severity', 'unknown')
            num_fingers = len(analysis.get('affected_fingers', []))

            text = f"FINGERS POINTING UP ({severity.upper()})"
            detail = f"{num_fingers} fingers affected"

            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, detail, (10, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y_offset += 60

    def get_gesture_statistics(self) -> Dict[str, float]:
        """Get current gesture detection statistics"""
        return self.gesture_timers.get_current_percentages()

    def get_gesture_counts(self) -> Dict[str, int]:
        """Get total gesture detection counts"""
        return self.gesture_timers.get_total_detections()

    def get_detailed_analysis(self) -> Dict:
        """Get detailed analysis from last detection"""
        return getattr(self, 'last_detailed_analysis', {})

    def get_posture_score(self) -> Dict:
        """Get overall posture score"""
        if hasattr(self, 'last_detailed_analysis'):
            return self.detector_3d.calculate_posture_score(self.last_detailed_analysis)
        return {'score': 100, 'grade': 'A', 'feedback': 'No data available'}

    def reset_statistics(self):
        """Reset all gesture tracking statistics"""
        self.gesture_timers.reset()

    def cleanup(self):
        """Cleanup resources"""
        # Nothing to cleanup for this detector
        pass


class GestureTimer:
    """
    Timer and statistics tracker for bad gesture detection
    Compatible with original implementation
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
