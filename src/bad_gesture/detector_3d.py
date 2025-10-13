#!/usr/bin/env python3
"""
Enhanced 3D Bad Gesture Detection System
Using scale-invariant metrics: angles, normalized ratios, and vector geometry

Key Improvements:
1. ✅ Angles instead of absolute distances
2. ✅ Normalized ratios (scale-invariant)
3. ✅ Vector projections for directional analysis
4. ✅ Self-calibrating to individual body proportions
5. ✅ View-invariant and perspective-invariant

This module is self-contained and requires no external dependencies beyond numpy and cv2.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Point3D:
    """3D point in world space"""
    x: float
    y: float
    z: float

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: 'Point3D') -> float:
        return np.linalg.norm(self.to_array() - other.to_array())


class GeometryUtils:
    """Utility functions for 3D geometry calculations"""

    # Global vertical direction in MediaPipe world coordinates
    # MediaPipe pose_world_landmarks use right-handed coordinate system:
    # - Origin: hip center
    # - X: lateral (positive = right)
    # - Y: vertical (positive = UP, unlike image coords where Y is down)
    # - Z: depth (positive = forward toward camera)
    VERTICAL = np.array([0, 1, 0])  # Y-up

    @staticmethod
    def build_body_frame(L_SH: np.ndarray, R_SH: np.ndarray,
                        L_HP: np.ndarray, R_HP: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a body-centric coordinate frame for view-invariant analysis

        Args:
            L_SH: Left shoulder position
            R_SH: Right shoulder position
            L_HP: Left hip position
            R_HP: Right hip position

        Returns:
            Tuple of (R, origin, C_sh) where:
                R: 3x3 rotation matrix with rows as basis vectors
                origin: Origin point (hip center)
                C_sh: Shoulder center point
        """
        C_hip = 0.5 * (L_HP + R_HP)
        C_sh = 0.5 * (L_SH + R_SH)

        # X-axis: left-to-right shoulder direction
        x_body = R_SH - L_SH
        x_body = x_body / (np.linalg.norm(x_body) + 1e-9)

        # Y-axis: make as close to global VERTICAL as possible but orthogonal to x_body
        y_body = GeometryUtils.VERTICAL - np.dot(GeometryUtils.VERTICAL, x_body) * x_body
        y_body = y_body / (np.linalg.norm(y_body) + 1e-9)

        # Z-axis: forward direction (cross product)
        z_body = np.cross(x_body, y_body)
        z_body = z_body / (np.linalg.norm(z_body) + 1e-9)

        # Rotation matrix: rows are basis vectors
        R = np.vstack([x_body, y_body, z_body])
        origin = C_hip

        return R, origin, C_sh

    @staticmethod
    def to_body_space(p: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
        """
        Transform a point from world space to body space

        Args:
            p: Point in world coordinates
            R: Rotation matrix (3x3) from build_body_frame
            origin: Origin point from build_body_frame

        Returns:
            Point in body-centric coordinates
        """
        return R @ (p - origin)

    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray, in_degrees: bool = True) -> float:
        """
        Calculate angle between two vectors
        Returns angle in degrees by default
        """
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return 0.0

        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm

        cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)

        return np.degrees(angle_rad) if in_degrees else angle_rad

    @staticmethod
    def project_point_onto_plane(point: np.ndarray, plane_normal: np.ndarray,
                                  plane_point: np.ndarray) -> np.ndarray:
        """
        Project a point onto a plane defined by normal and a point on the plane
        """
        normal_norm = np.linalg.norm(plane_normal)
        if normal_norm == 0:
            return point

        normal_unit = plane_normal / normal_norm

        # Vector from plane point to target point
        to_point = point - plane_point

        # Distance from point to plane
        distance = np.dot(to_point, normal_unit)

        # Project onto plane
        projected = point - distance * normal_unit

        return projected

    @staticmethod
    def signed_distance_to_plane(point: np.ndarray, plane_normal: np.ndarray,
                                  plane_point: np.ndarray) -> float:
        """
        Calculate signed distance from point to plane
        Positive = in direction of normal, negative = opposite
        """
        normal_norm = np.linalg.norm(plane_normal)
        if normal_norm == 0:
            return 0.0

        normal_unit = plane_normal / normal_norm
        to_point = point - plane_point

        return np.dot(to_point, normal_unit)

    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


class EnhancedBadGestureDetector3D:
    """
    Enhanced 3D bad gesture detection using scale-invariant metrics

    All thresholds are dimensionless ratios or angles:
    - Ratios: normalized by body segment lengths
    - Angles: naturally perspective-invariant
    - No absolute distance thresholds!
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize with scale-invariant thresholds

        All thresholds are ratios (0-1) or angles (degrees)
        """
        default_config = {
            'thresholds': {
                # TURTLE NECK thresholds (scale-invariant)
                'head_forward_ratio': 0.20,          # head forward > 20% of shoulder width
                'neck_forward_angle': 25,             # neck tilted > 25° from vertical
                'ear_shoulder_alignment_angle': 15,   # ears > 15° ahead of shoulders

                # HUNCHED BACK thresholds (angles only)
                'spine_vertical_angle': 15,           # spine > 15° from vertical
                'upper_lower_spine_angle': 160,       # upper/lower spine angle < 160° (bent)
                'shoulder_rotation_angle': 20,        # shoulders rotated > 20° forward
                'thoracic_curve_angle': 25,           # excessive upper back curve > 25°

                # LOW WRIST thresholds (normalized ratios)
                'wrist_drop_ratio': 0.08,             # wrist drop > 8% of forearm length
                'elbow_wrist_angle_min': 100,         # elbow bent < 100° (too much drop)
                'wrist_collapse_angle': 160,          # wrist-knuckle angle > 160° (collapsed)

                # FINGER POINTING thresholds (angles only)
                'finger_vertical_angle': 70,          # finger < 70° from vertical (pointing up) - more lenient
                'finger_hyperextension_angle': 200,   # finger > 200° extension
                'finger_curl_min_angle': 150,         # fingers should curve < 150°
            },
            'detection_sensitivity': {
                # Adjust severity thresholds
                'mild_multiplier': 1.2,      # threshold * 1.2 = mild
                'moderate_multiplier': 1.5,  # threshold * 1.5 = moderate
                'severe_multiplier': 2.0,    # threshold * 2.0 = severe
            }
        }

        self.config = default_config.copy()
        if config:
            self.config['thresholds'].update(config.get('thresholds', {}))
            self.config['detection_sensitivity'].update(config.get('detection_sensitivity', {}))

        self.geom = GeometryUtils()

    def calculate_body_proportions(self, landmarks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate individual's body segment lengths for normalization
        This makes detection scale-invariant!
        """
        proportions = {}

        # Shoulder width
        if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
            proportions['shoulder_width'] = np.linalg.norm(
                landmarks['right_shoulder'] - landmarks['left_shoulder']
            )

        # Torso length
        if all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            shoulder_center = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
            hip_center = (landmarks['left_hip'] + landmarks['right_hip']) / 2
            proportions['torso_length'] = np.linalg.norm(shoulder_center - hip_center)

        # Arm lengths
        for side in ['left', 'right']:
            shoulder_key = f'{side}_shoulder'
            elbow_key = f'{side}_elbow'
            wrist_key = f'{side}_wrist'

            if all(k in landmarks for k in [shoulder_key, elbow_key, wrist_key]):
                upper_arm = np.linalg.norm(landmarks[elbow_key] - landmarks[shoulder_key])
                forearm = np.linalg.norm(landmarks[wrist_key] - landmarks[elbow_key])
                proportions[f'{side}_upper_arm'] = upper_arm
                proportions[f'{side}_forearm'] = forearm
                proportions[f'{side}_arm_length'] = upper_arm + forearm

        # Neck length (if available)
        if all(k in landmarks for k in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']):
            ear_center = (landmarks['left_ear'] + landmarks['right_ear']) / 2
            shoulder_center = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
            proportions['neck_length'] = np.linalg.norm(ear_center - shoulder_center)

        return proportions

    def detect_all_3d(self, landmarks_3d: Dict[str, np.ndarray]) -> Dict:
        """
        Main detection function using view-invariant and scale-invariant 3D analysis

        Input: Dictionary of 3D points (from triangulation or MediaPipe world landmarks)
        Output: Comprehensive detection results with measurements

        Key Features:
        - Builds body-centric coordinate frame once for efficiency
        - All measurements are scale-invariant (normalized ratios and angles)
        - All measurements are view-invariant (body-space coordinates)
        """
        # Calculate body proportions first
        proportions = self.calculate_body_proportions(landmarks_3d)

        # Build body frame once (VIEW-INVARIANT coordinate system)
        # This transforms all measurements to be independent of camera viewpoint
        body_frame = None
        if all(k in landmarks_3d for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            R, origin, C_sh = self.geom.build_body_frame(
                landmarks_3d['left_shoulder'],
                landmarks_3d['right_shoulder'],
                landmarks_3d['left_hip'],
                landmarks_3d['right_hip']
            )
            body_frame = {'R': R, 'origin': origin, 'shoulder_center': C_sh}

        results = {
            'detections': {},
            'measurements': {},
            'confidence': {},
            'detailed_analysis': {},
            'body_proportions': proportions,
            'body_frame': body_frame  # Store for debugging/visualization
        }

        # Run each detection algorithm with shared body frame
        results['detections']['turtle_neck'], turtle_analysis = \
            self.detect_turtle_neck_3d(landmarks_3d, proportions, body_frame)
        results['detailed_analysis']['turtle_neck'] = turtle_analysis

        results['detections']['hunched_back'], hunched_analysis = \
            self.detect_hunched_back_3d(landmarks_3d, proportions, body_frame)
        results['detailed_analysis']['hunched_back'] = hunched_analysis

        results['detections']['low_wrists'], wrist_analysis = \
            self.detect_low_wrists_3d(landmarks_3d, proportions, body_frame)
        results['detailed_analysis']['low_wrists'] = wrist_analysis

        results['detections']['fingers_up'], finger_analysis = \
            self.detect_fingers_pointing_up_3d(landmarks_3d, proportions, body_frame)
        results['detailed_analysis']['fingers_up'] = finger_analysis

        return results

    def detect_turtle_neck_3d(self, landmarks: Dict[str, np.ndarray],
                              proportions: Dict[str, float],
                              body_frame: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        ENHANCED TURTLE NECK DETECTION (VIEW-INVARIANT)

        Uses body-space transformation for true view-invariance:
        1. Head forward ratio in body space (normalized by shoulder width)
        2. Neck angle from body-vertical
        3. Head position in body coordinates

        Scale-invariant AND view-invariant!

        Args:
            landmarks: Dictionary of 3D landmark positions
            proportions: Body proportions for normalization
            body_frame: Pre-computed body frame (optional, computed if None)
        """
        analysis = {
            'detected': False,
            'measurements': {},
            'severity': 'none',
            'correction_needed': {},
            'confidence': 0.0
        }

        # Check required landmarks
        required = ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']
        if not all(key in landmarks for key in required):
            return False, analysis

        # Get 3D points
        left_ear = landmarks['left_ear']
        right_ear = landmarks['right_ear']
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']

        # Use provided body frame or build new one
        if body_frame is not None:
            R = body_frame['R']
            origin = body_frame['origin']
            C_sh = body_frame['shoulder_center']
        elif all(k in landmarks for k in ['left_hip', 'right_hip']):
            left_hip = landmarks['left_hip']
            right_hip = landmarks['right_hip']
            R, origin, C_sh = self.geom.build_body_frame(
                left_shoulder, right_shoulder, left_hip, right_hip
            )
        else:
            # Fallback: cannot build body frame
            return False, analysis

        # Transform to body space
        head_center = (left_ear + right_ear) / 2
        head_center_b = self.geom.to_body_space(head_center, R, origin)
        shoulder_ctr_b = self.geom.to_body_space(C_sh, R, origin)
        L_SH_b = self.geom.to_body_space(left_shoulder, R, origin)
        R_SH_b = self.geom.to_body_space(right_shoulder, R, origin)

        # Shoulder width in body space
        shoulder_width = np.linalg.norm(R_SH_b - L_SH_b) + 1e-9

        # PRIMARY METHOD: Head forward ratio (in body space, +Z is forward)
        # This is the most direct and efficient measurement for turtle neck
        head_forward = head_center_b[2] - shoulder_ctr_b[2]  # Z-component in body space
        # Only consider forward motion (positive Z) as turtle neck
        # Negative Z (head behind shoulders) is not turtle neck
        head_forward_ratio = max(0, head_forward) / shoulder_width

        analysis['measurements']['head_forward_ratio'] = head_forward_ratio
        analysis['measurements']['head_forward_distance_relative'] = head_forward_ratio * 100  # as percentage
        analysis['measurements']['head_forward_raw_z'] = head_forward  # Store raw value for debugging

        # SECONDARY VALIDATION: Neck angle from body-vertical (optional, for confidence)
        # Only calculate if primary method detects issue (performance optimization)
        neck_vector_b = head_center_b - shoulder_ctr_b
        body_vertical = np.array([0, 1, 0])  # Y-axis in body space

        # Calculate angle in sagittal plane (YZ plane, remove X component)
        neck_sagittal = np.array([0, neck_vector_b[1], neck_vector_b[2]])
        neck_sagittal_norm = np.linalg.norm(neck_sagittal)

        if neck_sagittal_norm > 0:
            # Angle from vertical in sagittal plane
            neck_angle = self.geom.angle_between_vectors(neck_sagittal, body_vertical)
            # Forward tilt is when angle > 90° (tilted forward)
            neck_forward_angle = max(0, neck_angle - 90) if neck_angle > 90 else 0
        else:
            neck_forward_angle = 0

        analysis['measurements']['neck_forward_angle'] = neck_forward_angle

        # DETECTION LOGIC (primary method with optional validation)
        threshold_ratio = self.config['thresholds']['head_forward_ratio']
        threshold_angle = self.config['thresholds']['neck_forward_angle']

        # Primary detection: head forward ratio
        detected = head_forward_ratio > threshold_ratio

        # Calculate confidence based on how much threshold is exceeded
        if detected:
            ratio_excess = head_forward_ratio / threshold_ratio

            # Boost confidence if neck angle also agrees
            if neck_forward_angle > threshold_angle:
                analysis['confidence'] = min(1.0, 0.7 + 0.3 * (ratio_excess - 1.0))
            else:
                analysis['confidence'] = min(1.0, 0.5 + 0.5 * (ratio_excess - 1.0))

            # Calculate severity based on magnitude
            max_excess = ratio_excess

            if max_excess > 2.0:
                analysis['severity'] = 'severe'
            elif max_excess > 1.5:
                analysis['severity'] = 'moderate'
            else:
                analysis['severity'] = 'mild'

            analysis['detected'] = True

            # Generate correction (in relative terms)
            correction_ratio = head_forward_ratio - threshold_ratio
            analysis['correction_needed'] = {
                'action': 'Pull head back',
                'amount_relative': f"{correction_ratio * 100:.1f}% of shoulder width",
                'cue': "Align ears over shoulders, chin slightly tucked",
                'primary_metric': 'head_forward_ratio',
                'secondary_validation': 'neck_angle' if neck_forward_angle > threshold_angle else 'none'
            }

        return analysis['detected'], analysis

    def detect_hunched_back_3d(self, landmarks: Dict[str, np.ndarray],
                               proportions: Dict[str, float],
                               body_frame: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        ENHANCED HUNCHED BACK DETECTION (VIEW-INVARIANT)

        Uses body-space transformation for true view-invariance:
        1. Spine angle from body-vertical in body space
        2. Upper spine to lower spine angle (thoracic curve)
        3. Shoulder forward position in body space

        Scale-invariant AND view-invariant!

        Args:
            landmarks: Dictionary of 3D landmark positions
            proportions: Body proportions for normalization
            body_frame: Pre-computed body frame (optional, computed if None)
        """
        analysis = {
            'detected': False,
            'measurements': {},
            'severity': 'none',
            'issues': [],
            'confidence': 0.0
        }

        required = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        if not all(key in landmarks for key in required):
            return False, analysis

        # Get key points
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        left_hip = landmarks['left_hip']
        right_hip = landmarks['right_hip']

        # Use provided body frame or build new one
        if body_frame is not None:
            R = body_frame['R']
            origin = body_frame['origin']
            C_sh = body_frame['shoulder_center']
        else:
            R, origin, C_sh = self.geom.build_body_frame(
                left_shoulder, right_shoulder, left_hip, right_hip
            )

        # Transform to body space
        shoulder_ctr_b = self.geom.to_body_space(C_sh, R, origin)
        hip_ctr_b = self.geom.to_body_space((left_hip + right_hip) / 2, R, origin)
        L_SH_b = self.geom.to_body_space(left_shoulder, R, origin)
        R_SH_b = self.geom.to_body_space(right_shoulder, R, origin)

        body_vertical = np.array([0, 1, 0])  # Y-axis in body space

        # ANALYSIS 1: Spine angle from body-vertical
        spine_vector_b = shoulder_ctr_b - hip_ctr_b
        spine_angle = self.geom.angle_between_vectors(spine_vector_b, body_vertical)

        # Ideal is close to 0° (vertical in body space)
        analysis['measurements']['spine_angle_from_vertical'] = spine_angle

        threshold_spine = self.config['thresholds']['spine_vertical_angle']
        if spine_angle > threshold_spine:
            analysis['issues'].append('spine_tilted')
            analysis['measurements']['spine_tilt_severity'] = spine_angle / threshold_spine

        # ANALYSIS 2: Upper back curvature (if neck/head points available)
        if 'left_ear' in landmarks and 'right_ear' in landmarks:
            neck = (landmarks['left_ear'] + landmarks['right_ear']) / 2
            neck_b = self.geom.to_body_space(neck, R, origin)

            # Create two vectors in body space: neck-to-shoulder and shoulder-to-hip
            upper_spine_b = shoulder_ctr_b - neck_b
            lower_spine_b = hip_ctr_b - shoulder_ctr_b

            # Calculate angle between them
            upper_lower_angle = self.geom.angle_between_vectors(upper_spine_b, lower_spine_b)

            # Ideal is ~180° (straight line), hunched is < 160°
            analysis['measurements']['upper_lower_spine_angle'] = upper_lower_angle

            threshold_curve = self.config['thresholds']['upper_lower_spine_angle']
            if upper_lower_angle < threshold_curve:
                analysis['issues'].append('excessive_curve')
                curve_amount = 180 - upper_lower_angle
                analysis['measurements']['thoracic_curve_degrees'] = curve_amount

        # ANALYSIS 3: Shoulder forward position in body space
        # In body space, shoulders should be at Z≈0 (frontal plane)
        # Forward hunching means shoulders have positive Z
        shoulder_forward_z = shoulder_ctr_b[2]  # Z-component in body space
        shoulder_width = np.linalg.norm(R_SH_b - L_SH_b) + 1e-9
        shoulder_forward_ratio = shoulder_forward_z / shoulder_width

        analysis['measurements']['shoulder_forward_ratio'] = shoulder_forward_ratio

        # Check if shoulders are significantly forward
        threshold_rotation = self.config['thresholds']['shoulder_rotation_angle']
        # Convert angle threshold to ratio threshold (rough approximation)
        threshold_forward_ratio = np.tan(np.radians(threshold_rotation))

        if shoulder_forward_ratio > threshold_forward_ratio:
            analysis['issues'].append('shoulders_rounded')
            analysis['measurements']['shoulder_forward_distance'] = shoulder_forward_ratio * 100  # as percentage

        # DETECTION LOGIC - Calculate severity based on magnitude, not just count
        num_issues = len(analysis['issues'])

        if num_issues >= 1:
            # Calculate average severity multiplier across all issues
            severity_multipliers = []

            if 'spine_tilted' in analysis['issues']:
                severity_multipliers.append(analysis['measurements'].get('spine_tilt_severity', 1.0))

            if 'excessive_curve' in analysis['issues']:
                curve_degrees = analysis['measurements'].get('thoracic_curve_degrees', 0)
                # Normalize: 20° over threshold (180-160=20) = 1.0 multiplier baseline
                severity_multipliers.append(1.0 + (curve_degrees - 20) / 20)

            if 'shoulders_rounded' in analysis['issues']:
                forward_ratio = analysis['measurements'].get('shoulder_forward_ratio', 0)
                threshold_ratio = np.tan(np.radians(self.config['thresholds']['shoulder_rotation_angle']))
                severity_multipliers.append(forward_ratio / threshold_ratio if threshold_ratio > 0 else 1.0)

            # Calculate average severity
            if severity_multipliers:
                avg_severity = sum(severity_multipliers) / len(severity_multipliers)

                # Determine detection and severity based on magnitude
                if avg_severity > 2.0:
                    analysis['detected'] = True
                    analysis['severity'] = 'severe'
                    analysis['confidence'] = min(0.9 + (avg_severity - 2.0) * 0.1, 1.0)
                elif avg_severity > 1.5:
                    analysis['detected'] = True
                    analysis['severity'] = 'moderate'
                    analysis['confidence'] = 0.7 + (avg_severity - 1.5) * 0.4
                elif avg_severity > 1.2:  # Increased threshold to reduce false positives
                    analysis['detected'] = True
                    analysis['severity'] = 'mild'
                    analysis['confidence'] = 0.5 + (avg_severity - 1.2) * 0.5

                analysis['measurements']['average_severity_multiplier'] = avg_severity

        # Generate corrections
        if analysis['detected']:
            corrections = []
            if 'spine_tilted' in analysis['issues']:
                corrections.append(f"Straighten spine (currently {spine_angle:.1f}° from vertical)")
            if 'shoulders_rounded' in analysis['issues']:
                corrections.append("Pull shoulders back and down, open chest")
            if 'excessive_curve' in analysis['issues']:
                curve = analysis['measurements'].get('thoracic_curve_degrees', 0)
                corrections.append(f"Reduce upper back curve (currently {curve:.1f}°)")

            analysis['correction_needed'] = corrections

        return analysis['detected'], analysis

    def detect_low_wrists_3d(self, landmarks: Dict[str, np.ndarray],
                             proportions: Dict[str, float],
                             body_frame: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        ENHANCED LOW WRIST DETECTION (VIEW-INVARIANT)

        Uses body-space transformation for true view-invariance:
        1. Wrist drop ratio in body space (normalized by forearm length)
        2. Elbow-wrist-knuckle angle
        3. Wrist collapse angle

        Scale-invariant AND view-invariant!

        Args:
            landmarks: Dictionary of 3D landmark positions
            proportions: Body proportions for normalization
            body_frame: Pre-computed body frame (optional, computed if None)
        """
        analysis = {
            'detected': False,
            'measurements': {},
            'affected_hands': [],
            'severity': 'none',
            'confidence': 0.0
        }

        # Use provided body frame or build new one
        if body_frame is not None:
            R = body_frame['R']
            origin = body_frame['origin']
            use_body_space = True
        elif all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            R, origin, _ = self.geom.build_body_frame(
                landmarks['left_shoulder'], landmarks['right_shoulder'],
                landmarks['left_hip'], landmarks['right_hip']
            )
            use_body_space = True
        else:
            use_body_space = False
            R, origin = None, None

        body_vertical = np.array([0, 1, 0])  # Y-axis in body space

        # Check each hand separately
        for side in ['left', 'right']:
            wrist_key = f'{side}_wrist'
            elbow_key = f'{side}_elbow'
            shoulder_key = f'{side}_shoulder'

            if not all(k in landmarks for k in [wrist_key, elbow_key, shoulder_key]):
                continue

            wrist = landmarks[wrist_key]
            elbow = landmarks[elbow_key]
            shoulder = landmarks[shoulder_key]

            if use_body_space:
                # Transform to body space
                wrist_b = self.geom.to_body_space(wrist, R, origin)
                elbow_b = self.geom.to_body_space(elbow, R, origin)
                shoulder_b = self.geom.to_body_space(shoulder, R, origin)
            else:
                # Fallback to world coordinates
                wrist_b = wrist
                elbow_b = elbow
                shoulder_b = shoulder

            # Get forearm length for normalization
            forearm_length = proportions.get(f'{side}_forearm',
                                            np.linalg.norm(wrist_b - elbow_b))

            if forearm_length == 0:
                continue

            # METHOD 1: Vertical drop ratio in body space (scale-invariant!)
            # In body space, Y points UP, so wrist drop is elbow_Y - wrist_Y
            # Positive = wrist below elbow (dropped), Negative = wrist above elbow (raised)
            vertical_drop = elbow_b[1] - wrist_b[1]  # Y-axis difference in body space

            # NORMALIZE by forearm length (only flag positive drops, not negative)
            wrist_drop_ratio = max(0, vertical_drop) / forearm_length if forearm_length > 0 else 0

            analysis['measurements'][f'{side}_wrist_drop_ratio'] = wrist_drop_ratio
            analysis['measurements'][f'{side}_wrist_drop_percentage'] = wrist_drop_ratio * 100

            # METHOD 2: Elbow angle (joint angle analysis)
            upper_arm_b = elbow_b - shoulder_b
            forearm_b = wrist_b - elbow_b

            elbow_angle = self.geom.angle_between_vectors(upper_arm_b, forearm_b)
            analysis['measurements'][f'{side}_elbow_angle'] = elbow_angle

            # METHOD 3: Forearm angle from body-vertical
            forearm_angle_vertical = self.geom.angle_between_vectors(forearm_b, body_vertical)

            # Angle from horizontal is 90° - angle from vertical
            forearm_angle_horizontal = 90 - forearm_angle_vertical
            analysis['measurements'][f'{side}_forearm_angle_horizontal'] = forearm_angle_horizontal

            # Check for hand landmarks for more detail
            hand_wrist_key = f'hand_0_wrist' if side == 'left' else f'hand_1_wrist'
            hand_knuckle_key = f'hand_0_middle_mcp' if side == 'left' else f'hand_1_middle_mcp'

            if hand_wrist_key in landmarks and hand_knuckle_key in landmarks:
                hand_wrist = landmarks[hand_wrist_key]
                knuckle = landmarks[hand_knuckle_key]

                if use_body_space:
                    # Transform hand landmarks to body space
                    hand_wrist_b = self.geom.to_body_space(hand_wrist, R, origin)
                    knuckle_b = self.geom.to_body_space(knuckle, R, origin)
                else:
                    hand_wrist_b = hand_wrist
                    knuckle_b = knuckle

                # METHOD 4: Wrist-knuckle alignment (wrist collapse)
                hand_vector_b = knuckle_b - hand_wrist_b

                # Angle between forearm and hand in body space
                wrist_angle = self.geom.angle_between_vectors(forearm_b, hand_vector_b)
                analysis['measurements'][f'{side}_wrist_collapse_angle'] = wrist_angle

                # Check for wrist collapse (angle > 160° means collapsed)
                threshold_collapse = self.config['thresholds']['wrist_collapse_angle']
                if wrist_angle > threshold_collapse:
                    analysis['measurements'][f'{side}_wrist_collapsed'] = True

            # DETECTION LOGIC (using thresholds)
            threshold_drop = self.config['thresholds']['wrist_drop_ratio']
            threshold_elbow = self.config['thresholds']['elbow_wrist_angle_min']

            violations = []

            if wrist_drop_ratio > threshold_drop:
                violations.append('excessive_drop')

            # Elbow too bent (< 100°) often indicates wrist drop
            if elbow_angle < threshold_elbow:
                violations.append('elbow_bent')

            # Forearm angled too far down (> 30° below horizontal)
            if forearm_angle_horizontal < -30:
                violations.append('forearm_dropped')

            # If multiple violations, detect
            if len(violations) >= 2:
                analysis['affected_hands'].append(side)
                analysis['measurements'][f'{side}_violations'] = violations

        # Determine overall detection
        if len(analysis['affected_hands']) > 0:
            analysis['detected'] = True
            analysis['confidence'] = len(analysis['affected_hands']) / 2.0  # 0.5 or 1.0

            # Calculate severity
            max_drop_ratio = max([analysis['measurements'].get(f'{h}_wrist_drop_ratio', 0)
                                  for h in analysis['affected_hands']])

            threshold = self.config['thresholds']['wrist_drop_ratio']
            excess = max_drop_ratio / threshold if threshold > 0 else 1

            if excess > 2.0:
                analysis['severity'] = 'severe'
            elif excess > 1.5:
                analysis['severity'] = 'moderate'
            else:
                analysis['severity'] = 'mild'

            # Generate corrections (in relative terms)
            corrections = {}
            for hand in analysis['affected_hands']:
                drop_ratio = analysis['measurements'].get(f'{hand}_wrist_drop_ratio', 0)
                corrections[hand] = {
                    'action': f"Raise {hand} wrist",
                    'amount_relative': f"{drop_ratio * 100:.1f}% of forearm length",
                    'cue': f"Level {hand} forearm, support wrist from below"
                }

            analysis['correction_needed'] = corrections

        return analysis['detected'], analysis

    def detect_fingers_pointing_up_3d(self, landmarks: Dict[str, np.ndarray],
                                      proportions: Dict[str, float],
                                      body_frame: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        SIMPLIFIED PINKY-UP DETECTION FOR MUSICIANS (VIEW-INVARIANT)

        Detects the "pinky up" bad habit using 2 simple criteria:
        1. Pinky raised at angle from horizontal above threshold (e.g., > 30°)
        2. Other fingers are in working position (down/lower)

        Uses body-space transformation for true view-invariance.
        Scale-invariant AND view-invariant!

        Args:
            landmarks: Dictionary of 3D landmark positions
            proportions: Body proportions for normalization
            body_frame: Pre-computed body frame (optional, computed if None)
        """
        analysis = {
            'detected': False,
            'measurements': {},
            'affected_fingers': [],
            'severity': 'none',
            'confidence': 0.0
        }

        # Use provided body frame or build new one
        if body_frame is not None:
            R = body_frame['R']
            origin = body_frame['origin']
            use_body_space = True
        elif all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            R, origin, _ = self.geom.build_body_frame(
                landmarks['left_shoulder'], landmarks['right_shoulder'],
                landmarks['left_hip'], landmarks['right_hip']
            )
            use_body_space = True
        else:
            use_body_space = False
            R, origin = None, None

        body_vertical = np.array([0, 1, 0])  # Y-axis in body space

        # Detection threshold for pinky angle from horizontal
        pinky_angle_threshold = self.config['thresholds'].get('finger_vertical_angle', 30)

        # Check each hand
        for hand_idx in [0, 1]:
            hand_side = 'left' if hand_idx == 0 else 'right'

            # Get pinky landmarks
            pinky_tip_key = f'hand_{hand_idx}_pinky_tip'
            pinky_mcp_key = f'hand_{hand_idx}_pinky_mcp'

            # Need pinky landmarks to detect
            if not all(k in landmarks for k in [pinky_tip_key, pinky_mcp_key]):
                continue

            pinky_tip = landmarks[pinky_tip_key]
            pinky_mcp = landmarks[pinky_mcp_key]

            if use_body_space:
                pinky_tip_b = self.geom.to_body_space(pinky_tip, R, origin)
                pinky_mcp_b = self.geom.to_body_space(pinky_mcp, R, origin)
            else:
                pinky_tip_b = pinky_tip
                pinky_mcp_b = pinky_mcp

            # METHOD 1: Pinky angle from vertical (upward direction)
            pinky_vector_b = pinky_tip_b - pinky_mcp_b
            pinky_vertical_angle = self.geom.angle_between_vectors(pinky_vector_b, body_vertical)
            analysis['measurements'][f'{hand_side}_pinky_vertical_angle'] = pinky_vertical_angle

            # Check if pinky is raised (small angle from vertical = pointing upward)
            # When pinky points UP, angle from vertical is small (< 45°)
            # When pinky points DOWN, angle from vertical is large (> 135°)
            pinky_raised = pinky_vertical_angle < pinky_angle_threshold

            # METHOD 2: Check other fingers are in working position (down/lower)
            # Only check index, middle, ring (not thumb)
            other_fingers_down = 0
            other_fingers_total = 0

            for finger in ['index', 'middle', 'ring']:
                finger_tip_key = f'hand_{hand_idx}_{finger}_tip'
                finger_mcp_key = f'hand_{hand_idx}_{finger}_mcp'

                if finger_tip_key in landmarks and finger_mcp_key in landmarks:
                    other_fingers_total += 1
                    finger_tip = landmarks[finger_tip_key]
                    finger_mcp = landmarks[finger_mcp_key]

                    if use_body_space:
                        finger_tip_b = self.geom.to_body_space(finger_tip, R, origin)
                        finger_mcp_b = self.geom.to_body_space(finger_mcp, R, origin)
                    else:
                        finger_tip_b = finger_tip
                        finger_mcp_b = finger_mcp

                    # Check if finger is pointing down/horizontal (working position)
                    finger_vector_b = finger_tip_b - finger_mcp_b
                    finger_vert_angle = self.geom.angle_between_vectors(finger_vector_b, body_vertical)

                    # Finger is "down" if angle from vertical > 45° (not pointing up)
                    if finger_vert_angle > 45:
                        other_fingers_down += 1

            other_fingers_working = other_fingers_down >= 2  # At least 2 of 3 fingers working
            analysis['measurements'][f'{hand_side}_other_fingers_working'] = other_fingers_working
            analysis['measurements'][f'{hand_side}_other_fingers_down_count'] = other_fingers_down

            # SIMPLIFIED DETECTION LOGIC: Only 2 conditions
            # 1. Pinky raised above threshold angle
            # 2. Other fingers are in working position (down)

            if pinky_raised and other_fingers_working:
                analysis['affected_fingers'].append(f'{hand_side}_pinky')
                analysis['measurements'][f'{hand_side}_pinky_up_detected'] = True
                analysis['measurements'][f'{hand_side}_detection_reason'] = (
                    f"Pinky up: {pinky_vertical_angle:.1f}° from vertical, "
                    f"{other_fingers_down}/3 fingers working"
                )
            else:
                # Log why detection failed (for debugging)
                failed_checks = []
                if not pinky_raised:
                    failed_checks.append(f"vertical_angle={pinky_vertical_angle:.1f}° (threshold={pinky_angle_threshold}°)")
                if not other_fingers_working:
                    failed_checks.append(f"only {other_fingers_down}/3 fingers working")

                analysis['measurements'][f'{hand_side}_detection_failed'] = ", ".join(failed_checks)

        # Determine overall detection
        if len(analysis['affected_fingers']) > 0:
            analysis['detected'] = True

            # Confidence based on angle magnitude
            confidence_scores = []
            for hand_side in ['left', 'right']:
                if f'{hand_side}_pinky' in analysis['affected_fingers']:
                    h_angle = analysis['measurements'].get(f'{hand_side}_pinky_horizontal_angle', 0)

                    # Higher confidence for more extreme angles
                    # Base confidence: 0.6
                    # Bonus: stronger angle increases confidence
                    base_conf = 0.6
                    angle_excess = (h_angle - pinky_angle_threshold) / pinky_angle_threshold
                    angle_bonus = min(0.4, angle_excess * 0.2)

                    confidence_scores.append(base_conf + angle_bonus)

            analysis['confidence'] = max(confidence_scores) if confidence_scores else 0.5

            # Severity based on angle
            max_angle = max([analysis['measurements'].get(f'{hand}_pinky_horizontal_angle', 0)
                            for hand in ['left', 'right']
                            if f'{hand}_pinky' in analysis['affected_fingers']], default=0)

            # Severity scoring based on angle from horizontal
            if max_angle > 60:
                analysis['severity'] = 'severe'  # Very pronounced pinky up
            elif max_angle > 45:
                analysis['severity'] = 'moderate'  # Clear pinky up
            else:
                analysis['severity'] = 'mild'  # Slight pinky up

            # Generate corrections
            affected_hands = set([f.split('_')[0] for f in analysis['affected_fingers']])
            corrections = []

            for hand in affected_hands:
                h_angle = analysis['measurements'].get(f'{hand}_pinky_horizontal_angle', 0)

                corrections.append({
                    'hand': hand,
                    'issue': 'Pinky raised (pinky sticking up)',
                    'action': f"Lower {hand} pinky naturally alongside other fingers",
                    'cue': "Keep pinky curved and close to ring finger in working position",
                    'measurements': f"Currently: {h_angle:.1f}° from horizontal (threshold: {pinky_angle_threshold}°)"
                })

            analysis['correction_needed'] = corrections

        return analysis['detected'], analysis

    def calculate_posture_score(self, all_detections: Dict) -> Dict:
        """
        Calculate overall posture score based on all detections
        """
        score = 100
        deductions = {
            'turtle_neck': {
                'mild': 10,
                'moderate': 20,
                'severe': 30
            },
            'hunched_back': {
                'mild': 15,
                'moderate': 25,
                'severe': 35
            },
            'low_wrists': {
                'mild': 8,
                'moderate': 15,
                'severe': 25
            },
            'fingers_up': {
                'mild': 5,
                'moderate': 10,
                'severe': 15
            }
        }

        detailed_deductions = []

        for gesture_type, detection_result in all_detections.items():
            if detection_result.get('detected', False):
                severity = detection_result.get('severity', 'mild')
                deduction = deductions.get(gesture_type, {}).get(severity, 0)
                score -= deduction

                detailed_deductions.append({
                    'issue': gesture_type,
                    'severity': severity,
                    'points_lost': deduction,
                    'confidence': detection_result.get('confidence', 0.0)
                })

        # Ensure score doesn't go below 0
        score = max(0, score)

        # Determine grade
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


class BadGestureDetector3D:
    """
    Enhanced 3D bad gesture detection using MediaPipe's x, y, z coordinates
    Drop-in replacement for BadGestureDetector with improved accuracy

    This is a wrapper class that provides backward compatibility with the original
    BadGestureDetector interface while using the enhanced 3D detection internally.
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

        # Initialize enhanced 3D detector (no longer imports from external file!)
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
