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
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time


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
        2. Upper spine to lower spine angle
        3. Shoulder rotation in body space
        4. Thoracic curve angle

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

        # ANALYSIS 4: C-curve detection using all points in body space
        if len(analysis['issues']) > 0:
            # Calculate overall posture vector quality in body space
            # Check if multiple segments are misaligned

            if 'left_ear' in landmarks and 'right_ear' in landmarks:
                neck = (landmarks['left_ear'] + landmarks['right_ear']) / 2
                neck_b = self.geom.to_body_space(neck, R, origin)

                # Create chain in body space: hip -> shoulder -> neck
                segment1_b = shoulder_ctr_b - hip_ctr_b
                segment2_b = neck_b - shoulder_ctr_b

                # Both should point upward (Y-axis) with minimal forward (Z) component
                segment1_vertical = self.geom.angle_between_vectors(segment1_b, body_vertical)
                segment2_vertical = self.geom.angle_between_vectors(segment2_b, body_vertical)

                avg_deviation = (segment1_vertical + segment2_vertical) / 2
                analysis['measurements']['average_spine_deviation'] = avg_deviation

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
        ENHANCED FINGERS POINTING UP DETECTION (VIEW-INVARIANT)

        Uses body-space transformation for true view-invariance:
        1. Finger angle from body-vertical
        2. Finger hyperextension angle
        3. Finger curl angle

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

        # Check each hand
        for hand_idx in [0, 1]:
            hand_side = 'left' if hand_idx == 0 else 'right'

            # Check each finger and track their states
            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
            finger_states = {}  # Track which fingers are pointing up

            for finger in fingers:
                tip_key = f'hand_{hand_idx}_{finger}_tip'
                dip_key = f'hand_{hand_idx}_{finger}_dip'
                mcp_key = f'hand_{hand_idx}_{finger}_mcp'

                # Need at least tip and mcp
                if tip_key not in landmarks or mcp_key not in landmarks:
                    continue

                tip = landmarks[tip_key]
                mcp = landmarks[mcp_key]  # Base of finger

                if use_body_space:
                    # Transform to body space
                    tip_b = self.geom.to_body_space(tip, R, origin)
                    mcp_b = self.geom.to_body_space(mcp, R, origin)
                else:
                    tip_b = tip
                    mcp_b = mcp

                # METHOD 1: Finger direction angle from body-vertical
                finger_vector_b = tip_b - mcp_b

                # Angle from body-vertical
                finger_vertical_angle = self.geom.angle_between_vectors(finger_vector_b, body_vertical)

                # Angle from horizontal is complementary
                finger_horizontal_angle = 90 - finger_vertical_angle

                analysis['measurements'][f'{hand_side}_{finger}_vertical_angle'] = finger_vertical_angle
                analysis['measurements'][f'{hand_side}_{finger}_horizontal_angle'] = finger_horizontal_angle

                # METHOD 2: Finger extension angle (if DIP available)
                extension_angle = None
                if dip_key in landmarks:
                    dip = landmarks[dip_key]

                    if use_body_space:
                        dip_b = self.geom.to_body_space(dip, R, origin)
                    else:
                        dip_b = dip

                    # Proximal phalanx (mcp to dip)
                    proximal_b = dip_b - mcp_b
                    # Distal phalanx (dip to tip)
                    distal_b = tip_b - dip_b

                    # Extension angle
                    extension_angle = self.geom.angle_between_vectors(proximal_b, distal_b)
                    analysis['measurements'][f'{hand_side}_{finger}_extension_angle'] = extension_angle

                    # Check for hyperextension (> 200°)
                    threshold_hyperext = self.config['thresholds']['finger_hyperextension_angle']
                    if extension_angle > threshold_hyperext:
                        analysis['measurements'][f'{hand_side}_{finger}_hyperextended'] = True

                # DETECTION LOGIC: Track finger states
                threshold_vertical = self.config['thresholds']['finger_vertical_angle']

                # Determine if finger is pointing up
                is_pointing_up = False
                if finger_vertical_angle < threshold_vertical:
                    is_pointing_up = True

                # Additional check: if finger is too straight (> 150° curl)
                if extension_angle and extension_angle > self.config['thresholds']['finger_curl_min_angle'] and finger_vertical_angle < 75:
                    is_pointing_up = True

                finger_states[finger] = is_pointing_up

            # NEW LOGIC: Only detect if pinky is up while other fingers (except thumb) are down
            # This detects the classic musician "pinky up" bad habit
            if finger_states.get('pinky', False):
                # Check if this is just pinky pointing up (not a raised hand)
                other_fingers = ['index', 'middle', 'ring']
                other_fingers_up = sum([finger_states.get(f, False) for f in other_fingers])

                # Store finger states for debugging
                analysis['measurements'][f'{hand_side}_finger_states'] = finger_states.copy()
                analysis['measurements'][f'{hand_side}_other_fingers_up_count'] = other_fingers_up

                # Only flag as bad gesture if pinky is up but most other fingers are down
                # Allow 1 other finger to be slightly up (tolerance for measurement noise)
                # But if 2+ other fingers are up, it's likely the whole hand is raised (not a bad gesture)
                if other_fingers_up <= 1:
                    analysis['affected_fingers'].append(f'{hand_side}_pinky')
                    analysis['measurements'][f'{hand_side}_pinky_isolated'] = True
                    analysis['measurements'][f'{hand_side}_pinky_detected_reason'] = f"Pinky up, {other_fingers_up}/3 others up"

        # Determine overall detection
        if len(analysis['affected_fingers']) > 0:
            analysis['detected'] = True

            # Confidence based on number of fingers
            num_fingers = len(analysis['affected_fingers'])
            analysis['confidence'] = min(num_fingers / 5.0, 1.0)

            # Severity
            if num_fingers > 5:
                analysis['severity'] = 'severe'
            elif num_fingers > 2:
                analysis['severity'] = 'moderate'
            else:
                analysis['severity'] = 'mild'

            # Generate corrections
            affected_hands = set([f.split('_')[0] for f in analysis['affected_fingers']])
            corrections = []

            for hand in affected_hands:
                hand_fingers = [f for f in analysis['affected_fingers'] if f.startswith(hand)]
                corrections.append({
                    'hand': hand,
                    'fingers': [f.split('_')[1] for f in hand_fingers],
                    'action': f"Curve {hand} fingers naturally downward",
                    'cue': "Relax fingers into natural curve, don't point upward"
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


class ComparisonDemo:
    """
    Demonstrate the improvements: Old 2D vs Old 3D vs New Enhanced 3D
    """

    @staticmethod
    def show_improvements():
        """
        Show concrete examples of improvements
        """
        print("\n" + "="*80)
        print("EVOLUTION OF BAD GESTURE DETECTION")
        print("="*80)

        comparisons = {
            "TURTLE NECK DETECTION": {
                "2D_Approach": {
                    "method": "if nose_pixel_y < shoulder_pixel_y: detected = True",
                    "problems": [
                        "Camera angle changes detection",
                        "Distance from camera affects result",
                        "Side view vs front view inconsistent"
                    ]
                },
                "Old_3D_Approach": {
                    "method": "if head_forward_distance > 0.08m: detected = True",
                    "problems": [
                        "Requires metric calibration",
                        "Doesn't scale with body size",
                        "8cm for child vs adult wrong"
                    ]
                },
                "Enhanced_3D_Approach": {
                    "method": "ratio = head_forward / shoulder_width\nif ratio > 0.20: detected = True",
                    "advantages": [
                        "✅ Scale-invariant (works for any body size)",
                        "✅ View-invariant (any camera angle)",
                        "✅ No calibration needed",
                        "✅ Self-normalizing"
                    ]
                }
            },

            "LOW WRIST DETECTION": {
                "2D_Approach": {
                    "method": "if wrist_pixel_y > knuckle_pixel_y + 20: detected = True",
                    "problems": [
                        "Pixel threshold meaningless",
                        "Camera height changes everything",
                        "Resolution dependent"
                    ]
                },
                "Old_3D_Approach": {
                    "method": "if wrist_drop > 0.03m: detected = True",
                    "problems": [
                        "3cm drop means different for short vs long arms",
                        "Doesn't account for arm proportions",
                        "Fixed threshold breaks"
                    ]
                },
                "Enhanced_3D_Approach": {
                    "method": "ratio = wrist_drop / forearm_length\nif ratio > 0.08: detected = True",
                    "advantages": [
                        "✅ Adapts to arm length",
                        "✅ 8% works for everyone",
                        "✅ Proportional measurement",
                        "✅ Combined with joint angles"
                    ]
                }
            },

            "HUNCHED BACK DETECTION": {
                "2D_Approach": {
                    "method": "angle_2d = atan2(shoulder_y - hip_y, shoulder_x - hip_x)",
                    "problems": [
                        "Needs perfect side view",
                        "Rotation ruins measurement",
                        "Can't separate lean from hunch"
                    ]
                },
                "Old_3D_Approach": {
                    "method": "if spine_distance_from_vertical > 0.05m: detected = True",
                    "problems": [
                        "Tall person always detected",
                        "Doesn't measure actual curve",
                        "Absolute distance wrong metric"
                    ]
                },
                "Enhanced_3D_Approach": {
                    "method": "spine_angle = angle_between(spine_vector, vertical)\nif spine_angle > 15°: detected = True",
                    "advantages": [
                        "✅ Angles are scale-invariant",
                        "✅ Measures actual curve amount",
                        "✅ Works from any view",
                        "✅ Multiple angle checks"
                    ]
                }
            }
        }

        for detection_type, approaches in comparisons.items():
            print(f"\n{'='*80}")
            print(f"{detection_type}")
            print('='*80)

            for approach_name, details in approaches.items():
                print(f"\n{approach_name.replace('_', ' ')}:")
                print(f"  Code: {details['method']}")

                if 'problems' in details:
                    print("  ❌ Problems:")
                    for problem in details['problems']:
                        print(f"     • {problem}")

                if 'advantages' in details:
                    print("  ✅ Advantages:")
                    for advantage in details['advantages']:
                        print(f"     • {advantage}")

        print("\n" + "="*80)
        print("KEY PRINCIPLES OF ENHANCED 3D DETECTION")
        print("="*80)

        principles = {
            "1. SCALE INVARIANCE": [
                "All distances normalized by body segment lengths",
                "Ratios instead of absolute measurements",
                "Self-calibrating to individual proportions",
                "Works for children, adults, all sizes"
            ],
            "2. VIEW INVARIANCE": [
                "Uses true 3D geometry",
                "Angles measured in 3D space",
                "Not dependent on camera position",
                "Consistent from any perspective"
            ],
            "3. ROBUST METRICS": [
                "Angles (naturally invariant)",
                "Normalized ratios (scale-invariant)",
                "Vector projections (directional)",
                "Multiple confirming measurements"
            ],
            "4. NO CALIBRATION NEEDED": [
                "No absolute distance thresholds",
                "No metric scale required",
                "No camera calibration needed",
                "Works with any 3D pose system"
            ]
        }

        for principle, points in principles.items():
            print(f"\n{principle}")
            for point in points:
                print(f"  ✓ {point}")

        return comparisons


def complete_usage_example():
    """
    Complete example showing enhanced 3D detection pipeline
    """
    print("\n" + "="*80)
    print("ENHANCED 3D BAD GESTURE DETECTION - COMPLETE EXAMPLE")
    print("="*80)

    # Simulated 3D landmarks (realistic musician pose)
    landmarks_3d = {
        # Pose landmarks (in arbitrary 3D units - scale doesn't matter!)
        'nose': np.array([0.15, 0.12, 1.85]),
        'left_ear': np.array([0.10, 0.10, 1.87]),
        'right_ear': np.array([0.20, 0.10, 1.87]),
        'left_shoulder': np.array([0.05, 0.00, 1.70]),
        'right_shoulder': np.array([0.25, 0.00, 1.70]),
        'left_elbow': np.array([0.00, 0.15, 1.55]),
        'right_elbow': np.array([0.30, 0.15, 1.55]),
        'left_wrist': np.array([-0.05, 0.30, 1.35]),
        'right_wrist': np.array([0.35, 0.30, 1.40]),
        'left_hip': np.array([0.08, 0.00, 1.30]),
        'right_hip': np.array([0.22, 0.00, 1.30]),

        # Hand landmarks
        'hand_0_wrist': np.array([-0.05, 0.30, 1.35]),
        'hand_0_middle_mcp': np.array([-0.08, 0.35, 1.38]),
        'hand_0_middle_tip': np.array([-0.10, 0.38, 1.42]),
        'hand_0_middle_dip': np.array([-0.09, 0.37, 1.40]),
        'hand_0_pinky_tip': np.array([-0.12, 0.38, 1.45]),
        'hand_0_pinky_mcp': np.array([-0.10, 0.36, 1.37]),
        'hand_0_pinky_dip': np.array([-0.11, 0.37, 1.41]),
    }

    # Initialize enhanced detector
    detector = EnhancedBadGestureDetector3D()

    print("\nCalculating body proportions...")
    proportions = detector.calculate_body_proportions(landmarks_3d)

    print("\nBody Proportions (scale-invariant reference):")
    for key, value in proportions.items():
        print(f"  {key}: {value:.3f} units")

    # Run detection
    print("\n" + "="*80)
    print("Running Enhanced 3D Detection...")
    print("="*80)

    results = detector.detect_all_3d(landmarks_3d)

    # Display results
    for gesture_type in ['turtle_neck', 'hunched_back', 'low_wrists', 'fingers_up']:
        analysis = results['detailed_analysis'][gesture_type]

        print(f"\n{'-'*80}")
        print(f"{gesture_type.upper().replace('_', ' ')}")
        print(f"{'-'*80}")
        print(f"Detected: {'❌ YES' if analysis['detected'] else '✅ NO'}")

        if analysis['detected']:
            print(f"Severity: {analysis['severity'].upper()}")
            print(f"Confidence: {analysis['confidence']:.1%}")

            print("\nMeasurements (scale-invariant):")
            for key, value in analysis['measurements'].items():
                if isinstance(value, (int, float)):
                    if 'ratio' in key:
                        print(f"  • {key}: {value:.3f} ({value*100:.1f}%)")
                    elif 'angle' in key or 'degrees' in key:
                        print(f"  • {key}: {value:.1f}°")
                    elif 'percentage' in key:
                        print(f"  • {key}: {value:.1f}%")
                    else:
                        print(f"  • {key}: {value:.3f}")

            print("\nCorrections needed:")
            corrections = analysis.get('correction_needed', {})
            if isinstance(corrections, dict):
                for key, value in corrections.items():
                    if isinstance(value, dict):
                        print(f"  • {key}:")
                        for k, v in value.items():
                            print(f"      {k}: {v}")
                    else:
                        print(f"  • {key}: {value}")
            elif isinstance(corrections, list):
                for correction in corrections:
                    if isinstance(correction, dict):
                        for k, v in correction.items():
                            print(f"  • {k}: {v}")
                    else:
                        print(f"  • {correction}")

    # Calculate overall score
    print("\n" + "="*80)
    print("OVERALL POSTURE ASSESSMENT")
    print("="*80)

    all_detections = {k: results['detailed_analysis'][k]
                     for k in ['turtle_neck', 'hunched_back', 'low_wrists', 'fingers_up']}

    score_result = detector.calculate_posture_score(all_detections)

    print(f"\nPosture Score: {score_result['score']}/100")
    print(f"Grade: {score_result['grade']}")
    print(f"Feedback: {score_result['feedback']}")

    if score_result['deductions']:
        print("\nPoint Deductions:")
        for deduction in score_result['deductions']:
            print(f"  • {deduction['issue']} ({deduction['severity']}): "
                  f"-{deduction['points_lost']} points "
                  f"[confidence: {deduction['confidence']:.1%}]")

    # Show comparison
    ComparisonDemo.show_improvements()

    print("\n" + "="*80)
    print("✅ ENHANCED 3D DETECTION COMPLETE!")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Scale-invariant (works for any body size)")
    print("  ✓ View-invariant (works from any camera angle)")
    print("  ✓ No calibration needed (self-normalizing)")
    print("  ✓ Robust metrics (angles + normalized ratios)")
    print("  ✓ Multiple confirming measurements")
    print("  ✓ Confidence scores for reliability")
    print("="*80)


if __name__ == "__main__":
    # Run complete example
    complete_usage_example()
