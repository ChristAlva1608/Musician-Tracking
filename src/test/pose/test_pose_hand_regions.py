#!/usr/bin/env python3
"""
Test script to visualize pose landmarks and hand regions from pose detection
Shows hand bounding boxes derived from wrist landmarks
"""

import cv2
import numpy as np
import sys
import os
import time

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.pose.mediapipe import MediaPipePoseDetector
from src.models.pose.yolo import YOLOPoseDetector
from src.models.hand.mediapipe import MediaPipeHandDetector


def draw_hand_regions_from_pose(frame, pose_landmarks, pose_model='yolo'):
    """
    Draw hand regions based on pose wrist landmarks

    Args:
        frame: Input frame
        pose_landmarks: Pose landmarks from detector
        pose_model: 'yolo' or 'mediapipe' to determine keypoint indices

    Returns:
        Frame with drawn hand regions and list of hand bounding boxes
    """
    if not pose_landmarks or len(pose_landmarks) == 0:
        return frame, []

    height, width = frame.shape[:2]
    hand_bboxes = []
    annotated_frame = frame.copy()

    # Define wrist indices based on pose model
    if pose_model.lower() == 'yolo':
        # YOLO pose keypoints
        wrist_indices = {'left_wrist': 9, 'right_wrist': 10}
        keypoint_names = {
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear',
            4: 'right_ear',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_wrist',
            10: 'right_wrist',
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle'
        }
    else:  # MediaPipe
        wrist_indices = {'left_wrist': 15, 'right_wrist': 16}
        keypoint_names = {
            0: 'nose',
            1: 'left_eye_inner',
            2: 'left_eye',
            3: 'left_eye_outer',
            4: 'right_eye_inner',
            5: 'right_eye',
            6: 'right_eye_outer',
            7: 'left_ear',
            8: 'right_ear',
            9: 'mouth_left',
            10: 'mouth_right',
            11: 'left_shoulder',
            12: 'right_shoulder',
            13: 'left_elbow',
            14: 'right_elbow',
            15: 'left_wrist',
            16: 'right_wrist',
            17: 'left_pinky',
            18: 'right_pinky',
            19: 'left_index',
            20: 'right_index',
            21: 'left_thumb',
            22: 'right_thumb',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle',
            29: 'left_heel',
            30: 'right_heel',
            31: 'left_foot_index',
            32: 'right_foot_index'
        }

    # Handle different return formats from pose detectors
    # YOLO returns a single list of landmarks, MediaPipe might return differently
    if not pose_landmarks:
        return annotated_frame, hand_bboxes

    # Normalize to list of person landmarks
    if isinstance(pose_landmarks, list) and len(pose_landmarks) > 0:
        # Check if it's a list of landmark dicts (single person) or list of lists (multiple people)
        if isinstance(pose_landmarks[0], dict):
            # Single person as list of landmarks
            people_landmarks = [pose_landmarks]
        else:
            # Multiple people
            people_landmarks = pose_landmarks
    else:
        people_landmarks = [pose_landmarks]

    # Process each person's landmarks
    for person_landmarks in people_landmarks:
        if not person_landmarks:
            continue

        # Convert list to dict format with indices as keys
        if isinstance(person_landmarks, list):
            landmarks = {i: lm for i, lm in enumerate(person_landmarks)}
        else:
            landmarks = person_landmarks

        # Draw all pose keypoints with names
        for idx, landmark in landmarks.items():
            # Handle different landmark formats
            if isinstance(landmark, dict):
                x_val = landmark.get('x', 0)
                y_val = landmark.get('y', 0)
            else:
                # Skip if not a dict
                continue

            if x_val > 0 and y_val > 0:
                # Convert normalized coordinates to pixel coordinates
                px = int(x_val * width) if x_val <= 1 else int(x_val)
                py = int(y_val * height) if y_val <= 1 else int(y_val)

                # Determine color based on keypoint type
                if idx in wrist_indices.values():
                    color = (0, 255, 0)  # Green for wrists
                    radius = 8
                elif idx in [7, 8, 13, 14]:  # Elbows and knees
                    color = (255, 255, 0)  # Yellow
                    radius = 6
                elif idx in [5, 6, 11, 12, 23, 24]:  # Shoulders and hips
                    color = (255, 0, 255)  # Magenta
                    radius = 6
                else:
                    color = (255, 255, 255)  # White for others
                    radius = 4

                # Draw keypoint
                cv2.circle(annotated_frame, (px, py), radius, color, -1)

                # Add keypoint name
                if idx in keypoint_names:
                    name = keypoint_names[idx]
                    # Draw text background for better visibility
                    (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                    cv2.rectangle(annotated_frame, (px + 10, py - text_height - 2),
                                (px + 10 + text_width, py + 2), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, name, (px + 10, py),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Create bounding boxes around wrist areas for hand detection
        # Define elbow indices based on pose model
        if pose_model.lower() == 'yolo':
            elbow_indices = {'left_elbow': 7, 'right_elbow': 8}
        else:  # MediaPipe
            elbow_indices = {'left_elbow': 13, 'right_elbow': 14}

        for hand_name, wrist_idx in wrist_indices.items():
            if wrist_idx in landmarks:
                wrist = landmarks[wrist_idx]
                # Handle different landmark formats
                if isinstance(wrist, dict):
                    wrist_x = wrist.get('x', 0)
                    wrist_y = wrist.get('y', 0)
                else:
                    continue

                if wrist_x > 0 and wrist_y > 0:
                    # Get corresponding elbow position
                    elbow_key = 'left_elbow' if 'left' in hand_name else 'right_elbow'
                    elbow_idx = elbow_indices[elbow_key]

                    # Default box size (fallback if elbow not detected)
                    box_size = int(min(width, height) * 0.15)

                    # Calculate box size based on wrist-to-elbow distance if elbow is detected
                    if elbow_idx in landmarks:
                        elbow = landmarks[elbow_idx]
                        if isinstance(elbow, dict):
                            elbow_x = elbow.get('x', 0)
                            elbow_y = elbow.get('y', 0)

                            if elbow_x > 0 and elbow_y > 0:
                                # Convert to pixel coordinates
                                elbow_px = int(elbow_x * width) if elbow_x <= 1 else int(elbow_x)
                                elbow_py = int(elbow_y * height) if elbow_y <= 1 else int(elbow_y)
                                wrist_px = int(wrist_x * width) if wrist_x <= 1 else int(wrist_x)
                                wrist_py = int(wrist_y * height) if wrist_y <= 1 else int(wrist_y)

                                # Calculate distance from wrist to elbow
                                distance = np.sqrt((elbow_px - wrist_px) ** 2 + (elbow_py - wrist_py) ** 2)

                                # Box size is 2/3 of wrist-to-elbow distance
                                box_size = int(distance * 2 / 3)

                    # Calculate bbox coordinates with wrist at middle of bottom edge
                    cx = int(wrist_x * width) if wrist_x <= 1 else int(wrist_x)
                    cy = int(wrist_y * height) if wrist_y <= 1 else int(wrist_y)

                    # For a square box with wrist at middle of bottom edge:
                    # x1, x2 are centered on wrist
                    # y1 is above wrist by box_size, y2 is at wrist level
                    x1 = max(0, cx - box_size // 2)
                    y1 = max(0, cy - box_size)
                    x2 = min(width, cx + box_size // 2)
                    y2 = min(height, cy)

                    # Draw hand region bounding box
                    color = (0, 255, 0) if 'left' in hand_name else (255, 0, 0)  # Green for left, Blue for right
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Add label for hand region
                    label = f"{hand_name.replace('_', ' ').title()} Region"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Store bbox info
                    hand_bboxes.append({
                        'x': x1,
                        'y': y1,
                        'w': x2 - x1,
                        'h': y2 - y1,
                        'side': 'left' if 'left' in hand_name else 'right',
                        'wrist_x': cx,
                        'wrist_y': cy
                    })

    return annotated_frame, hand_bboxes


def test_pose_hand_regions_webcam(use_mediapipe=False):
    """
    Test pose detection with hand region visualization using webcam

    Args:
        use_mediapipe: If True, use MediaPipe pose detector, else use YOLO
    """
    print("=" * 50)
    print(f"Testing {'MediaPipe' if use_mediapipe else 'YOLO'} Pose Detection with Hand Regions")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Press 's' to switch between YOLO and MediaPipe")
    print("Press 'h' to toggle hand detection in regions")
    print("=" * 50)

    # Initialize pose detector
    if use_mediapipe:
        pose_detector = MediaPipePoseDetector(min_detection_confidence=0.5)
        model_name = "MediaPipe"
    else:
        pose_detector = YOLOPoseDetector(confidence=0.5)
        model_name = "YOLO"

    # Initialize hand detector for region-based detection
    hand_detector = MediaPipeHandDetector(min_detection_confidence=0.7)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detect_hands_in_regions = True
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        frame_count += 1

        # Detect pose
        pose_results = pose_detector.detect(frame)
        pose_landmarks = pose_detector.convert_to_dict(pose_results)

        # Draw pose landmarks with model's method
        annotated_frame = pose_detector.draw_landmarks(frame, pose_results)

        # Draw hand regions from pose landmarks
        model_type = 'mediapipe' if use_mediapipe else 'yolo'
        annotated_frame, hand_bboxes = draw_hand_regions_from_pose(
            annotated_frame, pose_landmarks, model_type
        )

        # Optionally detect actual hands in the regions
        if detect_hands_in_regions and hand_bboxes:
            for bbox in hand_bboxes:
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

                # Crop hand region
                hand_crop = frame[y:y+h, x:x+w]

                if hand_crop.size > 0:
                    # Detect hand landmarks in cropped region
                    hand_results = hand_detector.detect(hand_crop)

                    if hand_results and hasattr(hand_results, 'multi_hand_landmarks'):
                        # Draw hand landmarks on the cropped region
                        hand_crop_annotated = hand_detector.draw_landmarks(hand_crop, hand_results)
                        # Copy back to main frame
                        annotated_frame[y:y+h, x:x+w] = hand_crop_annotated

                        # Add status text
                        cv2.putText(annotated_frame, "Hand Detected!",
                                  (bbox['x'], bbox['y'] + bbox['h'] + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate and display FPS
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")

        # Add info text
        info_text = [
            f"Model: {model_name} Pose",
            f"Detected Persons: {len(pose_landmarks) if pose_landmarks else 0}",
            f"Hand Regions: {len(hand_bboxes)}",
            f"Hand Detection: {'ON' if detect_hands_in_regions else 'OFF'}",
            "Press 'q' to quit, 's' to switch model, 'h' to toggle hand detection"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(annotated_frame, text, (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        # Display frame
        cv2.imshow('Pose with Hand Regions', annotated_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Switch between YOLO and MediaPipe
            pose_detector.cleanup()
            use_mediapipe = not use_mediapipe
            if use_mediapipe:
                pose_detector = MediaPipePoseDetector(min_detection_confidence=0.5)
                model_name = "MediaPipe"
            else:
                pose_detector = YOLOPoseDetector(confidence=0.5)
                model_name = "YOLO"
            print(f"Switched to {model_name} pose detector")
        elif key == ord('h'):
            # Toggle hand detection
            detect_hands_in_regions = not detect_hands_in_regions
            print(f"Hand detection in regions: {'ON' if detect_hands_in_regions else 'OFF'}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.cleanup()
    hand_detector.cleanup()

    print("\nTest completed!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test pose landmarks with hand region detection')
    parser.add_argument('--mediapipe', action='store_true',
                       help='Use MediaPipe pose detector (default is YOLO)')

    args = parser.parse_args()

    # Run webcam test
    test_pose_hand_regions_webcam(use_mediapipe=args.mediapipe)