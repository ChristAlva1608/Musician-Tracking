#!/usr/bin/env python3
"""
Test script for two-stage MediaPipe hand detection
Stage 1: Detect hand and create precise bounding box
Stage 2: Crop and re-detect for higher accuracy
"""

import cv2
import numpy as np
import sys
import os
import time
import mediapipe as mp

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.hand.mediapipe import MediaPipeHandDetector


def get_hand_bbox_from_landmarks(landmarks, image_shape):
    """
    Calculate precise bounding box from hand landmarks
    Box covers all 21 hand landmarks (leftmost, rightmost, highest, lowest)

    Args:
        landmarks: MediaPipe hand landmarks (21 points)
        image_shape: (height, width, channels) of the image

    Returns:
        Dictionary with bbox coordinates and padding info
    """
    height, width = image_shape[:2]

    # MediaPipe hand landmark indices for key points
    WRIST = 0
    MIDDLE_FINGER_TIP = 12

    # Extract x, y coordinates for all 21 landmarks
    x_coords = []
    y_coords = []

    for landmark in landmarks.landmark:
        x_coords.append(landmark.x * width)
        y_coords.append(landmark.y * height)

    # Find actual boundaries from all 21 landmarks
    # This ensures the box covers the entire hand regardless of orientation
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Add some padding (10% of bbox size)
    width_padding = (x_max - x_min) * 0.1
    height_padding = (y_max - y_min) * 0.1

    # Calculate final bbox with padding
    x1 = max(0, int(x_min - width_padding))
    y1 = max(0, int(y_min - height_padding))
    x2 = min(width, int(x_max + width_padding))
    y2 = min(height, int(y_max + height_padding))

    return {
        'x': x1,
        'y': y1,
        'w': x2 - x1,
        'h': y2 - y1,
        'x_min_raw': x_min,
        'x_max_raw': x_max,
        'y_min_raw': y_min,
        'y_max_raw': y_max,
        'wrist_x': x_coords[WRIST],
        'wrist_y': y_coords[WRIST],
        'middle_tip_x': x_coords[MIDDLE_FINGER_TIP],
        'middle_tip_y': y_coords[MIDDLE_FINGER_TIP]
    }


def draw_hand_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw bounding box on image"""
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    # Draw key points
    wrist_x, wrist_y = int(bbox['wrist_x']), int(bbox['wrist_y'])
    middle_tip_x, middle_tip_y = int(bbox['middle_tip_x']), int(bbox['middle_tip_y'])

    # Draw wrist point
    cv2.circle(image, (wrist_x, wrist_y), 5, (255, 0, 0), -1)  # Blue for wrist
    cv2.putText(image, "Wrist", (wrist_x + 10, wrist_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw middle finger tip
    cv2.circle(image, (middle_tip_x, middle_tip_y), 5, (0, 0, 255), -1)  # Red for fingertip
    cv2.putText(image, "Middle Tip", (middle_tip_x + 10, middle_tip_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image


def test_two_stage_hand_detection_webcam():
    """
    Test two-stage hand detection with webcam
    """
    print("=" * 60)
    print("Two-Stage MediaPipe Hand Detection Test")
    print("=" * 60)
    print("Stage 1: Initial detection with full frame")
    print("Stage 2: Refined detection with cropped hand region")
    print("-" * 60)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'p' to pause/resume")
    print("Press '1' to show only stage 1")
    print("Press '2' to show only stage 2")
    print("Press 'b' to show both stages")
    print("=" * 60)

    # Initialize hand detectors
    # Stage 1: Lower confidence for initial detection
    detector_stage1 = MediaPipeHandDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Stage 2: Higher confidence for refined detection
    detector_stage2 = MediaPipeHandDetector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    paused = False
    display_mode = 'both'  # 'stage1', 'stage2', or 'both'
    frame_count = 0
    save_count = 0

    # Performance tracking
    stage1_times = []
    stage2_times = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            frame_count += 1

            # Create copies for visualization
            stage1_frame = frame.copy()
            stage2_frame = frame.copy()

            # Stage 1: Initial hand detection
            start_time = time.time()
            stage1_results = detector_stage1.detect(frame)
            stage1_time = (time.time() - start_time) * 1000
            stage1_times.append(stage1_time)

            hand_bboxes = []
            stage2_results_list = []

            if stage1_results and stage1_results.multi_hand_landmarks:
                # Draw all hand landmarks on stage1 frame
                stage1_frame = detector_stage1.draw_landmarks(stage1_frame, stage1_results)
                # Process each detected hand
                for hand_idx, hand_landmarks in enumerate(stage1_results.multi_hand_landmarks):
                    # Get precise bounding box
                    bbox = get_hand_bbox_from_landmarks(hand_landmarks, frame.shape)
                    hand_bboxes.append(bbox)

                    # Draw bbox on stage 1 frame
                    stage1_frame = draw_hand_bbox(stage1_frame, bbox, color=(0, 255, 0))

                    # Stage 2: Crop and re-detect
                    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

                    # Ensure valid crop region
                    if w > 20 and h > 20:  # Minimum size for meaningful detection
                        hand_crop = frame[y:y+h, x:x+w].copy()

                        # Detect on cropped region
                        start_time = time.time()
                        stage2_results = detector_stage2.detect(hand_crop)
                        stage2_time = (time.time() - start_time) * 1000
                        stage2_times.append(stage2_time)

                        if stage2_results and stage2_results.multi_hand_landmarks:
                            stage2_results_list.append({
                                'results': stage2_results,
                                'bbox': bbox,
                                'crop': hand_crop
                            })

                            # Draw refined detection on cropped region
                            hand_crop_annotated = detector_stage2.draw_landmarks(
                                hand_crop,
                                stage2_results
                            )

                            # Get refined bbox from stage 2
                            # refined_bbox = get_hand_bbox_from_landmarks(
                            #     stage2_results.multi_hand_landmarks[0],
                            #     hand_crop.shape
                            # )

                            # Convert refined bbox to original frame coordinates
                            # refined_bbox_global = {
                            #     'x': bbox['x'] + refined_bbox['x'],
                            #     'y': bbox['y'] + refined_bbox['y'],
                            #     'w': refined_bbox['w'],
                            #     'h': refined_bbox['h'],
                            #     'wrist_x': bbox['x'] + refined_bbox['wrist_x'],
                            #     'wrist_y': bbox['y'] + refined_bbox['wrist_y'],
                            #     'middle_tip_x': bbox['x'] + refined_bbox['middle_tip_x'],
                            #     'middle_tip_y': bbox['y'] + refined_bbox['middle_tip_y']
                            # }

                            # Draw refined bbox on stage 2 frame
                            # stage2_frame = draw_hand_bbox(stage2_frame, refined_bbox_global,
                            #                             color=(255, 0, 0), thickness=3)

                            # Place annotated crop back on stage 2 frame
                            stage2_frame[y:y+h, x:x+w] = hand_crop_annotated

                            # Add comparison info
                            cv2.putText(stage2_frame, f"Original bbox",
                                      (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (0, 255, 0), 1)
                            cv2.putText(stage2_frame, f"Refined bbox",
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (255, 0, 0), 2)

            # Calculate average processing times
            if len(stage1_times) > 30:
                stage1_times = stage1_times[-30:]
            if len(stage2_times) > 30:
                stage2_times = stage2_times[-30:]

            avg_stage1 = sum(stage1_times) / len(stage1_times) if stage1_times else 0
            avg_stage2 = sum(stage2_times) / len(stage2_times) if stage2_times else 0

            # Add performance info
            info_text = [
                f"Stage 1 time: {stage1_time:.1f}ms (avg: {avg_stage1:.1f}ms)",
                f"Stage 2 time: {stage2_times[-1] if stage2_times else 0:.1f}ms (avg: {avg_stage2:.1f}ms)",
                f"Hands detected: {len(hand_bboxes)}",
                f"Display mode: {display_mode}",
                "Press 1/2/b to switch display, q to quit"
            ]

            # Add info to both frames
            for i, text in enumerate(info_text):
                y_pos = 30 + i * 25
                cv2.putText(stage1_frame, text, (10, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(stage2_frame, text, (10, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display based on mode
            if display_mode == 'stage1':
                cv2.imshow('Two-Stage Hand Detection', stage1_frame)
            elif display_mode == 'stage2':
                cv2.imshow('Two-Stage Hand Detection', stage2_frame)
            else:  # both
                # Combine frames side by side
                combined = np.hstack([
                    cv2.resize(stage1_frame, (640, 360)),
                    cv2.resize(stage2_frame, (640, 360))
                ])
                # Add labels
                cv2.putText(combined, "Stage 1: Initial Detection", (10, 350),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, "Stage 2: Refined Detection", (650, 350),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow('Two-Stage Hand Detection', combined)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord('1'):
            display_mode = 'stage1'
            print("Showing Stage 1 only")
        elif key == ord('2'):
            display_mode = 'stage2'
            print("Showing Stage 2 only")
        elif key == ord('b'):
            display_mode = 'both'
            print("Showing both stages")
        elif key == ord('s') and not paused:
            # Save current frames
            save_count += 1
            cv2.imwrite(f'hand_stage1_{save_count}.jpg', stage1_frame)
            cv2.imwrite(f'hand_stage2_{save_count}.jpg', stage2_frame)
            print(f"Saved frames {save_count}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector_stage1.cleanup()
    detector_stage2.cleanup()

    # Print final statistics
    print("\n" + "=" * 60)
    print("Test completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Average Stage 1 time: {avg_stage1:.2f}ms")
    print(f"Average Stage 2 time: {avg_stage2:.2f}ms")
    print(f"Total average time: {avg_stage1 + avg_stage2:.2f}ms")
    print("=" * 60)


if __name__ == '__main__':
    test_two_stage_hand_detection_webcam()