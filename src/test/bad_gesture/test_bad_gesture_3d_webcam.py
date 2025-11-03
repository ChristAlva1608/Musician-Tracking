#!/usr/bin/env python3
"""
Test Enhanced 3D Bad Gesture Detection with Webcam
Simple script to verify the 3D detection system works correctly
"""

import cv2
import numpy as np
import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
test_bad_gesture_dir = current_dir  # src/test/bad_gesture/
test_dir = os.path.dirname(test_bad_gesture_dir)  # src/test/
src_dir = os.path.dirname(test_dir)  # src/
project_root = os.path.dirname(src_dir)  # project root
sys.path.insert(0, project_root)

from src.bad_gesture.detector_3d import BadGestureDetector3D
from src.models.pose.mediapipe import MediaPipePoseDetector
from src.models.hand.mediapipe import MediaPipeHandDetector


def draw_info_panel(frame, results, score_info, stats):
    """Draw simplified information panel on the frame"""
    height, width = frame.shape[:2]

    # Create compact semi-transparent overlay at top
    overlay = frame.copy()
    panel_height = 110
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    y = 25
    line_height = 22

    # Posture score (larger, more prominent)
    score = score_info.get('score', 100)
    grade = score_info.get('grade', 'A')
    score_color = (0, 255, 0) if score >= 80 else (0, 165, 255) if score >= 60 else (0, 0, 255)

    cv2.putText(frame, f"Score: {score}/100 ({grade})", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
    y += line_height + 5

    # Only show DETECTED gestures (compact, single line)
    detected_gestures = []
    if results.get('turtle_neck', False):
        detected_gestures.append("Turtle Neck")
    if results.get('hunched_back', False):
        detected_gestures.append("Hunched Back")
    if results.get('low_wrists', False):
        detected_gestures.append("Low Wrists")
    if results.get('fingers_pointing_up', False):
        detected_gestures.append("Pinky Up")

    if detected_gestures:
        warnings_text = "Issues: " + ", ".join(detected_gestures)
        cv2.putText(frame, warnings_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += line_height
    else:
        cv2.putText(frame, "Status: Good Posture", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += line_height

    # Simplified controls at bottom
    cv2.putText(frame, "Press: 's'=Score | 'd'=Details | 'r'=Reset | 'q'=Quit",
               (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def print_detailed_analysis(detector):
    """Print detailed analysis to console"""
    analysis = detector.get_detailed_analysis()
    score = detector.get_posture_score()

    print("\n" + "="*80)
    print("DETAILED POSTURE ANALYSIS")
    print("="*80)
    print(f"\nOverall Score: {score['score']}/100 (Grade: {score['grade']})")
    print(f"Feedback: {score['feedback']}")

    if score.get('deductions'):
        print("\nDeductions:")
        for deduction in score['deductions']:
            print(f"  {deduction['issue']}: -{deduction['points_lost']} points "
                  f"({deduction['severity']}, confidence: {deduction['confidence']:.1%})")

    print("\n" + "-"*80)
    print("GESTURE DETAILS")
    print("-"*80)

    for gesture_type, details in analysis.items():
        if details.get('detected'):
            print(f"\n=4 {gesture_type.upper().replace('_', ' ')}:")
            print(f"   Severity: {details.get('severity', 'unknown')}")
            print(f"   Confidence: {details.get('confidence', 0):.1%}")

            # Print key measurements
            measurements = details.get('measurements', {})
            if measurements:
                print("   Key Measurements:")
                count = 0
                for key, value in measurements.items():
                    if count >= 5:  # Show top 5 measurements
                        break
                    if isinstance(value, (int, float)):
                        if 'ratio' in key:
                            print(f"     {key}: {value:.3f} ({value*100:.1f}%)")
                            count += 1
                        elif 'angle' in key or 'degrees' in key:
                            print(f"     {key}: {value:.1f}°")
                            count += 1
                        elif 'percentage' in key:
                            print(f"     {key}: {value:.1f}%")
                            count += 1

            # Print corrections
            corrections = details.get('correction_needed', {})
            if corrections:
                print("   Corrections:")
                if isinstance(corrections, dict):
                    for key, value in corrections.items():
                        if isinstance(value, dict):
                            print(f"     {key}: {value.get('action', value)}")
                        else:
                            print(f"     {key}: {value}")
                elif isinstance(corrections, list):
                    for correction in corrections:
                        if isinstance(correction, dict):
                            print(f"     {correction.get('action', correction)}")
                        else:
                            print(f"     {correction}")
                else:
                    print(f"     {corrections}")

    print("\n" + "="*80 + "\n")


def test_3d_detection_webcam():
    """Main test function for webcam testing"""

    print("="*80)
    print("ENHANCED 3D BAD GESTURE DETECTION - WEBCAM TEST")
    print("="*80)
    print("\nInitializing detectors...")

    # Initialize detectors
    try:
        pose_detector = MediaPipePoseDetector()
        hand_detector = MediaPipeHandDetector()
        # Disable internal drawing to avoid overlap with our custom info panel
        bad_gesture_detector = BadGestureDetector3D(draw_on_frame=False)
        print(" Detectors initialized successfully")
    except Exception as e:
        print(f"L Failed to initialize detectors: {e}")
        import traceback
        traceback.print_exc()
        return

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("L Cannot open webcam")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(" Webcam opened successfully")
    print("\n" + "="*80)
    print("INSTRUCTIONS")
    print("="*80)
    print("  Stand/sit in front of camera so full upper body is visible")
    print("  Try different postures to test detection:")
    print("    - Lean head forward (turtle neck)")
    print("    - Round shoulders forward (hunched back)")
    print("    - Drop wrists below arm level (low wrists)")
    print("    - Point fingers upward (fingers pointing up)")
    print("\nKEYBOARD CONTROLS:")
    print("  's' - Show detailed posture score")
    print("  'd' - Print detailed analysis to console")
    print("  'r' - Reset statistics")
    print("  'q' - Quit")
    print("="*80 + "\n")

    frame_count = 0
    fps_start_time = cv2.getTickCount()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("L Failed to read frame")
                break

            frame_count += 1

            # Calculate FPS
            if frame_count % 30 == 0:
                fps_end_time = cv2.getTickCount()
                time_elapsed = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                fps = 30 / time_elapsed if time_elapsed > 0 else 0
                fps_start_time = fps_end_time

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect pose
            pose_results = pose_detector.detect(frame)
            pose_landmarks_3d = pose_results.pose_world_landmarks if pose_results else None
            pose_landmarks_2d = pose_results.pose_landmarks if pose_results else None

            # Detect hands
            hand_results = hand_detector.detect(frame)
            hand_landmarks_3d = hand_results.multi_hand_world_landmarks if hand_results else None
            hand_landmarks_2d = hand_results.multi_hand_landmarks if hand_results else None

            # Detect bad gestures using 3D coordinates
            bad_gestures = bad_gesture_detector.detect_all_gestures(
                frame=frame,
                hand_landmarks=hand_landmarks_3d,
                pose_landmarks=pose_landmarks_3d
            )

            # Draw pose landmarks (use regular landmarks for 2D visualization)
            if pose_landmarks_2d:
                import mediapipe as mp
                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose

                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks_2d,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

            # Draw hand landmarks (use 2D landmarks for visualization)
            if hand_landmarks_2d:
                import mediapipe as mp
                mp_drawing = mp.solutions.drawing_utils
                mp_hands = mp.solutions.hands

                for hand_landmarks_obj in hand_landmarks_2d:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks_obj,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

            # Get statistics and score
            stats = bad_gesture_detector.get_gesture_statistics()
            score = bad_gesture_detector.get_posture_score()

            # Draw info panel
            draw_info_panel(frame, bad_gestures, score, stats)

            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show frame
            cv2.imshow('Enhanced 3D Bad Gesture Detection - Webcam Test', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n=K Quitting...")
                break

            elif key == ord('s'):
                # Show score overlay
                score = bad_gesture_detector.get_posture_score()
                print(f"\n=� Current Score: {score['score']}/100 (Grade: {score['grade']})")
                print(f"   {score['feedback']}")

            elif key == ord('d'):
                # Print detailed analysis
                print_detailed_analysis(bad_gesture_detector)

            elif key == ord('r'):
                # Reset statistics
                bad_gesture_detector.reset_statistics()
                print("\n= Statistics reset")

    except KeyboardInterrupt:
        print("\n\n� Test interrupted by user")

    except Exception as e:
        print(f"\nL Error during processing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Show final statistics
        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)

        stats = bad_gesture_detector.get_gesture_statistics()
        counts = bad_gesture_detector.get_gesture_counts()
        final_score = bad_gesture_detector.get_posture_score()

        print(f"\nTotal frames processed: {frame_count}")
        print(f"Final Posture Score: {final_score['score']}/100 (Grade: {final_score['grade']})")
        print(f"Feedback: {final_score['feedback']}")

        print("\nGesture Detection Summary:")
        for gesture, count in counts.items():
            percentage = stats.get(gesture, 0)
            gesture_name = gesture.replace('_', ' ').title()
            print(f"  {gesture_name}: {count} frames ({percentage:.1f}% of time)")

        print("\n" + "="*80)
        print(" Test completed successfully!")
        print("="*80 + "\n")


if __name__ == "__main__":
    try:
        test_3d_detection_webcam()
    except Exception as e:
        print(f"\nL Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
