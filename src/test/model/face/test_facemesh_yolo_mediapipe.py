#!/usr/bin/env python3
"""
Test script for YOLO Face Detection + MediaPipe FaceMesh
Uses YOLO to detect faces first, then applies FaceMesh to cropped regions
"""

import cv2
import numpy as np
import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.face.yolo import YOLOFaceDetector
from src.models.facemesh.mediapipe import MediaPipeFaceMeshDetector

def main():
    print("🧪 Testing YOLO Face Detection + MediaPipe FaceMesh")
    print("=" * 60)
    
    # Initialize YOLO face detector
    try:
        yolo_detector = YOLOFaceDetector(
            model_path='src/checkpoints/yolov8n-face.pt',
            confidence=0.5
        )
        print("✅ YOLO Face Detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize YOLO Face Detector: {e}")
        return
    
    # Initialize MediaPipe facemesh detector
    try:
        facemesh_detector = MediaPipeFaceMeshDetector(min_detection_confidence=0.3)
        print("✅ MediaPipe FaceMesh initialized")
    except Exception as e:
        print(f"❌ Failed to initialize FaceMesh: {e}")
        return
    
    # Initialize video source
    cap = cv2.VideoCapture('/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/video/multi-cam video/vid_shot1/face closeup/cam_1.mp4')
    # cap = cv2.VideoCapture(0)  # Uncomment for webcam
    if not cap.isOpened():
        print("❌ Failed to open video source")
        return
    
    # Get video properties for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    output_filename = 'yolo_facemesh_output_with_annotations.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Failed to initialize video writer")
        return
    
    print("📹 Video source opened successfully")
    print(f"📹 Output video will be saved as: {output_filename}")
    print("Press 'q' to quit, 'p' to pause/resume, 's' to save current frame")
    
    paused = False
    frame_count = 0
    total_faces_detected = 0
    total_frames_with_facemesh = 0
    total_yolo_time = 0
    total_facemesh_time = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            annotated_frame = frame.copy()
            current_frame_has_facemesh = False
            
            # Step 1: YOLO Face Detection
            yolo_start = time.time()
            yolo_results = yolo_detector.detect(frame)
            face_bboxes = yolo_detector.extract_bboxes_with_pad(yolo_results, frame.shape, padding=30)
            yolo_end = time.time()
            yolo_time = yolo_end - yolo_start
            total_yolo_time += yolo_time
            
            if face_bboxes:
                total_faces_detected += len(face_bboxes)
                print(f"Frame {frame_count}: Found {len(face_bboxes)} faces with YOLO")
                
                # Step 2: Apply MediaPipe FaceMesh to each cropped face region
                facemesh_start = time.time()
                for i, bbox in enumerate(face_bboxes):
                    # Extract face region using padded coordinates
                    x_pad, y_pad, w_pad, h_pad = bbox['x_pad'], bbox['y_pad'], bbox['w_pad'], bbox['h_pad']
                    face_crop = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    
                    if face_crop.size > 0:
                        # Apply MediaPipe FaceMesh to cropped face
                        facemesh_results = facemesh_detector.detect(face_crop)
                        face_landmarks = facemesh_detector.convert_to_dict(facemesh_results)
                        
                        if face_landmarks and len(face_landmarks) > 0:
                            current_frame_has_facemesh = True
                            
                            # Convert landmarks back to original frame coordinates
                            adjusted_landmarks = []
                            for landmark in face_landmarks:
                                # Convert from normalized crop coordinates to original frame coordinates
                                orig_x = int(landmark['x'] * w_pad + x_pad)
                                orig_y = int(landmark['y'] * h_pad + y_pad)
                                adjusted_landmarks.append({'x': orig_x, 'y': orig_y})
                            
                            # Draw landmarks on original frame
                            for landmark in adjusted_landmarks:
                                cv2.circle(annotated_frame, (landmark['x'], landmark['y']), 1, (0, 255, 0), -1)
                            
                            # Draw landmark count on face
                            cv2.putText(annotated_frame, f"Landmarks: {len(face_landmarks)}", 
                                       (bbox['x'], bbox['y'] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                
                facemesh_end = time.time()
                facemesh_time = facemesh_end - facemesh_start
                total_facemesh_time += facemesh_time
                
                # Count frames with successful facemesh detection
                if current_frame_has_facemesh:
                    total_frames_with_facemesh += 1
                
                # Draw YOLO bounding boxes using the built-in method
                annotated_frame = yolo_detector.draw_bboxes(annotated_frame, yolo_results, 
                                                          draw_padding=True, padding=30)
            
            # Add comprehensive info text
            cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(annotated_frame, f"Total Frames with FaceMesh: {total_frames_with_facemesh}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"Total Faces Detected: {total_faces_detected}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
            
            # Add timing info
            if frame_count > 0:
                avg_yolo_time = total_yolo_time / frame_count * 1000  # Convert to ms
                avg_facemesh_time = total_facemesh_time / frame_count * 1000  # Convert to ms
                cv2.putText(annotated_frame, f"YOLO: {avg_yolo_time:.1f}ms avg", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
                cv2.putText(annotated_frame, f"FaceMesh: {avg_facemesh_time:.1f}ms avg", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
            
            # Write frame to output video (with all annotations)
            out.write(annotated_frame)
            
            # Show frame
            cv2.imshow('YOLO + MediaPipe FaceMesh Test', annotated_frame)
            frame_count += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
        elif key == ord('s'):
            # Save current frame
            filename = f"yolo_facemesh_frame_{frame_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"💾 Saved frame as {filename}")
    
    # Cleanup
    cap.release()
    out.release()  # Close video writer
    cv2.destroyAllWindows()
    yolo_detector.cleanup()
    facemesh_detector.cleanup()
    
    print(f"\n✅ Test completed!")
    print(f"📊 Processing Summary:")
    print(f"   • Total frames processed: {frame_count}")
    print(f"   • Total faces detected by YOLO: {total_faces_detected}")
    print(f"   • Total frames with successful FaceMesh: {total_frames_with_facemesh}")
    if frame_count > 0:
        print(f"   • Average faces per frame: {total_faces_detected / frame_count:.2f}")
        print(f"   • FaceMesh success rate: {total_frames_with_facemesh / frame_count * 100:.1f}% of frames")
        print(f"   • Average YOLO processing time: {total_yolo_time / frame_count * 1000:.1f}ms")
        print(f"   • Average FaceMesh processing time: {total_facemesh_time / frame_count * 1000:.1f}ms")
        print(f"   • Total YOLO time: {total_yolo_time:.2f}s")
        print(f"   • Total FaceMesh time: {total_facemesh_time:.2f}s")
    print(f"📹 Output video saved as: {output_filename}")

if __name__ == "__main__":
    main()