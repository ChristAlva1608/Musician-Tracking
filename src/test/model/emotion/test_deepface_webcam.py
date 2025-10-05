#!/usr/bin/env python3
"""
Test DeepFace emotion detection with webcam only
"""

import cv2
import time
import numpy as np
from src.models.emotion import DeepFaceDetector
import yaml

def load_config(config_path='src/config/config_v1.yaml'):
    """Load configuration"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except:
        # Fallback config
        return {
            'detection': {
                'emotion_settings': {
                    'deepface': {
                        'model_name': 'Facenet',
                        'detector_backend': 'retinaface',
                        'enforce_detection': False
                    }
                }
            }
        }

def test_deepface_webcam():
    """Test DeepFace with webcam or video file"""
    print("🎥 DeepFace Real-time Test")
    print("=" * 40)
    
    # Load config and initialize detector
    config = load_config()
    emotion_detector = DeepFaceDetector(config['detection'].get('emotion_settings', {}))
    
    print("🔧 Initializing DeepFace...")
    init_start = time.time()
    if not emotion_detector.initialize_model():
        print("❌ Failed to initialize DeepFace")
        return
    init_time = time.time() - init_start
    print(f"✅ DeepFace initialized in {init_time:.2f}s")
    
    # Try webcam first, then fallback to video file
    print("\n📹 Trying webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠️ Webcam not available, trying video file...")
        # Fallback to video file
        video_path = config.get('video', {}).get('source_path', '/Volumes/Extreme_Pro/Mitou Project/Musician Tracking/video/multi-cam video/vid_shot1/vid_shot1_cam_1.mp4')
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam or video file")
            return
        else:
            print(f"✅ Using video file: {video_path}")
    else:
        print("✅ Using webcam")
    
    # Set webcam resolution (lower = faster)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Webcam opened successfully")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to show/hide stats")
    print("  - Press 'f' to toggle face detection")
    
    # Performance tracking
    frame_count = 0
    total_inference_time = 0
    show_stats = True
    detect_faces = True
    fps_counter = 0
    fps_start = time.time()
    
    print(f"\n🚀 Starting real-time detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_start = time.time()
        
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # DeepFace emotion detection
        emotions_detected = []
        inference_time = 0
        
        if detect_faces:
            inference_start = time.time()
            try:
                emotions_detected = emotion_detector.detect_emotions(frame)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
            except Exception as e:
                print(f"⚠️ Detection error: {e}")
        
        # Draw results
        if emotions_detected:
            for emotion_info in emotions_detected:
                if 'bbox' in emotion_info:
                    x, y, w, h = emotion_info['bbox']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Get emotion info
                    if 'dominant_emotion' in emotion_info:
                        emotion = emotion_info['dominant_emotion']
                        confidence = emotion_info['confidence']
                        
                        # Draw emotion text
                        text = f"{emotion}: {confidence:.2f}"
                        cv2.putText(frame, text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw all emotion scores (smaller text)
                        if 'emotions' in emotion_info:
                            emotions = emotion_info['emotions']
                            y_offset = y + h + 20
                            for i, (emo, score) in enumerate(emotions.items()):
                                if score > 0.1:  # Only show significant emotions
                                    emo_text = f"{emo}: {score:.2f}"
                                    cv2.putText(frame, emo_text, (x, y_offset + i*15), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        else:
            current_fps = 0
        
        # Draw performance stats
        if show_stats:
            frame_time = (time.time() - frame_start) * 1000
            avg_inference = (total_inference_time / max(frame_count, 1)) * 1000
            
            # Background for stats
            cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, 120), (255, 255, 255), 1)
            
            # Stats text
            stats = [
                f"Frame: {frame_count}",
                f"Frame time: {frame_time:.0f}ms",
                f"Inference: {inference_time*1000:.0f}ms",
                f"Avg inference: {avg_inference:.0f}ms",
                f"FPS: {current_fps:.1f}" if current_fps > 0 else "FPS: calculating...",
                f"Faces: {len(emotions_detected)}"
            ]
            
            for i, stat in enumerate(stats):
                cv2.putText(frame, stat, (15, 30 + i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Status indicators
        status_color = (0, 255, 0) if detect_faces else (0, 0, 255)
        status_text = "DETECTING" if detect_faces else "PAUSED"
        cv2.putText(frame, status_text, (frame.shape[1]-100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Show frame
        cv2.imshow('DeepFace Webcam Test', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Quitting...")
            break
        elif key == ord('s'):
            show_stats = not show_stats
            print(f"📊 Stats display: {'ON' if show_stats else 'OFF'}")
        elif key == ord('f'):
            detect_faces = not detect_faces
            print(f"🔍 Face detection: {'ON' if detect_faces else 'OFF'}")
        
        frame_count += 1
        
        # Print periodic performance summary
        if frame_count % 30 == 0 and frame_count > 0:
            avg_inference_ms = (total_inference_time / frame_count) * 1000
            print(f"📊 Frame {frame_count}: Avg inference {avg_inference_ms:.0f}ms")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    if frame_count > 0:
        avg_inference_ms = (total_inference_time / frame_count) * 1000
        print(f"\n📈 FINAL STATS:")
        print(f"   Total frames: {frame_count}")
        print(f"   Avg inference time: {avg_inference_ms:.0f}ms")
        print(f"   Total inference time: {total_inference_time:.1f}s")
        
        if avg_inference_ms > 1000:
            print(f"⚠️  DeepFace is too slow for real-time ({avg_inference_ms:.0f}ms per frame)")
            print("   Consider switching to FER or other faster models")
        elif avg_inference_ms > 500:
            print(f"⚠️  DeepFace is slow ({avg_inference_ms:.0f}ms per frame)")
            print("   May struggle with real-time processing")
        else:
            print(f"✅ DeepFace performance is acceptable ({avg_inference_ms:.0f}ms per frame)")
    
    # Cleanup detector
    emotion_detector.cleanup()
    print("🧹 DeepFace cleaned up")

if __name__ == "__main__":
    test_deepface_webcam()