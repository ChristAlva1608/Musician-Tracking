#!/usr/bin/env python3
"""
Performance profiling version of detect.py to identify bottlenecks
"""

import time
import yaml
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.setup import MusicianDatabase
from src.models.emotion import DeepFaceDetector
from ultralytics import YOLO
import mediapipe as mp
import os
from datetime import datetime

def load_config(config_path='src/config/config_v1.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def profile_models():
    """Profile individual model loading and inference times"""
    print("🔍 PERFORMANCE PROFILING")
    print("=" * 50)
    
    # Load config
    config = load_config()
    VIDEO_PATH = config['video'].get('source_path')
    
    # Test frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, test_frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read test frame")
        return
        
    img_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    print(f"📐 Frame size: {test_frame.shape}")
    
    # 1. Test MediaPipe Hand Loading
    print("\n1️⃣ MEDIAPIPE HAND DETECTION")
    start = time.time()
    hand_model = mp.solutions.hands.Hands()
    load_time = time.time() - start
    print(f"   Loading: {load_time:.3f}s")
    
    # Test inference
    start = time.time()
    hand_results = hand_model.process(img_rgb)
    inference_time = time.time() - start
    print(f"   Inference: {inference_time:.3f}s")
    print(f"   Hands detected: {len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0}")
    
    # 2. Test YOLO Pose Loading  
    print("\n2️⃣ YOLO POSE DETECTION")
    start = time.time()
    pose_model = YOLO('src/checkpoints/yolo11n-pose.pt')
    load_time = time.time() - start
    print(f"   Loading: {load_time:.3f}s")
    
    # Test inference
    start = time.time()
    pose_results = pose_model(test_frame, verbose=False)
    inference_time = time.time() - start
    print(f"   Inference: {inference_time:.3f}s")
    print(f"   Poses detected: {len(pose_results) if pose_results else 0}")
    
    # 3. Test DeepFace Emotion Loading
    print("\n3️⃣ DEEPFACE EMOTION DETECTION")
    start = time.time()
    emotion_detector = DeepFaceDetector(config['detection'].get('emotion_settings', {}))
    emotion_detector.initialize_model()
    load_time = time.time() - start
    print(f"   Loading: {load_time:.3f}s")
    
    # Test inference (multiple times to get average)
    inference_times = []
    for i in range(3):
        start = time.time()
        emotions = emotion_detector.detect_emotions(test_frame)
        inference_time = time.time() - start
        inference_times.append(inference_time)
        print(f"   Inference #{i+1}: {inference_time:.3f}s")
    
    avg_emotion_time = np.mean(inference_times)
    print(f"   Average Inference: {avg_emotion_time:.3f}s")
    print(f"   Emotions detected: {len(emotions) if emotions else 0}")
    
    # 4. Test Database Connection
    print("\n4️⃣ DATABASE OPERATIONS")
    try:
        start = time.time()
        db = MusicianDatabase()
        connect_time = time.time() - start
        print(f"   Connection: {connect_time:.3f}s")
        
        # Test insert
        test_data = {
            'session_id': 'profile_test',
            'frame_number': 0,
            'video_file': 'test.mp4',
            'original_time': 0.0,
            'synced_time': 0.0,
            'processing_time_ms': 50,
            'hand_model': 'mediapipe',
            'pose_model': 'yolo', 
            'emotion_model': 'deepface'
        }
        
        start = time.time()
        db.insert_frame_data(**test_data)
        insert_time = time.time() - start
        print(f"   Insert: {insert_time:.3f}s")
        
        # Cleanup
        db.supabase.table('musician_frame_analysis').delete().eq('session_id', 'profile_test').execute()
        
    except Exception as e:
        print(f"   Database error: {e}")
    
    # 5. Estimated total per-frame time
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 30)
    total_per_frame = inference_time + avg_emotion_time + insert_time + 0.050  # +50ms for other operations
    print(f"Hand detection: ~{inference_time*1000:.0f}ms")
    print(f"Pose detection: ~{inference_time*1000:.0f}ms") 
    print(f"Emotion detection: ~{avg_emotion_time*1000:.0f}ms")
    print(f"Database insert: ~{insert_time*1000:.0f}ms")
    print(f"Other operations: ~50ms")
    print("-" * 30)
    print(f"TOTAL per frame: ~{total_per_frame*1000:.0f}ms")
    print(f"Expected FPS: ~{1/total_per_frame:.1f}")
    
    if total_per_frame > 1.0:
        print("\n⚠️  PERFORMANCE ISSUES DETECTED:")
        if avg_emotion_time > 2.0:
            print(f"   • DeepFace is very slow ({avg_emotion_time:.1f}s)")
            print("     Consider: FER model, disable emotions, or batch processing")
        if insert_time > 0.5:
            print(f"   • Database inserts are slow ({insert_time:.1f}s)")
            print("     Consider: Batch inserts, local buffering")

if __name__ == "__main__":
    profile_models()