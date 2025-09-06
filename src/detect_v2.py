#!/usr/bin/env python3
"""
Musician Tracking Detection System V2
Modular implementation with configurable model selection
"""

import cv2
import numpy as np
import yaml
import time
import os
import sys
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

# Add parent directory to path to find models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all model types
from models.hand.mediapipe import MediaPipeHandDetector
from models.hand.yolo import YOLOHandDetector

from models.pose.mediapipe import MediaPipePoseDetector
from models.pose.yolo import YOLOPoseDetector

from models.face.mediapipe import MediaPipeFaceDetector
from models.face.yolo import YOLOFaceDetector

from models.emotion.deepface import DeepFaceEmotionDetector
from models.emotion.ghostfacenet import GhostFaceNetEmotionDetector

# Import database
from src.database.setup import MusicianDatabase


class DetectorV2:
    """Main detector class for V2 implementation"""
    
    def __init__(self, config_path: str = 'src/config/config_v1.yaml'):
        """
        Initialize the detector with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.frame_count = 0
        self.session_id = f"session_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize models based on configuration
        self.hand_detector = self._init_hand_detector()
        self.pose_detector = self._init_pose_detector()
        self.face_detector = self._init_face_detector()
        self.emotion_detector = self._init_emotion_detector()
        
        # Initialize database if enabled
        self.db = None
        if self.config['database'].get('enabled', False):
            self._init_database()
        
        print(f"✅ DetectorV2 initialized")
        print(f"📝 Session ID: {self.session_id}")
        print(f"🤚 Hand Model: {self.config['detection'].get('hand_model', 'none')}")
        print(f"🏃 Pose Model: {self.config['detection'].get('pose_model', 'none')}")
        print(f"😊 Face Model: {self.config['detection'].get('facemesh_model', 'none')}")
        print(f"😢 Emotion Model: {self.config['detection'].get('emotion_model', 'none')}")
        print(f"💾 Database: {'Enabled' if self.db else 'Disabled'}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'database': {
                'enabled': False,
                'table_name': 'music_frame_analysis_1',
                'batch_size': 50,
                'batch_timeout': 5.0
            },
            'video': {
                'source_path': '0',
                'use_webcam': True,
                'skip_frames': 0,
                'display_output': True
            },
            'detection': {
                'hand_model': 'mediapipe',
                'pose_model': 'mediapipe',
                'facemesh_model': 'mediapipe',
                'emotion_model': 'none'
            }
        }
    
    def _init_hand_detector(self):
        """Initialize hand detection model based on config"""
        model_type = self.config['detection'].get('hand_model', 'none').lower()
        confidence = self.config['detection'].get('hand_confidence', 0.5)
        
        if model_type == 'mediapipe':
            return MediaPipeHandDetector(min_detection_confidence=confidence)
        elif model_type == 'yolo':
            return YOLOHandDetector(confidence=confidence)
        else:
            return None
    
    def _init_pose_detector(self):
        """Initialize pose detection model based on config"""
        model_type = self.config['detection'].get('pose_model', 'none').lower()
        confidence = self.config['detection'].get('pose_confidence', 0.5)
        
        if model_type == 'mediapipe':
            return MediaPipePoseDetector(min_detection_confidence=confidence)
        elif model_type == 'yolo':
            return YOLOPoseDetector(confidence=confidence)
        else:
            return None
    
    def _init_face_detector(self):
        """Initialize face detection model based on config"""
        model_type = self.config['detection'].get('facemesh_model', 'none').lower()
        confidence = self.config['detection'].get('facemesh_confidence', 0.5)
        
        if model_type == 'mediapipe':
            return MediaPipeFaceDetector(min_detection_confidence=confidence)
        elif model_type == 'yolo':
            return YOLOFaceDetector(confidence=confidence)
        else:
            return None
    
    def _init_emotion_detector(self):
        """Initialize emotion detection model based on config"""
        model_type = self.config['detection'].get('emotion_model', 'none').lower()
        
        if model_type == 'deepface':
            settings = self.config['detection']['emotion_settings']['deepface']
            return DeepFaceEmotionDetector(
                model_name=settings.get('model_name', 'Facenet'),
                detector_backend=settings.get('detector_backend', 'retinaface'),
                enforce_detection=settings.get('enforce_detection', False)
            )
        elif model_type == 'ghostfacenet':
            settings = self.config['detection']['emotion_settings']['ghostfacenet']
            return GhostFaceNetEmotionDetector(
                model_version=settings.get('model_version', 'v2'),
                batch_size=settings.get('batch_size', 32)
            )
        else:
            return None
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            table_name = self.config['database'].get('table_name', 'music_frame_analysis_1')
            self.db = MusicianDatabase(table_name=table_name)
            
            # Configure batch settings
            self.db.batch_size = self.config['database'].get('batch_size', 50)
            self.db.batch_timeout = self.config['database'].get('batch_timeout', 5.0)
            
            print(f"✅ Database initialized with table: {table_name}")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            self.db = None
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get status of all available models"""
        return {
            'hand': self.hand_detector is not None,
            'pose': self.pose_detector is not None,
            'face': self.face_detector is not None,
            'emotion': self.emotion_detector is not None,
            'database': self.db is not None
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through all models
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing all detection results
        """
        results = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'processing_times': {}
        }
        
        # Hand detection
        if self.hand_detector:
            start_time = time.time()
            hand_results = self.hand_detector.detect(frame)
            left_hand, right_hand = self.hand_detector.convert_to_dict(hand_results)
            results['left_hand_landmarks'] = left_hand
            results['right_hand_landmarks'] = right_hand
            results['processing_times']['hand'] = int((time.time() - start_time) * 1000)
        else:
            results['left_hand_landmarks'] = None
            results['right_hand_landmarks'] = None
            results['processing_times']['hand'] = 0
        
        # Pose detection
        if self.pose_detector:
            start_time = time.time()
            pose_results = self.pose_detector.detect(frame)
            pose_landmarks = self.pose_detector.convert_to_dict(pose_results)
            results['pose_landmarks'] = pose_landmarks
            results['processing_times']['pose'] = int((time.time() - start_time) * 1000)
        else:
            results['pose_landmarks'] = None
            results['processing_times']['pose'] = 0
        
        # Face detection
        if self.face_detector:
            start_time = time.time()
            face_results = self.face_detector.detect(frame)
            face_landmarks = self.face_detector.convert_to_dict(face_results)
            results['facemesh_landmarks'] = face_landmarks
            results['processing_times']['face'] = int((time.time() - start_time) * 1000)
        else:
            results['facemesh_landmarks'] = None
            results['processing_times']['face'] = 0
        
        # Emotion detection
        if self.emotion_detector:
            start_time = time.time()
            emotion_results = self.emotion_detector.detect(frame)
            emotion_data = self.emotion_detector.convert_to_dict(emotion_results)
            if emotion_data:
                results['emotions'] = emotion_data.get('emotions', {})
                results['dominant_emotion'] = emotion_data.get('dominant_emotion', 'neutral')
            else:
                results['emotions'] = {}
                results['dominant_emotion'] = None
            results['processing_times']['emotion'] = int((time.time() - start_time) * 1000)
        else:
            results['emotions'] = {}
            results['dominant_emotion'] = None
            results['processing_times']['emotion'] = 0
        
        # Calculate total processing time
        results['processing_times']['total'] = sum(results['processing_times'].values())
        
        return results
    
    def save_to_database(self, results: Dict[str, Any], video_file: str = None):
        """
        Save processing results to database using batch insert
        
        Args:
            results: Processing results from process_frame
            video_file: Name of the video file being processed
        """
        if not self.db:
            return
        
        try:
            # Add frame to batch
            self.db.add_frame_to_batch(
                session_id=self.session_id,
                frame_number=results['frame_number'],
                video_file=video_file,
                original_time=results['frame_number'] / 30.0,  # Assuming 30 fps
                synced_time=results['frame_number'] / 30.0,
                left_hand_landmarks=results.get('left_hand_landmarks'),
                right_hand_landmarks=results.get('right_hand_landmarks'),
                pose_landmarks=results.get('pose_landmarks'),
                facemesh_landmarks=results.get('facemesh_landmarks'),
                emotions=results.get('emotions', {}),
                bad_gestures=None,  # Will be null as requested
                processing_time_ms=results['processing_times']['total'],
                hand_processing_time_ms=results['processing_times']['hand'],
                pose_processing_time_ms=results['processing_times']['pose'],
                facemesh_processing_time_ms=results['processing_times']['face'],
                emotion_processing_time_ms=results['processing_times']['emotion'],
                hand_model=self.config['detection'].get('hand_model'),
                pose_model=self.config['detection'].get('pose_model'),
                facemesh_model=self.config['detection'].get('facemesh_model'),
                emotion_model=self.config['detection'].get('emotion_model')
            )
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
    
    def draw_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            results: Processing results
            
        Returns:
            Frame with drawn annotations
        """
        annotated_frame = frame.copy()
        
        # Draw hand landmarks
        if self.hand_detector and (results.get('left_hand_landmarks') or results.get('right_hand_landmarks')):
            # Create mock results for drawing (this is a simplification)
            # In production, you'd keep the original results for drawing
            pass
        
        # Draw pose landmarks
        if self.pose_detector and results.get('pose_landmarks'):
            pass
        
        # Draw face landmarks
        if self.face_detector and results.get('facemesh_landmarks'):
            pass
        
        # Draw emotion text
        if results.get('dominant_emotion'):
            cv2.putText(annotated_frame, f"Emotion: {results['dominant_emotion']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw processing time
        cv2.putText(annotated_frame, f"Processing: {results['processing_times']['total']}ms", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw frame number
        cv2.putText(annotated_frame, f"Frame: {results['frame_number']}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def process_video(self, video_path: str = None):
        """
        Process video file or webcam stream
        
        Args:
            video_path: Path to video file or None for webcam
        """
        # Determine video source
        if video_path is None:
            if self.config['video'].get('use_webcam', False):
                cap = cv2.VideoCapture(0)
                video_file = "webcam_session"
            else:
                video_path = self.config['video'].get('source_path', '0')
                cap = cv2.VideoCapture(video_path if video_path != '0' else 0)
                video_file = os.path.basename(video_path) if video_path != '0' else "webcam"
        else:
            cap = cv2.VideoCapture(video_path)
            video_file = os.path.basename(video_path)
        
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        
        print(f"📹 Processing: {video_file}")
        print(f"🎬 FPS: {fps}")
        
        skip_frames = self.config['video'].get('skip_frames', 0)
        display_output = self.config['video'].get('display_output', True)
        
        print(f"⚡ Skip frames: {skip_frames}")
        print(f"👁️ Display output: {display_output}")
        print("\nPress 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if skip_frames > 0 and self.frame_count % (skip_frames + 1) != 0:
                    self.frame_count += 1
                    continue
                
                # Process frame
                results = self.process_frame(frame)
                
                # Save to database
                self.save_to_database(results, video_file)
                
                # Display output if enabled
                if display_output:
                    annotated_frame = self.draw_results(frame, results)
                    cv2.imshow('Musician Tracking V2', annotated_frame)
                
                self.frame_count += 1
                
                # Print progress every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"📊 Processed {self.frame_count} frames...")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Flush any remaining database batches
        if self.db:
            self.db.flush_batch()
            self.db.close()
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Total frames processed: {self.frame_count}")
    
    def cleanup(self):
        """Cleanup all resources"""
        if self.hand_detector:
            self.hand_detector.cleanup()
        if self.pose_detector:
            self.pose_detector.cleanup()
        if self.face_detector:
            self.face_detector.cleanup()
        if self.emotion_detector:
            self.emotion_detector.cleanup()
        if self.db:
            self.db.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Musician Tracking Detection System V2')
    parser.add_argument('--config', '-c', default='src/config/config_v1.yaml',
                       help='Path to configuration file')
    parser.add_argument('--video', '-v', type=str,
                       help='Path to video file (overrides config)')
    parser.add_argument('--skip-frames', type=int,
                       help='Number of frames to skip (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DetectorV2(config_path=args.config)
    
    # Override config with command line arguments
    if args.skip_frames is not None:
        detector.config['video']['skip_frames'] = args.skip_frames
    
    try:
        # Process video
        detector.process_video(args.video)
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        detector.cleanup()
        print("🧹 Cleanup complete")


if __name__ == '__main__':
    main()