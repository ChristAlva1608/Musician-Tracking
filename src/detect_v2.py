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
import re
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

# Add parent directory to path to find models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all model types
from models.hand.mediapipe import MediaPipeHandDetector
from models.hand.yolo import YOLOHandDetector

from models.pose.mediapipe import MediaPipePoseDetector
from models.pose.yolo import YOLOPoseDetector

from models.facemesh.mediapipe import MediaPipeFaceMeshDetector
from models.face.yolo import YOLOFaceDetector

# Import bad gesture detection
from src.bad_gesture_detection import BadGestureDetector

from models.emotion.deepface import DeepFaceEmotionDetector
from models.emotion.ghostfacenet import GhostFaceNetEmotionDetector

# Import database
from src.database.setup import MusicianDatabase, VideoAlignmentDatabase

# Import transcript model
from models.transcript.whisper_realtime import WhisperRealtimeTranscriber


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
        self.start_time = None  # Will be set when processing starts
        self.start_time_offset = 0.0  # Retrieved from video_alignment_offsets table
        self.matching_duration = 0.0  # Retrieved from video_alignment_offsets table
        self.end_time_offset = None  # Calculated as start_time_offset + matching_duration
        
        # Statistics for report generation
        self.stats = {
            'total_frames': 0,
            'successful_hand_detections': 0,
            'successful_pose_detections': 0,
            'successful_facemesh_detections': 0,
            'successful_emotion_detections': 0,
            'successful_transcript_detections': 0,
            'total_processing_times': {
                'hand': 0,
                'pose': 0,
                'face': 0,
                'emotion': 0,
                'bad_gestures': 0,
                'transcript': 0,
                'total': 0
            }
        }
        
        # Video writer for output
        self.video_writer = None
        self.preserve_audio = False
        self.temp_video_path = None
        self.final_video_path = None
        self.original_video_path = None
        
        # Transcript processing
        self.transcript_data = []  # Store transcript segments with timestamps
        self.current_transcript_text = ""  # Current transcript for display
        self.video_path_for_transcript = None  # Store video path for transcript processing
        self.transcript_segments_saved = False  # Track if segments are saved to DB
        self.transcript_segment_map = {}  # Map time ranges to segment IDs
        
        # Initialize video alignment database
        self.alignment_db = None
        try:
            self.alignment_db = VideoAlignmentDatabase()
        except Exception as e:
            print(f"⚠️ VideoAlignmentDatabase connection failed: {e}")
        
        # Initialize models based on configuration
        self.hand_detector = self._init_hand_detector()
        self.pose_detector = self._init_pose_detector()
        self.facemesh_detector = self._init_facemesh_detector()
        self.emotion_detector = self._init_emotion_detector()
        self.bad_gesture_detector = self._init_bad_gesture_detector()
        self.transcript_detector = self._init_transcript_detector()
        
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
        print(f"🎤 Transcript Model: {self.config['detection'].get('transcript_model', 'none')}")
        print(f"⚠️ Bad Gestures: {'Enabled' if self.bad_gesture_detector else 'Disabled'}")
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
    
    def _init_facemesh_detector(self):
        """Initialize face detection model based on config"""
        model_type = self.config['detection'].get('facemesh_model', 'none').lower()
        confidence = self.config['detection'].get('facemesh_confidence', 0.5)
        
        if model_type == 'mediapipe':
            # Option 1: Only MediaPipe FaceMesh (similar to test_facemesh_mediapipe.py)
            return MediaPipeFaceMeshDetector(min_detection_confidence=confidence)
        elif model_type == 'yolo+mediapipe':
            # Option 2: YOLO face detection + MediaPipe FaceMesh (similar to test_facemesh_yolo_mediapipe.py)
            face_confidence = self.config['detection'].get('face_confidence', 0.5)
            yolo_model_path = self.config['detection'].get('yolo_face_model_path', 'checkpoints/yolov8n-face.pt')
            
            # Initialize both detectors
            yolo_detector = YOLOFaceDetector(
                model_path=yolo_model_path,
                confidence=face_confidence
            )
            facemesh_detector = MediaPipeFaceMeshDetector(min_detection_confidence=confidence)
            
            # Return both as a tuple to indicate hybrid approach
            return {'yolo': yolo_detector, 'mediapipe': facemesh_detector, 'type': 'hybrid'}
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
    
    def _init_bad_gesture_detector(self):
        """Initialize bad gesture detector based on config"""
        bad_gesture_config = self.config.get('bad_gestures', {})
        return BadGestureDetector(config=bad_gesture_config)
    
    def _init_transcript_detector(self):
        """Initialize transcript detector based on config"""
        model_type = self.config['detection'].get('transcript_model', 'none').lower()
        transcript_enabled = self.config['detection'].get('transcript_settings', {}).get('enabled', False)
        
        if model_type == 'whisper' and transcript_enabled:
            try:
                settings = self.config['detection']['transcript_settings']
                return WhisperRealtimeTranscriber(
                    model_size=settings.get('model_size', 'tiny'),
                    language=settings.get('language', 'en')
                )
            except ImportError:
                print("❌ Whisper not available. Install with: pip install openai-whisper")
                return None
            except Exception as e:
                print(f"❌ Error initializing transcript detector: {e}")
                return None
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
            'face': self.facemesh_detector is not None,
            'emotion': self.emotion_detector is not None,
            'transcript': self.transcript_detector is not None,
            'bad_gestures': self.bad_gesture_detector is not None,
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
            results['hand_raw_results'] = hand_results  # Store raw results for drawing
            
            # Add hand_bboxes for YOLO models
            hand_model_type = self.config['detection'].get('hand_model', 'none').lower()
            if hand_model_type == 'yolo' and hand_results:
                # Extract bounding boxes from YOLO results
                hand_bboxes = []
                try:
                    for result in hand_results:
                        if result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            for box, conf in zip(boxes, confidences):
                                x1, y1, x2, y2 = box
                                hand_bboxes.append({
                                    'x1': float(x1), 'y1': float(y1), 
                                    'x2': float(x2), 'y2': float(y2),
                                    'confidence': float(conf)
                                })
                except Exception as e:
                    print(f"⚠️ Error extracting hand bboxes: {e}")
                    hand_bboxes = []
                results['hand_bboxes'] = hand_bboxes
            else:
                results['hand_bboxes'] = None
                
            results['processing_times']['hand'] = float(f"{(time.time() - start_time) * 1000:.3f}")
        else:
            results['left_hand_landmarks'] = None
            results['right_hand_landmarks'] = None
            results['hand_raw_results'] = None
            results['hand_bboxes'] = None
            results['processing_times']['hand'] = 0
        
        # Pose detection
        if self.pose_detector:
            start_time = time.time()
            pose_results = self.pose_detector.detect(frame)
            pose_landmarks = self.pose_detector.convert_to_dict(pose_results)
            results['pose_landmarks'] = pose_landmarks
            results['pose_raw_results'] = pose_results  # Store raw results for drawing
            results['processing_times']['pose'] = float(f"{(time.time() - start_time) * 1000:.3f}")
        else:
            results['pose_landmarks'] = None
            results['pose_raw_results'] = None
            results['processing_times']['pose'] = 0
        
        # Face detection
        if self.facemesh_detector:
            start_time = time.time()
            
            # Check if hybrid YOLO+MediaPipe approach
            if isinstance(self.facemesh_detector, dict) and self.facemesh_detector.get('type') == 'hybrid':
                yolo_detector = self.facemesh_detector['yolo']
                facemesh_detector = self.facemesh_detector['mediapipe']
                
                # Step 1: YOLO Face Detection
                yolo_results = yolo_detector.detect(frame)
                face_bboxes = yolo_detector.extract_bboxes_with_pad(yolo_results, frame.shape, padding=30)
                
                face_landmarks = []
                if face_bboxes:
                    # Step 2: Apply MediaPipe FaceMesh to each cropped face region
                    for bbox in face_bboxes:
                        x_pad, y_pad, w_pad, h_pad = bbox['x_pad'], bbox['y_pad'], bbox['w_pad'], bbox['h_pad']
                        face_crop = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                        
                        if face_crop.size > 0:
                            facemesh_results = facemesh_detector.detect(face_crop)
                            crop_landmarks = facemesh_detector.convert_to_dict(facemesh_results)
                            
                            if crop_landmarks and len(crop_landmarks) > 0:
                                # Convert landmarks back to original frame coordinates
                                adjusted_landmarks = []
                                for landmark in crop_landmarks:
                                    orig_x = landmark['x'] * w_pad + x_pad
                                    orig_y = landmark['y'] * h_pad + y_pad
                                    adjusted_landmarks.append({'x': orig_x, 'y': orig_y})
                                face_landmarks.extend(adjusted_landmarks)
                
                results['facemesh_landmarks'] = face_landmarks if face_landmarks else None
                results['face_bboxes'] = face_bboxes  # Store YOLO bounding boxes for drawing
                results['facemesh_raw_results'] = {'yolo': yolo_results, 'mediapipe': None}  # Store raw results for drawing
            else:
                # Standard single-model approach (MediaPipe only)
                face_results = self.facemesh_detector.detect(frame)
                face_landmarks = self.facemesh_detector.convert_to_dict(face_results)
                results['facemesh_landmarks'] = face_landmarks
                results['face_bboxes'] = None
                results['facemesh_raw_results'] = face_results  # Store raw results for drawing
                
            results['processing_times']['face'] = float(f"{(time.time() - start_time) * 1000:.3f}")
        else:
            results['facemesh_landmarks'] = None
            results['face_bboxes'] = None
            results['facemesh_raw_results'] = None
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
            results['processing_times']['emotion'] = float(f"{(time.time() - start_time) * 1000:.3f}")
        else:
            results['emotions'] = {}
            results['dominant_emotion'] = None
            results['processing_times']['emotion'] = 0
        
        # Bad gesture detection
        if self.bad_gesture_detector:
            start_time = time.time()
            
            # Convert hand landmarks from dict format back to MediaPipe format for gesture detection
            hand_landmarks = None
            if results['hand_raw_results'] and self.config['detection'].get('hand_model', 'none').lower() == 'mediapipe':
                # For MediaPipe hand model, hand_raw_results contains MediaPipe results directly
                hand_landmarks = results['hand_raw_results'].multi_hand_landmarks if hasattr(results['hand_raw_results'], 'multi_hand_landmarks') else None
            
            bad_gestures = self.bad_gesture_detector.detect_all_gestures(
                frame=frame,
                hand_landmarks=hand_landmarks,
                pose_results=results['pose_raw_results']
            )
            
            results['bad_gestures'] = bad_gestures
            results['processing_times']['bad_gestures'] = float(f"{(time.time() - start_time) * 1000:.3f}")
        else:
            results['bad_gestures'] = {
                'low_wrists': False,
                'turtle_neck': False,
                'hunched_back': False,
                'fingers_pointing_up': False
            }
            results['processing_times']['bad_gestures'] = 0
        
        # Calculate total processing time (will be updated after transcript processing)
        current_total = sum(v for k, v in results['processing_times'].items() if k != 'total')
        results['processing_times']['total'] = float(f"{current_total:.3f}")
        
        return results
    
    def _get_alignment_data(self, video_file: str) -> tuple[float, float]:
        """
        Retrieve start_time_offset and matching_duration from video_alignment_offsets table
        
        Args:
            video_file: Video file name (e.g., "cam_1.mp4")
            
        Returns:
            Tuple of (start_time_offset, matching_duration) in seconds
        """
        if not self.alignment_db:
            return 0.0, 0.0
            
        try:
            # Extract source name and camera type from video file
            # For example: "cam_1.mp4" -> source="vid_shot1", camera_type=1
            # We need to match this with the database records
            
            # Extract camera number from filename
            video_basename = os.path.basename(video_file)
            camera_type = 1  # Default
            
            if video_basename.startswith('cam_') and video_basename.endswith('.mp4'):
                try:
                    camera_type = int(video_basename.split('_')[1].split('.')[0])
                except (ValueError, IndexError):
                    camera_type = 1
            else:
                # Fallback: extract any number from filename
                numbers = re.findall(r'\d+', video_basename)
                camera_type = int(numbers[0]) if numbers else 1
            
            # Get source name from config or try to detect from video path
            source_name = self.config.get('video_source', 'vid_shot1')
            
            # Alternative: try to extract source from video file path
            # Look for vid_shot pattern in the video file path
            video_path = self.config['video'].get('source_path', '')
            path_parts = video_path.split('/')
            vid_shot_parts = [part for part in path_parts if part.startswith('vid_shot')]
            if vid_shot_parts:
                source_name = vid_shot_parts[0]
            
            print(f"🔍 Looking for alignment data: source={source_name}, camera_type={camera_type}")
            
            # Get alignment data
            result = self.alignment_db.supabase.table('video_alignment_offsets').select('*').eq(
                'source', source_name
            ).eq('camera_type', camera_type).execute()
            
            if result.data and len(result.data) > 0:
                data = result.data[0]
                start_offset = float(data.get('start_time_offset', 0.0))
                matching_duration = float(data.get('matching_duration', 0.0))
                print(f"✅ Found alignment data: offset={start_offset:.3f}s, duration={matching_duration:.1f}s")
                return start_offset, matching_duration
            else:
                print(f"⚠️ No alignment data found for source={source_name}, camera_type={camera_type}")
                
        except Exception as e:
            print(f"⚠️ Error retrieving alignment data: {e}")
        
        return 0.0, 0.0
    
    def _update_statistics(self, results: Dict[str, Any]):
        """Update statistics for report generation"""
        self.stats['total_frames'] += 1
        
        # Count successful detections
        if results.get('left_hand_landmarks') or results.get('right_hand_landmarks'):
            self.stats['successful_hand_detections'] += 1
        
        if results.get('pose_landmarks'):
            self.stats['successful_pose_detections'] += 1
        
        if results.get('facemesh_landmarks'):
            self.stats['successful_facemesh_detections'] += 1
        
        if results.get('dominant_emotion'):
            self.stats['successful_emotion_detections'] += 1
        
        if results['processing_times'].get('transcript', 0) > 0:
            self.stats['successful_transcript_detections'] += 1
        
        # Accumulate processing times
        for key in ['hand', 'pose', 'face', 'emotion', 'bad_gestures', 'transcript', 'total']:
            self.stats['total_processing_times'][key] += results['processing_times'].get(key, 0)
    
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
            # Calculate proper timing values considering video alignment
            original_time = results['frame_number'] / 30.0  # Assuming 30 fps
            synced_time = self.start_time_offset + original_time
            
            # Get transcript segment ID for this frame
            transcript_segment_id = None
            
            if self.transcript_data and self.config['database'].get('link_frame_to_transcript', True):
                transcript_segment_id = self._get_transcript_segment_id(synced_time)
            
            # Add frame to batch
            self.db.add_frame_to_batch(
                session_id=self.session_id,
                frame_number=results['frame_number'],
                video_file=video_file,
                original_time=float(f"{original_time:.3f}"),
                synced_time=float(f"{synced_time:.3f}"),
                left_hand_landmarks=results.get('left_hand_landmarks'),
                right_hand_landmarks=results.get('right_hand_landmarks'),
                pose_landmarks=results.get('pose_landmarks'),
                facemesh_landmarks=results.get('facemesh_landmarks'),
                emotions=results.get('emotions', {}),
                bad_gestures=results.get('bad_gestures', {}),
                processing_time_ms=results['processing_times']['total'],
                hand_processing_time_ms=results['processing_times']['hand'],
                pose_processing_time_ms=results['processing_times']['pose'],
                facemesh_processing_time_ms=results['processing_times']['face'],
                emotion_processing_time_ms=results['processing_times']['emotion'],
                bad_gesture_processing_time_ms=results['processing_times']['bad_gestures'],
                hand_model=self.config['detection'].get('hand_model'),
                pose_model=self.config['detection'].get('pose_model'),
                facemesh_model=self.config['detection'].get('facemesh_model'),
                emotion_model=self.config['detection'].get('emotion_model'),
                transcript_segment_id=transcript_segment_id  # Reference to transcript_video table
            )
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
    
    def _save_transcript_segments_to_db(self):
        """Save transcript segments to transcript_video table"""
        if not self.db or not self.transcript_data:
            return
        
        save_segments = self.config['database'].get('save_transcript_segments', True)
        if not save_segments:
            return
        
        try:
            print("💾 Saving transcript segments to database...")
            
            # Get current video file name
            video_file = os.path.basename(self.video_path_for_transcript) if self.video_path_for_transcript else "unknown"
            
            for i, segment in enumerate(self.transcript_data):
                # Insert segment into transcript_video table
                result = self.db.supabase.table('transcript_video').insert({
                    'video_file': video_file,
                    'session_id': self.session_id,
                    'segment_id': i,
                    'start_time': float(segment['start']),
                    'end_time': float(segment['end']),
                    'duration': float(segment['end'] - segment['start']),
                    'text': segment['text'],
                    'word_count': len(segment['text'].split()),
                    'language': self.config['detection'].get('transcript_settings', {}).get('language', 'en'),
                    'model_size': self.config['detection'].get('transcript_settings', {}).get('model_size', 'tiny'),
                    'chunk_duration': self.config['detection'].get('transcript_settings', {}).get('chunk_duration', 15.0),
                    'words_json': segment.get('words', [])
                }).execute()
                
                # Store segment ID for frame reference
                if result.data and len(result.data) > 0:
                    segment_db_id = result.data[0]['id']
                    # Map time range to database ID
                    self.transcript_segment_map[(segment['start'], segment['end'])] = segment_db_id
                    
            self.transcript_segments_saved = True
            print(f"✅ Saved {len(self.transcript_data)} transcript segments to database")
            
        except Exception as e:
            print(f"❌ Error saving transcript segments: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_transcript_segment_id(self, current_time: float) -> int:
        """Get the database ID of transcript segment for given time"""
        if not self.transcript_segments_saved:
            return None
            
        for (start_time, end_time), segment_id in self.transcript_segment_map.items():
            if start_time <= current_time <= end_time:
                return segment_id
                
        return None
    
    def _combine_video_with_audio(self) -> bool:
        """
        Combine the processed video with original audio using ffmpeg
        
        Returns:
            True if successful, False otherwise
        """
        if not hasattr(self, 'temp_video_path') or not hasattr(self, 'final_video_path'):
            return False
        
        try:
            import subprocess
            
            print(f"🎵 Combining video with original audio using ffmpeg...")
            
            # Build ffmpeg command to combine video and audio
            # -i temp_video: input video (no audio)
            # -i original_video: input audio source
            # -c:v copy: copy video codec (no re-encoding)
            # -c:a aac: encode audio as AAC
            # -map 0:v: use video from first input (temp video)
            # -map 1:a: use audio from second input (original video)
            # -shortest: finish when shortest stream ends
            # -y: overwrite output file
            
            # Handle video alignment timing - build command with proper audio offset
            if hasattr(self, 'start_time_offset') and self.start_time_offset > 0:
                # Apply -ss to audio input only (before second -i parameter)
                cmd = [
                    'ffmpeg',
                    '-i', self.temp_video_path,          # Processed video (no audio)
                    '-ss', str(self.start_time_offset),   # Seek audio source to offset
                    '-i', self.original_video_path,       # Original video (audio source)
                    '-c:v', 'copy',                       # Copy video without re-encoding
                    '-c:a', 'aac',                        # Encode audio as AAC
                    '-map', '0:v',                        # Video from first input
                    '-map', '1:a',                        # Audio from second input
                    '-shortest',                          # Stop when shortest stream ends
                    '-y',                                 # Overwrite output
                    self.final_video_path
                ]
            else:
                # No offset needed - standard command
                cmd = [
                    'ffmpeg',
                    '-i', self.temp_video_path,      # Processed video (no audio)
                    '-i', self.original_video_path,   # Original video (audio source)
                    '-c:v', 'copy',                   # Copy video without re-encoding
                    '-c:a', 'aac',                    # Encode audio as AAC
                    '-map', '0:v',                    # Video from first input
                    '-map', '1:a',                    # Audio from second input
                    '-shortest',                      # Stop when shortest stream ends
                    '-y',                             # Overwrite output
                    self.final_video_path
                ]
            
            print(f"🔧 Running: {' '.join(cmd[:8])}...")  # Show abbreviated command
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully combined video with audio")
                return True
            else:
                print(f"❌ FFmpeg failed with return code {result.returncode}")
                if result.stderr:
                    # Only show first few lines of error to avoid spam
                    error_lines = result.stderr.split('\n')[:5]
                    print(f"❌ Error: {' '.join(error_lines)}")
                return False
                
        except FileNotFoundError:
            print(f"❌ FFmpeg not found. Please install ffmpeg to preserve audio.")
            print(f"💡 On macOS: brew install ffmpeg")
            print(f"💡 On Ubuntu: sudo apt install ffmpeg")
            return False
        except Exception as e:
            print(f"❌ Error combining video with audio: {e}")
            return False
    
    def _generate_report(self) -> str:
        """Generate a comprehensive processing report"""
        if self.stats['total_frames'] == 0:
            return "No frames processed."
        
        # Calculate percentages
        hand_success_rate = (self.stats['successful_hand_detections'] / self.stats['total_frames']) * 100
        pose_success_rate = (self.stats['successful_pose_detections'] / self.stats['total_frames']) * 100
        facemesh_success_rate = (self.stats['successful_facemesh_detections'] / self.stats['total_frames']) * 100
        emotion_success_rate = (self.stats['successful_emotion_detections'] / self.stats['total_frames']) * 100
        transcript_success_rate = (self.stats['successful_transcript_detections'] / self.stats['total_frames']) * 100
        
        # Calculate average processing times
        avg_times = {}
        for key in ['hand', 'pose', 'face', 'emotion', 'bad_gestures', 'transcript', 'total']:
            avg_times[key] = self.stats['total_processing_times'][key] / self.stats['total_frames']
        
        # Get model names
        hand_model = self.config['detection'].get('hand_model', 'none')
        pose_model = self.config['detection'].get('pose_model', 'none')
        facemesh_model = self.config['detection'].get('facemesh_model', 'none')
        emotion_model = self.config['detection'].get('emotion_model', 'none')
        transcript_model = self.config['detection'].get('transcript_model', 'none')
        
        # Generate report content
        report = f"""
========================================
MUSICIAN TRACKING DETECTION REPORT
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.session_id}

BASIC INFORMATION:
- Total frames: {self.stats['total_frames']}
- Start time offset: {self.start_time_offset:.3f}s
- Duration of video: {self.matching_duration:.1f}s
- Skip frames: {self.config['video'].get('skip_frames', 0)}

MODEL CONFIGURATION:
- Hand: {hand_model}
- Pose: {pose_model}  
- Facemesh: {facemesh_model}
- Emotion: {emotion_model}
- Transcript: {transcript_model}
- Bad Gestures: {'enabled' if self.bad_gesture_detector else 'disabled'}

DETECTION SUCCESS RATES:
- Hand detection: {hand_success_rate:.1f}% ({self.stats['successful_hand_detections']}/{self.stats['total_frames']} frames)
- Pose detection: {pose_success_rate:.1f}% ({self.stats['successful_pose_detections']}/{self.stats['total_frames']} frames)
- Facemesh detection: {facemesh_success_rate:.1f}% ({self.stats['successful_facemesh_detections']}/{self.stats['total_frames']} frames)
- Emotion detection: {emotion_success_rate:.1f}% ({self.stats['successful_emotion_detections']}/{self.stats['total_frames']} frames)
- Transcript detection: {transcript_success_rate:.1f}% ({self.stats['successful_transcript_detections']}/{self.stats['total_frames']} frames)

AVERAGE PROCESSING TIMES:
- Hand: {avg_times['hand']:.3f}ms
- Pose: {avg_times['pose']:.3f}ms
- Facemesh: {avg_times['face']:.3f}ms
- Emotion: {avg_times['emotion']:.3f}ms
- Bad Gestures: {avg_times['bad_gestures']:.3f}ms
- Transcript: {avg_times['transcript']:.3f}ms
- Total per frame: {avg_times['total']:.3f}ms

TOTAL PROCESSING TIMES:
- Hand: {self.stats['total_processing_times']['hand']:.0f}ms
- Pose: {self.stats['total_processing_times']['pose']:.0f}ms
- Facemesh: {self.stats['total_processing_times']['face']:.0f}ms
- Emotion: {self.stats['total_processing_times']['emotion']:.0f}ms
- Bad Gestures: {self.stats['total_processing_times']['bad_gestures']:.0f}ms
- Transcript: {self.stats['total_processing_times']['transcript']:.0f}ms
- Total: {self.stats['total_processing_times']['total']:.0f}ms

PERFORMANCE METRICS:
- Frames per second: {self.stats['total_frames'] / (self.stats['total_processing_times']['total'] / 1000):.1f} FPS (processing speed)
- Total processing duration: {self.stats['total_processing_times']['total'] / 1000:.1f} seconds

========================================
"""
        return report
    
    def _save_report(self, report_path: str):
        """Save the generated report to file"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            report_content = self._generate_report()
            with open(report_path, 'w') as f:
                f.write(report_content)
            print(f"✅ Report saved to: {report_path}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
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
        
        # Draw hand landmarks using model's draw_landmarks method
        if self.hand_detector and results.get('hand_raw_results'):
            annotated_frame = self.hand_detector.draw_landmarks(annotated_frame, results['hand_raw_results'])
        
        # Draw pose landmarks using model's draw_landmarks method
        if self.pose_detector and results.get('pose_raw_results'):
            annotated_frame = self.pose_detector.draw_landmarks(annotated_frame, results['pose_raw_results'])
        
        # Draw face landmarks using model's draw_landmarks method
        if self.facemesh_detector and results.get('facemesh_raw_results'):
            if isinstance(self.facemesh_detector, dict) and self.facemesh_detector.get('type') == 'hybrid':
                # For hybrid approach, draw YOLO bounding boxes
                if results['facemesh_raw_results']['yolo']:
                    yolo_detector = self.facemesh_detector['yolo']
                    annotated_frame = yolo_detector.draw_bboxes(annotated_frame, results['facemesh_raw_results']['yolo'], 
                                                              draw_padding=True, padding=30)
                
                # Draw MediaPipe face mesh if available (currently not implemented in hybrid mode)
                # The face mesh landmarks are already processed and stored in results['facemesh_landmarks']
                if results.get('facemesh_landmarks'):
                    for landmark in results['facemesh_landmarks']:
                        x, y = int(landmark['x']), int(landmark['y'])
                        cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
            else:
                # Standard MediaPipe FaceMesh approach
                annotated_frame = self.facemesh_detector.draw_landmarks(annotated_frame, results['facemesh_raw_results'])
        
        # Add comprehensive annotation text
        y_offset = 30
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Skip frames
        skip_frames = self.config['video'].get('skip_frames', 0)
        cv2.putText(annotated_frame, f"Skip_frames: {skip_frames}", 
                   (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        # Start time offset (retrieved from database)
        cv2.putText(annotated_frame, f"Start_time_offset: {self.start_time_offset:.3f}s", 
                   (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        # Matching duration (retrieved from database)
        cv2.putText(annotated_frame, f"Matching_duration: {self.matching_duration:.1f}s", 
                   (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        # Current video position
        if hasattr(self, 'end_time_offset') and self.end_time_offset is not None:
            # Try to get current position from video capture
            try:
                current_position_ms = results.get('current_position_ms', 0)  # We'll add this in process_frame
                if current_position_ms == 0:
                    # Fallback: estimate based on frame count and fps
                    fps = 30.0  # Default fallback
                    # Since video was already seeked, frame 0 corresponds to start_time_offset
                    current_position_s = self.start_time_offset + (results['frame_number'] / fps)
                else:
                    current_position_s = current_position_ms / 1000.0
                
                remaining_time = max(0, self.end_time_offset - current_position_s)
                cv2.putText(annotated_frame, f"Video_position: {current_position_s:.1f}s (remaining: {remaining_time:.1f}s)", 
                           (10, y_offset), font, font_scale, (255, 255, 0), thickness)
                y_offset += line_height
            except:
                pass
        
        # Frame number (keep on left)
        cv2.putText(annotated_frame, f"Frame: {results['frame_number']}", 
                   (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        
        # Move all processing times to right corner
        self._draw_processing_times_right_corner(annotated_frame, results)
        
        # Draw emotion result if available
        if results.get('dominant_emotion'):
            y_offset += line_height
            cv2.putText(annotated_frame, f"Detected emotion: {results['dominant_emotion']}", 
                       (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        # Bad gesture annotations removed (but detection still runs in background)
        
        # Add transcript overlay if available
        if self.transcript_detector:
            start_time = time.time()
            current_position_ms = results.get('current_position_ms', 0)
            
            # Calculate current video time (position already accounts for seeking)
            if current_position_ms > 0:
                current_time_s = current_position_ms / 1000.0
            else:
                # Fallback: estimate based on frame count and fps (assuming 30 fps)
                # Since video was already seeked, frame 0 corresponds to start_time_offset
                current_time_s = self.start_time_offset + (results['frame_number'] / 30.0)
            
            # Show transcript status and data if available
            if self.transcript_data:
                current_transcript = self._get_current_transcript(current_time_s)
                if current_transcript:
                    annotated_frame = self._add_transcript_overlay(annotated_frame, current_transcript, current_time_s)
                else:
                    annotated_frame = self._add_transcript_overlay(annotated_frame, "[No speech detected]", current_time_s)
            else:
                # Show processing status
                annotated_frame = self._add_transcript_overlay(annotated_frame, "🔄 Processing transcript...", current_time_s)
            
            results['processing_times']['transcript'] = float(f"{(time.time() - start_time) * 1000:.3f}")
        else:
            results['processing_times']['transcript'] = 0
        
        # Update total processing time to include transcript time
        current_total = sum(v for k, v in results['processing_times'].items() if k != 'total')
        results['processing_times']['total'] = float(f"{current_total:.3f}")
        
        return annotated_frame
    
    def _draw_processing_times_right_corner(self, frame: np.ndarray, results: Dict[str, Any]):
        """Draw processing times in the top-right corner of the frame"""
        _, width = frame.shape[:2]
        
        # Text styling for right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        margin = 10
        
        # Processing time data
        processing_data = [
            ("Hand", self.config['detection'].get('hand_model', 'none'), results['processing_times'].get('hand', 0), (0, 255, 255)),
            ("Pose", self.config['detection'].get('pose_model', 'none'), results['processing_times'].get('pose', 0), (255, 0, 255)), 
            ("Face", self.config['detection'].get('facemesh_model', 'none'), results['processing_times'].get('face', 0), (0, 255, 0)),
            ("Emotion", self.config['detection'].get('emotion_model', 'none'), results['processing_times'].get('emotion', 0), (255, 128, 0)),
            ("BadGest", "enabled" if self.bad_gesture_detector else "disabled", results['processing_times'].get('bad_gestures', 0), (255, 0, 255)),
            ("Transcript", self.config['detection'].get('transcript_model', 'none'), results['processing_times'].get('transcript', 0), (0, 255, 255)),
            ("TOTAL", "", results['processing_times'].get('total', 0), (0, 0, 255))
        ]
        
        # Draw each processing time line
        y_start = 30
        for i, (label, model, time_ms, color) in enumerate(processing_data):
            y_pos = y_start + i * line_height
            
            if label == "TOTAL":
                text = f"TOTAL: {time_ms}ms"
            else:
                text = f"{label}: {model} - {time_ms}ms"
            
            # Calculate text width for right alignment
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x_pos = width - text_width - margin
            
            # Draw background rectangle for better readability
            cv2.rectangle(frame, 
                         (x_pos - 5, y_pos - text_height - 3),
                         (x_pos + text_width + 5, y_pos + 3),
                         (0, 0, 0), -1)  # Black background
            
            # Draw text
            cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, color, thickness)
    
    def _start_transcript_processing(self, video_path: str):
        """Start transcript processing in background"""
        if not self.transcript_detector:
            return
        
        self.video_path_for_transcript = video_path
        
        try:
            # Process entire video file for transcript in background
            chunk_duration = self.config['detection'].get('transcript_settings', {}).get('chunk_duration', 15.0)
            
            # Process transcript directly in main thread for simplicity and reliability
            try:
                print(f"🎤 Starting transcript processing for: {video_path}")
                print(f"🎤 Using video alignment: start_offset={self.start_time_offset:.3f}s, end_offset={self.end_time_offset}")

                results = self.transcript_detector.transcribe_video_file(
                    video_path,
                    chunk_duration=chunk_duration,
                    start_time_offset=self.start_time_offset,
                    end_time_offset=self.end_time_offset
                )

                # Build transcript timeline from results
                if results:
                    self.transcript_data = self.transcript_detector._build_transcript_timeline(results)
                    print(f"✅ Transcript processing complete: {len(self.transcript_data)} segments")

                    # Save transcript segments to database
                    self._save_transcript_segments_to_db()

                    # Also store individual segment texts for easier access
                    for i, segment in enumerate(self.transcript_data):
                        print(f"🎤 Segment {i+1}: [{segment['start']//60:02.0f}:{segment['start']%60:05.2f}-{segment['end']//60:02.0f}:{segment['end']%60:05.2f}] {segment['text'][:80]}{'...' if len(segment['text']) > 80 else ''}")
                else:
                    print("❌ No transcript results returned from Whisper")
                    self.transcript_data = []

                print("✅ Transcript processing complete, starting video processing")

            except Exception as e:
                print(f"❌ Error in transcript processing: {e}")
                import traceback
                traceback.print_exc()
                self.transcript_data = []
            
        except Exception as e:
            print(f"❌ Error starting transcript processing: {e}")
    
    def _get_current_transcript(self, current_time_s: float) -> str:
        """Get transcript text that should be displayed at current time"""
        current_transcripts = []

        for transcript in self.transcript_data:
            if transcript['start'] <= current_time_s <= transcript['end']:
                current_transcripts.append(transcript['text'])


        return " ".join(current_transcripts) if current_transcripts else ""
    
    def _add_transcript_overlay(self, frame: np.ndarray, transcript_text: str, current_time_s: float) -> np.ndarray:
        """Add transcript overlay to frame - similar to test_transcript.py style"""
        if not transcript_text:
            return frame
        
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay area at bottom (similar to test_transcript.py)
        overlay_height = int(height * 0.25)  # Use bottom 25% of frame
        overlay_y = height - overlay_height
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, overlay_y), (width, height), (0, 0, 0), -1)
        
        # Blend overlay with original frame
        alpha = 0.7  # Transparency (same as test_transcript.py)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Format timestamp (similar to test_transcript.py)
        mins = int(current_time_s // 60)
        secs = int(current_time_s % 60)
        time_str = f"[{mins:02d}:{secs:02d}]"
        
        # Text styling (matching test_transcript.py)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)  # White text
        timestamp_color = (100, 255, 100)  # Light green for timestamp
        thickness = 2
        line_height = 30
        
        # Add timestamp (matching test_transcript.py style)
        cv2.putText(frame, time_str, (20, overlay_y + 30), font, font_scale * 0.8, timestamp_color, thickness - 1)
        
        # Prepare text and word wrap (using same logic as test_transcript.py)
        words = transcript_text.split()
        lines = []
        current_line = ""
        max_width = width - 40  # Leave margin
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width > max_width:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, add it anyway
                    lines.append(word)
                    current_line = ""
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # Add transcript lines (similar to test_transcript.py)
        y_offset = overlay_y + 60
        for line in lines:
            if y_offset < height - 10:
                cv2.putText(frame, line, (20, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
        
        return frame
    
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Processing: {video_file}")
        print(f"🎬 FPS: {fps}, Resolution: {width}x{height}")
        
        skip_frames = self.config['video'].get('skip_frames', 0)
        display_output = self.config['video'].get('display_output', True)
        save_output_video = self.config['video'].get('save_output_video', False)
        generate_report = self.config['video'].get('generate_report', False)
        
        print(f"⚡ Skip frames: {skip_frames}")
        print(f"👁️ Display output: {display_output}")
        print(f"💾 Save output video: {save_output_video}")
        print(f"📊 Generate report: {generate_report}")
        
        # Initialize video writer if output video is enabled
        if save_output_video:
            preserve_audio = self.config['video'].get('preserve_audio', True)
            output_video_path = self.config['video'].get('output_video_path', 'output/annotated_video.mp4')
            
            if preserve_audio and video_path and video_path != '0':
                # Use temporary path for video without audio, will combine later
                temp_video_path = self.config['video'].get('temp_video_path', 'output/temp_video_no_audio.mp4')
                os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                
                if self.video_writer.isOpened():
                    print(f"✅ Temporary video (no audio) will be saved to: {temp_video_path}")
                    print(f"✅ Final video with audio will be saved to: {output_video_path}")
                    self.temp_video_path = temp_video_path
                    self.final_video_path = output_video_path
                    self.original_video_path = video_path
                    self.preserve_audio = True
                else:
                    print(f"❌ Failed to initialize video writer")
                    self.video_writer = None
                    self.preserve_audio = False
            else:
                # No audio preservation needed (webcam or disabled)
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                if self.video_writer.isOpened():
                    print(f"✅ Output video will be saved to: {output_video_path}")
                else:
                    print(f"❌ Failed to initialize video writer")
                    self.video_writer = None
                self.preserve_audio = False
        
        # Retrieve alignment data from database before starting
        self.start_time_offset, self.matching_duration = self._get_alignment_data(video_file)
        print(f"⏰ Start time offset: {self.start_time_offset:.3f}s")
        print(f"⏱️ Matching duration: {self.matching_duration:.1f}s")
        
        # Start transcript processing if enabled
        if self.transcript_detector:
            print("🎤 Starting transcript processing...")
            # Use the actual video path being processed
            transcript_video_path = video_path if video_path else self.config['video'].get('source_path', '')
            self._start_transcript_processing(transcript_video_path)
            print(f"🎤 Transcript will be synced with video alignment: offset={self.start_time_offset:.3f}s, duration={self.matching_duration:.1f}s")
        
        # Seek to start_time_offset position if we have alignment data
        if self.start_time_offset > 0:
            seek_position_ms = self.start_time_offset * 1000  # Convert seconds to milliseconds
            success = cap.set(cv2.CAP_PROP_POS_MSEC, seek_position_ms)
            if success:
                print(f"✅ Seeked to position: {self.start_time_offset:.3f}s")
            else:
                print(f"⚠️ Failed to seek to position: {self.start_time_offset:.3f}s")
        
        # Calculate end time for processing (start_time_offset + matching_duration)
        if self.matching_duration > 0:
            self.end_time_offset = self.start_time_offset + self.matching_duration
            print(f"🏁 Processing will end at: {self.end_time_offset:.1f}s")
        else:
            self.end_time_offset = None  # Process entire video
        
        print("Press 'q' to quit, 'p' to pause/resume")
        
        # Set start time for elapsed time calculation
        self.start_time = time.time()
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we've reached the end time based on matching_duration
                if self.end_time_offset is not None:
                    current_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    current_time_s = current_position_ms / 1000.0
                    if current_time_s >= self.end_time_offset:
                        print(f"🏁 Reached end of matching duration at {current_time_s:.1f}s")
                        break
                
                # Skip frames if configured
                if skip_frames > 0 and self.frame_count % (skip_frames + 1) != 0:
                    self.frame_count += 1
                    continue
                
                # Get current video position
                current_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                # Process frame
                results = self.process_frame(frame)
                results['current_position_ms'] = current_position_ms
                
                # Update statistics for report
                self._update_statistics(results)
                
                # Save to database
                self.save_to_database(results, video_file)
                
                # Draw results on frame
                annotated_frame = self.draw_results(frame, results)
                
                # Save to output video if enabled
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(annotated_frame)
                
                # Display output if enabled
                if display_output:
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
        if self.facemesh_detector:
            if isinstance(self.facemesh_detector, dict) and self.facemesh_detector.get('type') == 'hybrid':
                self.facemesh_detector['yolo'].cleanup()
                self.facemesh_detector['mediapipe'].cleanup()
            else:
                self.facemesh_detector.cleanup()
        if self.emotion_detector:
            self.emotion_detector.cleanup()
        if self.bad_gesture_detector:
            self.bad_gesture_detector.cleanup()
        if self.transcript_detector:
            self.transcript_detector.stop_transcription()
        
        # Close video writer and combine with audio if needed
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            
            # Combine video with original audio if audio preservation is enabled
            if hasattr(self, 'preserve_audio') and self.preserve_audio:
                success = self._combine_video_with_audio()
                if success:
                    print(f"✅ Output video with audio saved")
                    # Clean up temporary video file
                    try:
                        os.remove(self.temp_video_path)
                        print(f"🧹 Cleaned up temporary video file")
                    except Exception as e:
                        print(f"⚠️ Could not remove temporary file: {e}")
                else:
                    print(f"⚠️ Audio combination failed, temporary video available at: {self.temp_video_path}")
            else:
                print(f"✅ Output video saved")
        
        # Generate and save report if enabled
        if self.config['video'].get('generate_report', False):
            report_path = self.config['video'].get('report_path', 'output/detection_report.txt')
            self._save_report(report_path)
        
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