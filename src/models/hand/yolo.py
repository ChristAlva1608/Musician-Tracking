from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from .base_hand_detector import BaseHandDetector


class YOLOHandDetector(BaseHandDetector):
    """YOLO hand detection model"""
    
    def __init__(self, model_path: str = "src/checkpoints/yolo11n-hand.pt", confidence: float = 0.5):
        super().__init__(confidence=confidence)
        self.model_path = model_path
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO hand model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO hand model: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Detect hands in the frame using YOLO
        
        Args:
            frame: Input image frame
            
        Returns:
            YOLO detection results or None
        """
        if self.model is None:
            return None
            
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            return results
        except Exception as e:
            print(f"❌ YOLO hand detection failed: {e}")
            return None
    
    def convert_to_dict(self, results: Any) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Convert YOLO hand detection results to dictionary format
        
        Args:
            results: YOLO detection results
            
        Returns:
            Tuple of (left_hand_landmarks, right_hand_landmarks)
        """
        if not results or len(results) == 0:
            return None, None
        
        left_hand = None
        right_hand = None
        
        try:
            for result in results:
                # Check for keypoints/landmarks first (if the model supports them)
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints
                    print(f"🔍 Found keypoints for {len(keypoints.data)} objects")
                    
                    for obj_idx, obj_keypoints in enumerate(keypoints.data):
                        # obj_keypoints shape should be [num_keypoints, 3] (x, y, visibility)
                        kpts = obj_keypoints.cpu().numpy()
                        print(f"   Object {obj_idx}: {len(kpts)} keypoints")
                        
                        # Convert keypoints to list of dictionaries
                        landmarks = []
                        for kpt_idx, (x, y, vis) in enumerate(kpts):
                            if vis > 0:  # Only include visible keypoints
                                landmarks.append({
                                    "x": float(x),
                                    "y": float(y), 
                                    "z": 0.0,  # YOLO doesn't provide depth
                                    "visibility": float(vis),
                                    "landmark_id": kpt_idx
                                })
                        
                        print(f"   Converted {len(landmarks)} visible landmarks")
                        
                        # Get class information to determine left/right hand
                        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > obj_idx:
                            box = result.boxes[obj_idx]
                            if hasattr(box, 'cls') and box.cls is not None:
                                class_id = int(box.cls[0].cpu().numpy())
                                class_name = getattr(self.model, 'names', {}).get(class_id, f"class_{class_id}")
                                print(f"   Class: {class_name} (ID: {class_id})")
                                
                                # Assign based on class name or position
                                if 'left' in class_name.lower() or (obj_idx == 0 and left_hand is None):
                                    left_hand = landmarks
                                    print(f"   Assigned to left hand")
                                elif 'right' in class_name.lower() or (obj_idx == 1 and right_hand is None):
                                    right_hand = landmarks
                                    print(f"   Assigned to right hand")
                                else:
                                    # Default assignment if no specific class info
                                    if left_hand is None:
                                        left_hand = landmarks
                                        print(f"   Default assigned to left hand")
                                    elif right_hand is None:
                                        right_hand = landmarks
                                        print(f"   Default assigned to right hand")
                            else:
                                # No class info, assign based on order
                                if obj_idx == 0:
                                    left_hand = landmarks
                                    print(f"   Assigned to left hand (by order)")
                                elif obj_idx == 1:
                                    right_hand = landmarks
                                    print(f"   Assigned to right hand (by order)")
                        else:
                            # No box info, assign based on order
                            if obj_idx == 0:
                                left_hand = landmarks
                                print(f"   Assigned to left hand (no box info)")
                            elif obj_idx == 1:
                                right_hand = landmarks
                                print(f"   Assigned to right hand (no box info)")
                
                # Fallback to bounding boxes if no keypoints available
                elif result.boxes is not None and len(result.boxes) > 0:
                    print(f"🔍 No keypoints found, using bounding boxes ({len(result.boxes)} boxes)")
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Create simplified hand data with bounding box center as single landmark
                        hand_data = [{
                            "x": float(center_x),
                            "y": float(center_y),
                            "z": 0.0,  # YOLO doesn't provide depth
                            "visibility": float(conf),
                            "landmark_id": 0  # Center point
                        }]
                        
                        # Get class information if available
                        class_name = "hand"
                        if hasattr(result.boxes[i], 'cls') and result.boxes[i].cls is not None:
                            class_id = int(result.boxes[i].cls[0].cpu().numpy())
                            class_name = getattr(self.model, 'names', {}).get(class_id, f"class_{class_id}")
                        
                        print(f"   Box {i}: {class_name} at ({center_x:.1f}, {center_y:.1f}) conf={conf:.3f}")
                        
                        # Assign to left or right hand based on class name or detection order
                        if 'left' in class_name.lower() or (i == 0 and left_hand is None):
                            left_hand = hand_data
                            print(f"   Assigned to left hand")
                        elif 'right' in class_name.lower() or (i == 1 and right_hand is None):
                            right_hand = hand_data
                            print(f"   Assigned to right hand")
                        else:
                            # Default assignment
                            if left_hand is None:
                                left_hand = hand_data
                                print(f"   Default assigned to left hand")
                            elif right_hand is None:
                                right_hand = hand_data
                                print(f"   Default assigned to right hand")
                        
        except Exception as e:
            print(f"❌ Error converting YOLO hand results: {e}")
            return None, None
        
        print(f"🤚 Final result - Left: {'Yes' if left_hand else 'None'}, Right: {'Yes' if right_hand else 'None'}")
        return left_hand, right_hand
    
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw hand detection results on the frame
        
        Args:
            frame: Input image frame
            results: YOLO detection results
            
        Returns:
            Frame with drawn detections
        """
        if not results or len(results) == 0:
            return frame
        
        try:
            for result in results:
                # Draw keypoints if available
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    for obj_keypoints in result.keypoints.data:
                        kpts = obj_keypoints.cpu().numpy()
                        
                        # Draw each keypoint
                        for kpt_idx, (x, y, vis) in enumerate(kpts):
                            if vis > 0:  # Only draw visible keypoints
                                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                                # Draw keypoint number
                                cv2.putText(frame, str(kpt_idx), (int(x), int(y-5)), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw bounding boxes
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw confidence score
                        label = f"Hand: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"❌ Error drawing YOLO hand results: {e}")
        
        return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "YOLO Hand Detector",
            "type": "hand",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence,
            "available": self.model is not None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # YOLO models don't need explicit cleanup
        pass