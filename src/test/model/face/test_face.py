import unittest
import cv2
import numpy as np
import sys
import os
import json

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.facemesh.mediapipe import MediaPipeFaceMeshDetector
from src.models.face.yolo import YOLOFaceDetector


class TestFaceDetection(unittest.TestCase):
    """Test cases for face detection models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.test_image_path = "test/images/test_face.jpg"
        cls.ground_truth_path = "test/images/test_face_ground_truth.json"
        
        # Create a simple test image if it doesn't exist
        if not os.path.exists(cls.test_image_path):
            cls._create_test_image()
        
        # Load test image
        cls.test_image = cv2.imread(cls.test_image_path)
        if cls.test_image is None:
            raise FileNotFoundError(f"Could not load test image: {cls.test_image_path}")
    
    @classmethod
    def _create_test_image(cls):
        """Create a simple test image for face detection"""
        # Create a simple colored image (this would be replaced with actual face image)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add simple face-like shape
        # Face outline
        cv2.ellipse(test_image, (320, 240), (80, 100), 0, 0, 360, (200, 180, 160), -1)
        # Eyes
        cv2.circle(test_image, (290, 220), 10, (50, 50, 50), -1)
        cv2.circle(test_image, (350, 220), 10, (50, 50, 50), -1)
        # Nose
        cv2.circle(test_image, (320, 240), 5, (150, 120, 100), -1)
        # Mouth
        cv2.ellipse(test_image, (320, 270), (20, 10), 0, 0, 180, (100, 50, 50), -1)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cls.test_image_path), exist_ok=True)
        cv2.imwrite(cls.test_image_path, test_image)
        print(f"Created test image: {cls.test_image_path}")
    
    def test_mediapipe_face_detection(self):
        """Test MediaPipe face detection"""
        print("\n=== Testing MediaPipe Face Detection ===")
        
        detector = MediaPipeFaceMeshDetector()
        
        # Test detection (might fail if model file doesn't exist)
        results = detector.detect(self.test_image)
        
        if detector.model is None:
            print("MediaPipe face model not available - this is expected if model file doesn't exist")
            self.skipTest("MediaPipe face model not available")
        
        # Test conversion to dict
        face_landmarks = detector.convert_to_dict(results)
        print(f"Face landmarks detected: {face_landmarks is not None}")
        if face_landmarks:
            print(f"Number of landmarks: {len(face_landmarks)}")
        
        # Test drawing (should not raise exception)
        output_frame = detector.draw_landmarks(self.test_image.copy(), results)
        self.assertEqual(output_frame.shape, self.test_image.shape, "Output frame should have same shape as input")
        
        # Test bounding box extraction
        face_bbox = detector.get_face_bbox(results)
        print(f"Face bounding box: {face_bbox}")
        
        # Test model info
        model_info = detector.get_model_info()
        self.assertEqual(model_info["type"], "facemesh")
        self.assertEqual(model_info["name"], "MediaPipe FaceLandmarker")
        self.assertEqual(model_info["num_landmarks"], 468)
        
        # Save results as ground truth
        ground_truth = {
            "model": "mediapipe",
            "face_landmarks": face_landmarks,
            "face_bbox": face_bbox,
            "model_info": model_info
        }
        
        self._save_ground_truth("mediapipe", ground_truth)
        
        # Cleanup
        detector.cleanup()
    
    def test_yolo_face_detection(self):
        """Test YOLO face detection"""
        print("\n=== Testing YOLO Face Detection ===")
        
        detector = YOLOFaceDetector()
        
        # Test detection (might fail if model doesn't exist, which is expected)
        results = detector.detect(self.test_image)
        
        if detector.model is None:
            print("YOLO face model not available - this is expected if model file doesn't exist")
            self.skipTest("YOLO face model not available")
        
        # Test bounding box extraction (new YOLO method)
        face_bboxes = detector.extract_bboxes(results, self.test_image.shape) if results else []
        print(f"Face bounding boxes detected: {len(face_bboxes)}")
        if face_bboxes:
            print(f"First bbox: {face_bboxes[0]}")
        
        # Test drawing (should not raise exception)
        output_frame = detector.draw_bboxes(self.test_image.copy(), results) if results else self.test_image.copy()
        self.assertEqual(output_frame.shape, self.test_image.shape, "Output frame should have same shape as input")
        
        # Test padded bounding box extraction
        padded_bboxes = detector.extract_bboxes_with_pad(results, self.test_image.shape) if results else []
        print(f"Padded bounding boxes: {len(padded_bboxes)}")
        
        # Test model info
        model_info = detector.get_model_info()
        self.assertEqual(model_info["type"], "face_detection")
        self.assertEqual(model_info["name"], "YOLO Face Detector")
        
        # Save results as ground truth
        ground_truth = {
            "model": "yolo",
            "face_bboxes": face_bboxes,
            "padded_bboxes": padded_bboxes,
            "model_info": model_info
        }
        
        self._save_ground_truth("yolo", ground_truth)
        
        # Cleanup
        detector.cleanup()
    
    def _save_ground_truth(self, model_name, ground_truth):
        """Save ground truth results to file"""
        try:
            # Load existing ground truth if it exists
            if os.path.exists(self.ground_truth_path):
                with open(self.ground_truth_path, 'r') as f:
                    all_ground_truth = json.load(f)
            else:
                all_ground_truth = {}
            
            # Update with new results
            all_ground_truth[model_name] = ground_truth
            
            # Save updated ground truth
            with open(self.ground_truth_path, 'w') as f:
                json.dump(all_ground_truth, f, indent=2)
            
            print(f"Ground truth saved for {model_name}")
            
        except Exception as e:
            print(f"Failed to save ground truth: {e}")
    
    def test_compare_with_ground_truth(self):
        """Compare current results with saved ground truth"""
        if not os.path.exists(self.ground_truth_path):
            self.skipTest("No ground truth file available")
        
        print("\n=== Comparing with Ground Truth ===")
        
        try:
            with open(self.ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
            
            # Test MediaPipe consistency (if available)
            if "mediapipe" in ground_truth:
                detector = MediaPipeFaceMeshDetector()
                if detector.model is not None:
                    results = detector.detect(self.test_image)
                    face_landmarks = detector.convert_to_dict(results)
                    
                    gt_landmarks = ground_truth["mediapipe"]["face_landmarks"]
                    
                    # Compare presence of face landmarks (basic consistency check)
                    self.assertEqual((face_landmarks is not None), (gt_landmarks is not None), 
                                   "Face landmark detection should be consistent")
                    
                    if face_landmarks and gt_landmarks:
                        self.assertEqual(len(face_landmarks), len(gt_landmarks), 
                                       "Number of landmarks should be consistent")
                    
                    detector.cleanup()
                    print("MediaPipe consistency check passed")
                else:
                    print("MediaPipe model not available for consistency check")
            
        except Exception as e:
            self.fail(f"Ground truth comparison failed: {e}")


def generate_ground_truth():
    """Generate ground truth data by running models on test image"""
    print("=== Generating Ground Truth for Face Detection ===")
    
    # Create test suite and run tests to generate ground truth
    suite = unittest.TestSuite()
    suite.addTest(TestFaceDetection('test_mediapipe_face_detection'))
    suite.addTest(TestFaceDetection('test_yolo_face_detection'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test face detection models')
    parser.add_argument('--generate-ground-truth', action='store_true',
                       help='Generate ground truth data')
    
    args = parser.parse_args()
    
    if args.generate_ground_truth:
        generate_ground_truth()
    else:
        unittest.main(verbosity=2)