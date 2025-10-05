import unittest
import cv2
import numpy as np
import sys
import os
import json

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pose.mediapipe import MediaPipePoseDetector
from src.models.pose.yolo import YOLOPoseDetector


class TestPoseDetection(unittest.TestCase):
    """Test cases for pose detection models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.test_image_path = "test/images/test_pose.jpg"
        cls.ground_truth_path = "test/images/test_pose_ground_truth.json"
        
        # Create a simple test image if it doesn't exist
        if not os.path.exists(cls.test_image_path):
            cls._create_test_image()
        
        # Load test image
        cls.test_image = cv2.imread(cls.test_image_path)
        if cls.test_image is None:
            raise FileNotFoundError(f"Could not load test image: {cls.test_image_path}")
    
    @classmethod
    def _create_test_image(cls):
        """Create a simple test image for pose detection"""
        # Create a simple colored image (this would be replaced with actual person image)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add simple shapes to simulate a person
        # Head
        cv2.circle(test_image, (320, 100), 40, (200, 150, 100), -1)
        # Body
        cv2.rectangle(test_image, (280, 140), (360, 300), (150, 100, 50), -1)
        # Arms
        cv2.rectangle(test_image, (200, 160), (280, 180), (100, 75, 25), -1)
        cv2.rectangle(test_image, (360, 160), (440, 180), (100, 75, 25), -1)
        # Legs
        cv2.rectangle(test_image, (300, 300), (320, 450), (75, 50, 25), -1)
        cv2.rectangle(test_image, (320, 300), (340, 450), (75, 50, 25), -1)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cls.test_image_path), exist_ok=True)
        cv2.imwrite(cls.test_image_path, test_image)
        print(f"Created test image: {cls.test_image_path}")
    
    def test_mediapipe_pose_detection(self):
        """Test MediaPipe pose detection"""
        print("\n=== Testing MediaPipe Pose Detection ===")
        
        detector = MediaPipePoseDetector()
        
        # Test detection
        results = detector.detect(self.test_image)
        self.assertIsNotNone(detector.model, "MediaPipe pose model should be initialized")
        
        # Test conversion to dict
        pose_landmarks = detector.convert_to_dict(results)
        print(f"Pose landmarks detected: {pose_landmarks is not None}")
        if pose_landmarks:
            print(f"Number of landmarks: {len(pose_landmarks)}")
        
        # Test drawing (should not raise exception)
        output_frame = detector.draw_landmarks(self.test_image.copy(), results)
        self.assertEqual(output_frame.shape, self.test_image.shape, "Output frame should have same shape as input")
        
        # Test specific landmark retrieval
        left_wrist = detector.get_landmark_by_name(results, "LEFT_WRIST")
        right_shoulder = detector.get_landmark_by_name(results, "RIGHT_SHOULDER")
        print(f"Left wrist detected: {left_wrist is not None}")
        print(f"Right shoulder detected: {right_shoulder is not None}")
        
        # Test model info
        model_info = detector.get_model_info()
        self.assertEqual(model_info["type"], "pose")
        self.assertEqual(model_info["name"], "MediaPipe Pose")
        self.assertEqual(model_info["num_landmarks"], 33)
        
        # Save results as ground truth
        ground_truth = {
            "model": "mediapipe",
            "pose_landmarks": pose_landmarks,
            "left_wrist": left_wrist,
            "right_shoulder": right_shoulder,
            "model_info": model_info
        }
        
        self._save_ground_truth("mediapipe", ground_truth)
        
        # Cleanup
        detector.cleanup()
    
    def test_yolo_pose_detection(self):
        """Test YOLO pose detection"""
        print("\n=== Testing YOLO Pose Detection ===")
        
        detector = YOLOPoseDetector()
        
        # Test detection (might fail if model doesn't exist, which is expected)
        results = detector.detect(self.test_image)
        
        if detector.model is None:
            print("YOLO pose model not available - this is expected if model file doesn't exist")
            self.skipTest("YOLO pose model not available")
        
        # Test conversion to dict
        pose_landmarks = detector.convert_to_dict(results)
        print(f"Pose landmarks detected: {pose_landmarks is not None}")
        if pose_landmarks:
            print(f"Number of landmarks: {len(pose_landmarks)}")
        
        # Test drawing (should not raise exception)
        output_frame = detector.draw_landmarks(self.test_image.copy(), results)
        self.assertEqual(output_frame.shape, self.test_image.shape, "Output frame should have same shape as input")
        
        # Test specific landmark retrieval
        left_wrist = detector.get_landmark_by_name(results, "left_wrist")
        right_shoulder = detector.get_landmark_by_name(results, "right_shoulder")
        print(f"Left wrist detected: {left_wrist is not None}")
        print(f"Right shoulder detected: {right_shoulder is not None}")
        
        # Test model info
        model_info = detector.get_model_info()
        self.assertEqual(model_info["type"], "pose")
        self.assertEqual(model_info["name"], "YOLO Pose Detector")
        self.assertEqual(model_info["num_landmarks"], 17)
        
        # Save results as ground truth
        ground_truth = {
            "model": "yolo",
            "pose_landmarks": pose_landmarks,
            "left_wrist": left_wrist,
            "right_shoulder": right_shoulder,
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
            
            # Test MediaPipe consistency
            if "mediapipe" in ground_truth:
                detector = MediaPipePoseDetector()
                results = detector.detect(self.test_image)
                pose_landmarks = detector.convert_to_dict(results)
                
                gt_landmarks = ground_truth["mediapipe"]["pose_landmarks"]
                
                # Compare presence of pose landmarks (basic consistency check)
                self.assertEqual((pose_landmarks is not None), (gt_landmarks is not None), 
                               "Pose landmark detection should be consistent")
                
                if pose_landmarks and gt_landmarks:
                    self.assertEqual(len(pose_landmarks), len(gt_landmarks), 
                                   "Number of landmarks should be consistent")
                
                detector.cleanup()
                print("MediaPipe consistency check passed")
            
        except Exception as e:
            self.fail(f"Ground truth comparison failed: {e}")


def generate_ground_truth():
    """Generate ground truth data by running models on test image"""
    print("=== Generating Ground Truth for Pose Detection ===")
    
    # Create test suite and run tests to generate ground truth
    suite = unittest.TestSuite()
    suite.addTest(TestPoseDetection('test_mediapipe_pose_detection'))
    suite.addTest(TestPoseDetection('test_yolo_pose_detection'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test pose detection models')
    parser.add_argument('--generate-ground-truth', action='store_true',
                       help='Generate ground truth data')
    
    args = parser.parse_args()
    
    if args.generate_ground_truth:
        generate_ground_truth()
    else:
        unittest.main(verbosity=2)