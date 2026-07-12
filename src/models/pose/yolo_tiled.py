"""
Tiled multi-person YOLO pose detector.

Differences vs YOLOPoseDetector (src/models/pose/yolo.py):
- largest pose model by default (yolo11x-pose) instead of nano
- hybrid full-frame + overlapping-tile inference (small/distant people found)
- 360 seam handling (person straddling the equirectangular seam kept whole)
- returns ALL detected people, largest first, with real per-joint confidence;
  convert_to_dict() keeps the old single-person format for compatibility
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any

from .base_pose_detector import BasePoseDetector
from ..tiled_inference import TiledYOLO

# people are persistent in a closed-room 360 recording; distinct stable colors
PERSON_COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0)]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
]

# tried in order; first loadable wins (yolo26 needs a newer ultralytics,
# falls through to yolo11x-pose automatically on older installs)
DEFAULT_MODEL_CANDIDATES = ["yolo26x-pose.pt", "yolo11x-pose.pt"]


class YOLOTiledPoseDetector(BasePoseDetector):
    """Multi-person tiled pose detection for high-resolution / 360 footage."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence: float = 0.4,
                 tile_target: int = 1200,
                 overlap: float = 0.25,
                 wrap_360: bool = True,
                 max_people: int = 0,
                 min_joint_conf: float = 0.3):
        super().__init__(confidence=confidence)
        self.max_people = max_people  # 0 = keep everyone
        self.min_joint_conf = min_joint_conf
        self.engine = None

        candidates = [model_path] if model_path else DEFAULT_MODEL_CANDIDATES
        for cand in candidates:
            try:
                self.engine = TiledYOLO(
                    cand, confidence=confidence, tile_target=tile_target,
                    overlap=overlap, wrap_360=wrap_360)
                self.model_path = cand
                print(f"✅ Tiled YOLO pose model loaded: {cand} "
                      f"(device={self.engine.device}, wrap_360={wrap_360})")
                break
            except Exception as e:
                print(f"⚠️ Could not load {cand}: {e}")
        if self.engine is None:
            print("❌ Failed to load any tiled pose model")

        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    # ---- detection ----

    def detect(self, frame: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """Returns merged people list (largest first), or None."""
        if self.engine is None:
            return None
        try:
            people = self.engine.predict(frame)
            if self.max_people > 0:
                people = people[:self.max_people]
            return people or None
        except Exception as e:
            print(f"❌ Tiled YOLO pose detection failed: {e}")
            return None

    def get_all_people(self, results: Any) -> List[Dict[str, Any]]:
        """Full multi-person output: bbox, conf, keypoints, crosses_seam."""
        return results or []

    # ---- compatibility with the single-person pipeline ----

    def convert_to_dict(self, results: Any) -> Optional[List[Dict]]:
        """Old-format landmarks for the primary (largest) person."""
        if not results:
            return None
        kp = results[0].get('keypoints')
        if not kp:
            return None
        return [{
            "x": float(x), "y": float(y), "z": 0.0,
            "confidence": float(c),
        } for x, y, c in kp]

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        if not results:
            return frame
        for idx, person in enumerate(results):
            color = PERSON_COLORS[idx % len(PERSON_COLORS)]
            x1, y1, x2, y2 = [int(v) for v in person['bbox']]
            w = frame.shape[1]
            if person.get('crosses_seam'):
                cv2.rectangle(frame, (x1, y1), (w - 1, y2), color, 2)
                cv2.rectangle(frame, (0, y1), (x2 % w, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"P{idx + 1} {person['conf']:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            kp = person.get('keypoints')
            if not kp:
                continue
            pts = np.asarray(kp)
            for x, y, c in pts:
                if c >= self.min_joint_conf:
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            for a, b in SKELETON:
                if (pts[a][2] >= self.min_joint_conf and
                        pts[b][2] >= self.min_joint_conf):
                    seg_w = abs(pts[a][0] - pts[b][0])
                    if seg_w < frame.shape[1] / 2:  # don't draw across the seam
                        cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                                 (int(pts[b][0]), int(pts[b][1])), color, 2)
        return frame

    def get_landmark_by_name(self, results: Any, landmark_name: str,
                             person_index: int = 0) -> Optional[Dict]:
        if not results or person_index >= len(results):
            return None
        name = landmark_name.lower()
        if name not in self.keypoint_names:
            return None
        kp = results[person_index].get('keypoints')
        if not kp:
            return None
        x, y, c = kp[self.keypoint_names.index(name)]
        return {"x": float(x), "y": float(y), "z": 0.0, "confidence": float(c)}

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Tiled YOLO Pose Detector",
            "type": "pose",
            "model_path": getattr(self, 'model_path', None),
            "confidence_threshold": self.confidence,
            "multi_person": True,
            "max_people": self.max_people,
            "wrap_360": self.engine.wrap_360 if self.engine else None,
            "device": self.engine.device if self.engine else None,
            "num_landmarks": len(self.keypoint_names),
            "available": self.engine is not None,
            "keypoint_names": self.keypoint_names,
        }

    def cleanup(self):
        pass
