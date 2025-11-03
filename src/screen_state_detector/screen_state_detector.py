"""
Multicam Screen Change Detector

Detects layout changes and camera state transitions in multicam video recordings.
Identifies grid vs fullscreen layouts and tracks which quadrants are active.
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import argparse
from pathlib import Path


@dataclass
class StateFingerprint:
    """Represents the state of the video screen at a given frame."""
    layout: str  # 'grid' or 'fullscreen'
    quadrants: List[int]  # 1 for active, 0 for inactive
    fingerprint: str  # Unique identifier for this state

    def to_dict(self):
        return asdict(self)


@dataclass
class StateChange:
    """Represents a detected change in screen state."""
    time: float  # Time in seconds
    frame_number: int
    from_state: StateFingerprint
    to_state: StateFingerprint

    def to_dict(self):
        return {
            'time': self.time,
            'frame_number': self.frame_number,
            'from_state': self.from_state.to_dict(),
            'to_state': self.to_state.to_dict()
        }


class ScreenStateDetector:
    """
    Detects changes in multicam screen layout and camera states.

    Algorithm:
    1. Layout Detection: Analyzes center dividers to distinguish grid (2x2) vs fullscreen
    2. Activity Detection: Checks each quadrant's brightness and edge density
    3. Change Detection: Compares state fingerprints with stability threshold
    4. Motion Filtering: Ignores content movement, only tracks structural changes
    """

    # Configuration constants
    STABILITY_THRESHOLD = 5  # Frames needed to confirm a change
    BRIGHTNESS_THRESHOLD = 30  # Threshold for black screen detection
    EDGE_THRESHOLD = 0.1  # Threshold for edge density
    EDGE_DIFF_THRESHOLD = 20  # Threshold for detecting edges in layout
    EDGE_CONTENT_THRESHOLD = 30  # Threshold for content variation

    def __init__(self, video_path: str):
        """
        Initialize the detector with a video file.

        Args:
            video_path: Path to the video file to analyze
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # State tracking
        self.changes: List[StateChange] = []
        self.current_state: Optional[StateFingerprint] = None
        self.previous_state: Optional[StateFingerprint] = None
        self.stable_state_fingerprint: Optional[str] = None
        self.stable_frames: int = 0
        self.frame_count: int = 0

    def detect_layout(self, frame: np.ndarray) -> str:
        """
        Detect if the screen is in grid or fullscreen mode.

        Analyzes the center vertical and horizontal lines for strong edges
        that would indicate dividers in a grid layout.

        Args:
            frame: BGR image frame

        Returns:
            'grid' or 'fullscreen'
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        center_x = width // 2
        center_y = height // 2

        # Sample vertical center line
        vertical_edges = 0
        y_start = int(height * 0.3)
        y_end = int(height * 0.7)

        for y in range(y_start, y_end):
            # Compare brightness at center with neighbors
            center_val = int(gray[y, center_x])
            left_val = int(gray[y, max(0, center_x - 1)])
            right_val = int(gray[y, min(width - 1, center_x + 1)])

            diff = abs(center_val - left_val)
            if diff > self.EDGE_DIFF_THRESHOLD:
                vertical_edges += 1

        # Sample horizontal center line
        horizontal_edges = 0
        x_start = int(width * 0.3)
        x_end = int(width * 0.7)

        for x in range(x_start, x_end):
            center_val = int(gray[center_y, x])
            top_val = int(gray[max(0, center_y - 1), x])
            bottom_val = int(gray[min(height - 1, center_y + 1), x])

            diff = abs(center_val - top_val)
            if diff > self.EDGE_DIFF_THRESHOLD:
                horizontal_edges += 1

        # Calculate edge ratios
        vertical_edge_ratio = vertical_edges / (y_end - y_start)
        horizontal_edge_ratio = horizontal_edges / (x_end - x_start)

        # If both dividers are strong, it's grid mode
        if vertical_edge_ratio > 0.3 and horizontal_edge_ratio > 0.3:
            return 'grid'
        return 'fullscreen'

    def is_quadrant_active(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """
        Check if a quadrant has active video content.

        A quadrant is considered active if it has reasonable brightness
        OR sufficient edge density (content variation).

        Args:
            frame: BGR image frame
            x, y: Top-left coordinates of quadrant
            w, h: Width and height of quadrant

        Returns:
            True if quadrant is active, False otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        total_brightness = 0
        edge_count = 0
        sample_count = 0

        # Sample points every 10 pixels for speed
        for py in range(y, min(y + h, height), 10):
            for px in range(x, min(x + w, width), 10):
                brightness = int(gray[py, px])
                total_brightness += brightness

                # Check for edges (content variation)
                if px < min(x + w - 1, width - 1) and py < min(y + h - 1, height - 1):
                    brightness2 = int(gray[py, px + 1])
                    if abs(brightness - brightness2) > self.EDGE_CONTENT_THRESHOLD:
                        edge_count += 1

                sample_count += 1

        if sample_count == 0:
            return False

        avg_brightness = total_brightness / sample_count
        edge_density = edge_count / sample_count

        # Active if: reasonable brightness OR has content variation
        return avg_brightness > self.BRIGHTNESS_THRESHOLD or edge_density > self.EDGE_THRESHOLD

    def get_state_fingerprint(self, frame: np.ndarray) -> StateFingerprint:
        """
        Generate a state fingerprint for the current frame.

        Args:
            frame: BGR image frame

        Returns:
            StateFingerprint object describing the current state
        """
        height, width = frame.shape[:2]
        layout = self.detect_layout(frame)

        quadrants = [0, 0, 0, 0]

        if layout == 'grid':
            # Check all 4 quadrants
            half_w = width // 2
            half_h = height // 2

            # Top-left, Top-right, Bottom-left, Bottom-right
            quadrants[0] = 1 if self.is_quadrant_active(frame, 0, 0, half_w, half_h) else 0
            quadrants[1] = 1 if self.is_quadrant_active(frame, half_w, 0, half_w, half_h) else 0
            quadrants[2] = 1 if self.is_quadrant_active(frame, 0, half_h, half_w, half_h) else 0
            quadrants[3] = 1 if self.is_quadrant_active(frame, half_w, half_h, half_w, half_h) else 0
        else:
            # Fullscreen - only check if there's content
            quadrants[0] = 1 if self.is_quadrant_active(frame, 0, 0, width, height) else 0

        fingerprint = f"{layout}-{''.join(map(str, quadrants))}"

        return StateFingerprint(
            layout=layout,
            quadrants=quadrants,
            fingerprint=fingerprint
        )

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[StateChange]:
        """
        Process a single frame and detect state changes.

        Uses stability threshold to avoid false positives from transient frames.

        Args:
            frame: BGR image frame
            timestamp: Time in seconds

        Returns:
            StateChange object if a change was detected, None otherwise
        """
        state = self.get_state_fingerprint(frame)
        self.current_state = state

        change_detected = None

        # Check for changes
        if self.previous_state and state.fingerprint != self.previous_state.fingerprint:
            # State has changed - wait for stability
            if self.stable_state_fingerprint == state.fingerprint:
                self.stable_frames += 1

                # If stable for enough frames, register the change
                if self.stable_frames >= self.STABILITY_THRESHOLD:
                    change_detected = StateChange(
                        time=timestamp,
                        frame_number=self.frame_count,
                        from_state=self.previous_state,
                        to_state=state
                    )
                    self.changes.append(change_detected)
                    self.previous_state = state
                    self.stable_frames = 0
            else:
                # New potential state
                self.stable_state_fingerprint = state.fingerprint
                self.stable_frames = 1
        else:
            # State unchanged
            self.stable_frames = 0
            self.stable_state_fingerprint = None

        if not self.previous_state:
            self.previous_state = state

        self.frame_count += 1
        return change_detected

    def process_video(self, show_preview: bool = False, save_output: bool = True) -> List[StateChange]:
        """
        Process the entire video and detect all state changes.

        Args:
            show_preview: If True, display video with current state overlay
            save_output: If True, save results to JSON file

        Returns:
            List of detected StateChange objects
        """
        print(f"Processing video: {self.video_path}")
        print(f"Duration: {self.duration:.2f}s, FPS: {self.fps:.2f}, Frames: {self.total_frames}")
        print(f"Stability threshold: {self.STABILITY_THRESHOLD} frames")
        print("-" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = self.frame_count / self.fps
            change = self.process_frame(frame, timestamp)

            if change:
                print(f"Change detected at {self._format_time(timestamp)}")
                print(f"  From: {change.from_state.fingerprint}")
                print(f"  To:   {change.to_state.fingerprint}")

            # Show preview if requested
            if show_preview:
                self._draw_state_overlay(frame)
                cv2.imshow('Screen State Detector', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress update
            if self.frame_count % 100 == 0:
                progress = (self.frame_count / self.total_frames) * 100
                print(f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames} frames)", end='\r')

        print("\n" + "-" * 60)
        print(f"Processing complete! Detected {len(self.changes)} changes.")

        if save_output:
            self._save_results()

        self.cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        return self.changes

    def _draw_state_overlay(self, frame: np.ndarray):
        """Draw current state information on the frame."""
        if not self.current_state:
            return

        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        y_offset = 40
        cv2.putText(frame, f"Layout: {self.current_state.layout}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 30
        if self.current_state.layout == 'grid':
            active_count = sum(self.current_state.quadrants)
            cv2.putText(frame, f"Active Cameras: {active_count}/4",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw quadrant grid
            y_offset += 30
            for i, active in enumerate(self.current_state.quadrants):
                color = (0, 255, 0) if active else (0, 0, 255)
                text = f"Q{i+1}: {'ON' if active else 'OFF'}"
                x = 20 + (i % 2) * 100
                y = y_offset + (i // 2) * 25
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            status = 'Active' if self.current_state.quadrants[0] else 'Inactive'
            cv2.putText(frame, f"Status: {status}",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS.MS format."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 100)
        return f"{mins}:{secs:02d}.{ms:02d}"

    def _save_results(self):
        """Save detection results to JSON file."""
        video_name = Path(self.video_path).stem
        output_path = Path(self.video_path).parent / f"{video_name}_screen_changes.json"

        results = {
            'video_path': self.video_path,
            'duration': self.duration,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'changes_detected': len(self.changes),
            'changes': [change.to_dict() for change in self.changes]
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


def main():
    """Command-line interface for the screen state detector."""
    parser = argparse.ArgumentParser(
        description='Detect layout and state changes in multicam videos'
    )
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--preview', action='store_true',
                       help='Show video preview with state overlay')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to JSON file')

    args = parser.parse_args()

    detector = ScreenStateDetector(args.video_path)
    changes = detector.process_video(
        show_preview=args.preview,
        save_output=not args.no_save
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, change in enumerate(changes, 1):
        print(f"\nChange #{i} at {detector._format_time(change.time)}")
        print(f"  From: {change.from_state.layout} - {change.from_state.quadrants}")
        print(f"  To:   {change.to_state.layout} - {change.to_state.quadrants}")


if __name__ == '__main__':
    main()
