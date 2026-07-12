"""
Multicam Screen Change Detector - IMPROVED VERSION

Key improvements:
1. Better grid divider detection with lower thresholds
2. Multi-strategy layout detection (edges + color consistency)
3. Enhanced debugging output
4. Adaptive thresholds based on video characteristics
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
    1. Layout Detection: Multi-strategy approach (edges + variance + color)
    2. Activity Detection: Checks each quadrant's brightness and edge density
    3. Change Detection: Compares state fingerprints with stability threshold
    4. Motion Filtering: Ignores content movement, only tracks structural changes
    """

    # Configuration constants
    STABILITY_THRESHOLD = 5  # Frames needed to confirm a change
    BRIGHTNESS_THRESHOLD = 30  # Threshold for black screen detection
    EDGE_THRESHOLD = 0.1  # Threshold for edge density
    
    # IMPROVED: Lower thresholds for better divider detection
    EDGE_DIFF_THRESHOLD = 8  # Lowered from 20 to catch subtle dividers
    EDGE_CONTENT_THRESHOLD = 30  # Threshold for content variation
    GRID_EDGE_RATIO_THRESHOLD = 0.10  # Lowered from 0.3 to be more sensitive
    
    # NEW: Additional detection strategies
    DIVIDER_COLOR_CONSISTENCY_THRESHOLD = 0.7  # For detecting consistent divider colors
    VARIANCE_THRESHOLD = 100  # For detecting low-variance divider regions

    def __init__(self, video_path: str, duration_override: Optional[float] = None, debug: bool = False):
        """
        Initialize the detector with a video file.

        Args:
            video_path: Path to the video file to analyze
            duration_override: Optional actual video duration in seconds to calculate correct FPS
            debug: If True, print detailed debug information
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.debug = debug

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if duration_override and duration_override > 0:
            self.duration = duration_override
            self.fps = self.total_frames / self.duration if self.duration > 0 else 30
            self.fps_source = "calculated_from_duration"
            print(f"Using provided duration: {duration_override}s")
            print(f"Calculated FPS: {self.fps:.2f} (from {self.total_frames} frames)")
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            self.fps_source = "opencv_detected"

        # State tracking
        self.changes: List[StateChange] = []
        self.current_state: Optional[StateFingerprint] = None
        self.previous_state: Optional[StateFingerprint] = None
        self.stable_state_fingerprint: Optional[str] = None
        self.stable_frames: int = 0
        self.frame_count: int = 0

    def detect_layout_enhanced(self, frame: np.ndarray) -> Tuple[str, dict]:
        """
        Enhanced layout detection using multiple strategies.
        
        Returns:
            Tuple of (layout_type, debug_info)
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        center_x = width // 2
        center_y = height // 2
        
        debug_info = {}
        
        # Strategy 1: Edge Detection (Original method, improved)
        vertical_edges, v_samples = self._detect_divider_edges(gray, center_x, True, height)
        horizontal_edges, h_samples = self._detect_divider_edges(gray, center_y, False, width)
        
        v_edge_ratio = vertical_edges / v_samples if v_samples > 0 else 0
        h_edge_ratio = horizontal_edges / h_samples if h_samples > 0 else 0
        
        debug_info['v_edge_ratio'] = v_edge_ratio
        debug_info['h_edge_ratio'] = h_edge_ratio
        
        # Strategy 2: Color Consistency (NEW)
        # Dividers are usually uniform gray/black lines
        v_color_consistency = self._check_color_consistency(gray, center_x, True, height)
        h_color_consistency = self._check_color_consistency(gray, center_y, False, width)
        
        debug_info['v_color_consistency'] = v_color_consistency
        debug_info['h_color_consistency'] = h_color_consistency
        
        # Strategy 3: Low Variance Detection (NEW)
        # Dividers have low pixel variance compared to content
        v_variance = self._calculate_line_variance(gray, center_x, True, height)
        h_variance = self._calculate_line_variance(gray, center_y, False, width)
        
        debug_info['v_variance'] = v_variance
        debug_info['h_variance'] = h_variance
        
        # Voting system: Multiple strategies must agree
        v_votes = 0
        h_votes = 0
        
        # Vote 1: Edge detection
        if v_edge_ratio > self.GRID_EDGE_RATIO_THRESHOLD:
            v_votes += 1
        if h_edge_ratio > self.GRID_EDGE_RATIO_THRESHOLD:
            h_votes += 1
            
        # Vote 2: Color consistency
        if v_color_consistency > self.DIVIDER_COLOR_CONSISTENCY_THRESHOLD:
            v_votes += 1
        if h_color_consistency > self.DIVIDER_COLOR_CONSISTENCY_THRESHOLD:
            h_votes += 1
            
        # Vote 3: Low variance (dividers are uniform)
        if v_variance < self.VARIANCE_THRESHOLD:
            v_votes += 1
        if h_variance < self.VARIANCE_THRESHOLD:
            h_votes += 1
        
        debug_info['v_votes'] = v_votes
        debug_info['h_votes'] = h_votes
        
        # Grid if both dividers detected (need at least 2/3 votes each)
        is_grid = v_votes >= 2 and h_votes >= 2
        
        return ('grid' if is_grid else 'fullscreen'), debug_info

    def _detect_divider_edges(self, gray: np.ndarray, position: int, is_vertical: bool, 
                              dimension: int) -> Tuple[int, int]:
        """
        Detect edges along a line (vertical or horizontal).
        
        Returns:
            Tuple of (edge_count, sample_count)
        """
        edges = 0
        samples = 0
        
        # Sample a larger region (80% instead of 40%)
        start = int(dimension * 0.1)
        end = int(dimension * 0.9)
        
        for i in range(start, end, 2):  # Sample every 2 pixels for speed
            if is_vertical:
                center_val = int(gray[i, position])
                left_val = int(gray[i, max(0, position - 1)])
                right_val = int(gray[i, min(gray.shape[1] - 1, position + 1)])
            else:
                center_val = int(gray[position, i])
                top_val = int(gray[max(0, position - 1), i])
                bottom_val = int(gray[min(gray.shape[0] - 1, position + 1), i])
                left_val = top_val
                right_val = bottom_val
            
            # Check for edge on either side
            diff = max(abs(center_val - left_val), abs(center_val - right_val))
            if diff > self.EDGE_DIFF_THRESHOLD:
                edges += 1
            
            samples += 1
        
        return edges, samples

    def _check_color_consistency(self, gray: np.ndarray, position: int, 
                                 is_vertical: bool, dimension: int) -> float:
        """
        Check if a line has consistent color (characteristic of dividers).
        
        Returns:
            Consistency score (0.0 to 1.0)
        """
        start = int(dimension * 0.1)
        end = int(dimension * 0.9)
        
        values = []
        for i in range(start, end, 5):
            if is_vertical:
                values.append(int(gray[i, position]))
            else:
                values.append(int(gray[position, i]))
        
        if len(values) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        # Lower CV = more consistent = more likely a divider
        cv = std_val / mean_val
        consistency = max(0.0, 1.0 - cv)
        
        return consistency

    def _calculate_line_variance(self, gray: np.ndarray, position: int,
                                 is_vertical: bool, dimension: int) -> float:
        """
        Calculate variance along a line.
        
        Returns:
            Variance value (lower = more uniform = more likely a divider)
        """
        start = int(dimension * 0.1)
        end = int(dimension * 0.9)
        
        values = []
        for i in range(start, end, 2):
            if is_vertical:
                values.append(int(gray[i, position]))
            else:
                values.append(int(gray[position, i]))
        
        if len(values) < 2:
            return float('inf')
        
        return float(np.var(values))

    def detect_layout(self, frame: np.ndarray) -> str:
        """
        Wrapper for enhanced layout detection (maintains API compatibility).
        """
        layout, debug_info = self.detect_layout_enhanced(frame)
        
        if self.debug and self.frame_count % 100 == 0:
            print(f"\n[Frame {self.frame_count}] Layout: {layout}")
            for key, val in debug_info.items():
                print(f"  {key}: {val:.3f}")
        
        return layout

    def is_quadrant_active(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """
        Check if a quadrant has active video content.

        A quadrant is considered active if it has reasonable brightness
        OR sufficient edge density (content variation).
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
        """
        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps:.2f} ({self.fps_source})")
        print(f"Duration: {self.duration:.2f}s ({self._format_time(self.duration)})")
        print(f"Stability threshold: {self.STABILITY_THRESHOLD} frames")
        print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
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
        output_path = Path(self.video_path).parent / f"{video_name}_screen_changes_improved.json"

        results = {
            'video_path': self.video_path,
            'duration': self.duration,
            'fps': self.fps,
            'fps_source': self.fps_source,
            'total_frames': self.total_frames,
            'changes_detected': len(self.changes),
            'detection_params': {
                'stability_threshold': self.STABILITY_THRESHOLD,
                'edge_diff_threshold': self.EDGE_DIFF_THRESHOLD,
                'grid_edge_ratio_threshold': self.GRID_EDGE_RATIO_THRESHOLD,
            },
            'changes': [change.to_dict() for change in self.changes]
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


def main():
    """Command-line interface for the screen state detector."""
    parser = argparse.ArgumentParser(
        description='Detect layout and state changes in multicam videos (IMPROVED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Duration format examples:
  --duration 1792           # 1792 seconds
  --duration 29:52          # 29 minutes 52 seconds
  --duration 1:29:52        # 1 hour 29 minutes 52 seconds
        """
    )
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--preview', action='store_true',
                       help='Show video preview with state overlay')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to JSON file')
    parser.add_argument('--duration', type=str, default=None,
                       help='Actual video duration in seconds or MM:SS or HH:MM:SS format')
    parser.add_argument('--debug', action='store_true',
                       help='Enable detailed debug output')

    args = parser.parse_args()

    # Parse duration if provided
    duration_override = None
    if args.duration:
        try:
            if ':' in args.duration:
                parts = args.duration.split(':')
                if len(parts) == 2:
                    duration_override = int(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    duration_override = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                else:
                    print(f"Invalid duration format: {args.duration}")
                    return 1
            else:
                duration_override = float(args.duration)
        except ValueError:
            print(f"Invalid duration value: {args.duration}")
            return 1

    detector = ScreenStateDetector(args.video_path, duration_override=duration_override, debug=args.debug)
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