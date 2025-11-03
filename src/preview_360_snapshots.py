#!/usr/bin/env python3
"""
Generate preview snapshots at different angles from 360° video
Works on macOS without requiring window interaction
"""

import cv2
import numpy as np
import sys
import os

def equirect_to_perspective(equirect_img, yaw, pitch, fov):
    """Convert equirectangular to perspective view"""
    h, w = equirect_img.shape[:2]
    out_h, out_w = 720, 1280

    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    fov_rad = np.deg2rad(fov)

    y_coords, x_coords = np.mgrid[0:out_h, 0:out_w]
    x_norm = (x_coords - out_w / 2) / (out_w / 2)
    y_norm = (y_coords - out_h / 2) / (out_h / 2)

    x_norm *= np.tan(fov_rad / 2)
    y_norm *= np.tan(fov_rad / 2) * (out_h / out_w)

    z = np.ones_like(x_norm)
    x = x_norm
    y = y_norm

    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    y_rot = y * np.cos(pitch_rad) - z * np.sin(pitch_rad)
    z_rot = y * np.sin(pitch_rad) + z * np.cos(pitch_rad)

    x_final = x * np.cos(yaw_rad) + z_rot * np.sin(yaw_rad)
    z_final = -x * np.sin(yaw_rad) + z_rot * np.cos(yaw_rad)

    theta = np.arctan2(x_final, z_final)
    phi = np.arcsin(np.clip(y_rot, -1, 1))

    equirect_x = ((theta / np.pi + 1) * w / 2).astype(np.float32)
    equirect_y = ((phi / (np.pi / 2) + 1) * h / 2).astype(np.float32)

    result = cv2.remap(equirect_img, equirect_x, equirect_y,
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # Add text overlay
    cv2.putText(result, f"Yaw: {yaw}  Pitch: {pitch}  FOV: {fov}",
               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python preview_360_snapshots.py <video_file>")
        print("Example: python preview_360_snapshots.py VID_20250709_203100_00_010.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"\n📹 Processing 360° video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        sys.exit(1)

    # Get middle frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()

    if not ret:
        print(f"❌ Error: Cannot read frame from video")
        cap.release()
        sys.exit(1)

    print(f"✅ Extracted frame {total_frames // 2} / {total_frames}")

    # Generate previews at different angles
    angles = [
        {'name': 'front', 'yaw': 0, 'pitch': 0, 'fov': 120},
        {'name': 'front_down', 'yaw': 0, 'pitch': -15, 'fov': 120},
        {'name': 'right', 'yaw': 90, 'pitch': 0, 'fov': 120},
        {'name': 'back', 'yaw': 180, 'pitch': 0, 'fov': 120},
        {'name': 'left', 'yaw': 270, 'pitch': 0, 'fov': 120},
        {'name': 'front_wide', 'yaw': 0, 'pitch': 0, 'fov': 140},
    ]

    print(f"\n📸 Generating {len(angles)} preview snapshots...")
    print("="*70)

    for angle in angles:
        preview = equirect_to_perspective(frame, angle['yaw'], angle['pitch'], angle['fov'])

        output_path = os.path.join(output_dir, f"{video_name}_preview_{angle['name']}.jpg")
        cv2.imwrite(output_path, preview)

        print(f"✅ {angle['name']:12s} → {output_path}")
        print(f"   Yaw={angle['yaw']:3d}°, Pitch={angle['pitch']:3d}°, FOV={angle['fov']:3d}°")

        # Generate FFmpeg command for this angle
        output_video = os.path.join(output_dir, f"{video_name}_converted_{angle['name']}.mp4")
        print(f"   FFmpeg command:")
        print(f"   ffmpeg -i \"{video_path}\" \\")
        print(f"     -vf \"v360=e:flat:iv_fov=360:ih_fov=180:yaw={angle['yaw']}:pitch={angle['pitch']}:roll=0:w=1920:h=1080:interp=linear\" \\")
        print(f"     -c:v libx264 -preset medium -crf 23 -c:a copy \\")
        print(f"     \"{output_video}\"")
        print()

    cap.release()

    print("="*70)
    print(f"\n🎉 Done! Preview images saved in: {output_dir}")
    print("\n📋 Next steps:")
    print("1. Open the preview images and pick your favorite angle")
    print("2. Copy the corresponding FFmpeg command from above")
    print("3. Run it to convert your video")
    print("4. Update config_v2.yaml with the converted video path")
    print("5. Run: python src/detect_v2_3d.py --config src/config/config_v2.yaml")


if __name__ == '__main__':
    main()
