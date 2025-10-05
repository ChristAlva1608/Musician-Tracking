"""
File system navigation and management endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import mimetypes
from pathlib import Path
import sys

# Add project src directory to path
# From routers/files.py -> routers -> backend -> web -> src
routers_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(routers_dir)
web_dir = os.path.dirname(backend_dir)
src_root = os.path.dirname(web_dir)
sys.path.insert(0, src_root)

router = APIRouter()

class FileItem(BaseModel):
    name: str
    path: str
    type: str  # "file" or "directory"
    size: Optional[int] = None
    modified: Optional[str] = None
    extension: Optional[str] = None
    is_video: bool = False

class DirectoryListing(BaseModel):
    current_path: str
    parent_path: Optional[str]
    items: List[FileItem]

# Video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

def is_safe_path(path: str) -> bool:
    """Check if path is safe to access"""
    try:
        # Resolve path and check if it's within project directory or common video directories
        resolved_path = os.path.abspath(path)
        safe_paths = [
            os.path.abspath(src_root),
            os.path.expanduser("~/Videos"),
            os.path.expanduser("~/Movies"),
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"),
            "/Volumes"  # macOS external drives
        ]

        return any(resolved_path.startswith(safe_path) for safe_path in safe_paths)
    except:
        return False

def get_file_info(path: str) -> FileItem:
    """Get information about a file or directory"""
    stat = os.stat(path)
    name = os.path.basename(path)
    is_dir = os.path.isdir(path)

    extension = None
    is_video = False
    if not is_dir:
        extension = Path(path).suffix.lower()
        is_video = extension in VIDEO_EXTENSIONS

    return FileItem(
        name=name,
        path=path,
        type="directory" if is_dir else "file",
        size=None if is_dir else stat.st_size,
        modified=str(stat.st_mtime),
        extension=extension,
        is_video=is_video
    )

@router.get("/browse")
async def browse_directory(path: str = None) -> DirectoryListing:
    """Browse files and directories"""
    if path is None:
        path = os.path.join(src_root, "video")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Path not found")

    if not is_safe_path(path):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Path is not a directory")

    try:
        items = []
        for item_name in sorted(os.listdir(path)):
            item_path = os.path.join(path, item_name)
            try:
                # Skip hidden files and system files
                if item_name.startswith('.'):
                    continue

                file_info = get_file_info(item_path)
                items.append(file_info)
            except (OSError, PermissionError):
                # Skip files we can't access
                continue

        parent_path = os.path.dirname(path) if path != "/" else None
        if parent_path and not is_safe_path(parent_path):
            parent_path = None

        return DirectoryListing(
            current_path=path,
            parent_path=parent_path,
            items=items
        )

    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}")

@router.get("/video-folders")
async def get_video_folders():
    """Get common video folder locations"""
    folders = []

    # Project video folder
    project_video_path = os.path.join(src_root, "video")
    if os.path.exists(project_video_path):
        folders.append({
            "name": "Project Videos",
            "path": project_video_path,
            "description": "Videos in the project directory"
        })

    # Common user directories
    common_paths = [
        ("~/Videos", "User Videos"),
        ("~/Movies", "User Movies"),
        ("~/Desktop", "Desktop"),
        ("~/Documents", "Documents")
    ]

    for path_pattern, name in common_paths:
        expanded_path = os.path.expanduser(path_pattern)
        if os.path.exists(expanded_path):
            folders.append({
                "name": name,
                "path": expanded_path,
                "description": f"System {name.lower()} folder"
            })

    # External drives (macOS)
    volumes_path = "/Volumes"
    if os.path.exists(volumes_path):
        try:
            for volume in os.listdir(volumes_path):
                volume_path = os.path.join(volumes_path, volume)
                if os.path.isdir(volume_path) and volume != "Macintosh HD":
                    folders.append({
                        "name": f"External: {volume}",
                        "path": volume_path,
                        "description": f"External drive: {volume}"
                    })
        except PermissionError:
            pass

    return {"folders": folders}

@router.get("/search")
async def search_files(query: str, path: str = None, file_type: str = "video"):
    """Search for files"""
    if path is None:
        path = os.path.join(src_root, "video")

    if not is_safe_path(path):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    results = []
    query_lower = query.lower()

    try:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                if file.startswith('.'):
                    continue

                file_path = os.path.join(root, file)
                file_info = get_file_info(file_path)

                # Filter by file type
                if file_type == "video" and not file_info.is_video:
                    continue

                # Check if query matches filename
                if query_lower in file.lower():
                    results.append(file_info)

                # Limit results
                if len(results) >= 50:
                    break

            if len(results) >= 50:
                break

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    return {"results": results, "query": query, "total": len(results)}

@router.get("/info")
async def get_file_info_endpoint(path: str):
    """Get detailed information about a file"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    if not is_safe_path(path):
        raise HTTPException(status_code=403, detail="Access to this file is not allowed")

    try:
        file_info = get_file_info(path)

        # Add additional info for video files
        if file_info.is_video:
            try:
                import cv2
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0

                    file_info.video_info = {
                        "fps": fps,
                        "frame_count": frame_count,
                        "width": width,
                        "height": height,
                        "duration": duration
                    }
                    cap.release()
            except:
                pass

        return file_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting file info: {str(e)}")

@router.get("/download")
async def download_file(path: str):
    """Download a file"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    if not is_safe_path(path):
        raise HTTPException(status_code=403, detail="Access to this file is not allowed")

    if os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Cannot download directory")

    filename = os.path.basename(path)
    media_type = mimetypes.guess_type(path)[0]

    return FileResponse(
        path=path,
        filename=filename,
        media_type=media_type
    )

@router.get("/outputs")
async def get_output_files():
    """Get list of generated output files"""
    output_dirs = [
        os.path.join(src_root, "output"),
        os.path.join(src_root, "output", "aligned_videos"),
        os.path.join(src_root, "output", "annotated_detection_videos"),
        os.path.join(src_root, "output", "unified_videos")
    ]

    output_files = []

    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if not file.startswith('.'):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        file_info = get_file_info(file_path)
                        file_info.category = os.path.basename(output_dir)
                        output_files.append(file_info)

    # Sort by modification time (newest first)
    output_files.sort(key=lambda x: float(x.modified), reverse=True)

    return {"output_files": output_files}