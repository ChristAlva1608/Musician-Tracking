"""
Processing endpoints for video analysis
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import sys
import asyncio
import uuid
from datetime import datetime, timedelta

# Add project src directory to path
# From routers/processing.py -> routers -> backend -> web -> src
routers_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(routers_dir)
web_dir = os.path.dirname(backend_dir)
src_root = os.path.dirname(web_dir)
sys.path.insert(0, src_root)
sys.path.insert(0, backend_dir)  # Add backend dir for websocket_manager

from detect_v2 import DetectorV2
from integrated_video_processor import IntegratedVideoProcessor
from websocket_manager import manager

router = APIRouter()

# Pydantic models
class SingleVideoRequest(BaseModel):
    video_path: str
    skip_frames: int = 0
    hand_model: str = "mediapipe"
    pose_model: str = "yolo"
    facemesh_model: str = "yolo+mediapipe"
    emotion_model: str = "none"
    transcript_model: str = "whisper"
    save_output_video: bool = True
    display_output: bool = False

class FolderProcessingRequest(BaseModel):
    folder_path: str
    skip_frames: int = 0
    hand_model: str = "mediapipe"
    pose_model: str = "yolo"
    facemesh_model: str = "yolo+mediapipe"
    emotion_model: str = "none"
    transcript_model: str = "whisper"
    processing_type: str = "full_frames"
    unified_videos: bool = True
    limit_processing_duration: bool = False
    max_processing_duration: float = 10.0

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str

# Global job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}

# Initialize processing jobs database
try:
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Import from the correct path
    sys.path.insert(0, os.path.join(src_root, 'database'))
    from processing_jobs import ProcessingJobsDatabase
    jobs_db = ProcessingJobsDatabase()
    print(f"✅ ProcessingJobsDatabase initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize ProcessingJobsDatabase: {e}")
    import traceback
    traceback.print_exc()
    jobs_db = None

def create_job(job_type: str, request_data: Dict[str, Any]) -> str:
    """Create a new processing job"""
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "type": job_type,
        "status": "queued",
        "progress": 0,
        "message": "Job created",
        "request_data": request_data,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "output_files": [],
        "logs": []
    }
    active_jobs[job_id] = job_data

    # Also save to database if available
    if jobs_db:
        try:
            print(f"Attempting to save job {job_id} to database...")
            result = jobs_db.create_job(job_id, job_type, request_data)
            print(f"Database save result: {result}")
        except Exception as e:
            print(f"Error saving job to database: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Warning: jobs_db is None, cannot save to database")

    return job_id

async def update_job_status(job_id: str, status: str, progress: int = None, message: str = None,
                           output_files: List[str] = None, frame_metrics: Dict[str, Any] = None):
    """Update job status and broadcast to WebSocket clients"""
    if job_id in active_jobs:
        active_jobs[job_id]["status"] = status
        active_jobs[job_id]["updated_at"] = datetime.now().isoformat()

        if progress is not None:
            active_jobs[job_id]["progress"] = progress
        if message:
            active_jobs[job_id]["message"] = message
            active_jobs[job_id]["logs"].append({
                "timestamp": datetime.now().isoformat(),
                "message": message
            })
        if output_files:
            active_jobs[job_id]["output_files"] = output_files
        if frame_metrics:
            active_jobs[job_id]["frame_metrics"] = frame_metrics
            # Calculate expected finish time
            if frame_metrics.get("frames_processed") and frame_metrics.get("total_frames"):
                avg_time_per_frame = frame_metrics.get("avg_processing_time_ms", 100) / 1000.0  # Convert to seconds
                frames_remaining = frame_metrics["total_frames"] - frame_metrics["frames_processed"]
                estimated_seconds_remaining = frames_remaining * avg_time_per_frame
                estimated_finish_time = datetime.now() + timedelta(seconds=estimated_seconds_remaining)
                active_jobs[job_id]["estimated_finish_time"] = estimated_finish_time.isoformat()
                active_jobs[job_id]["estimated_seconds_remaining"] = estimated_seconds_remaining

        # Update database if available
        if jobs_db:
            try:
                updates = {"status": status}
                if progress is not None:
                    updates["progress"] = progress
                if message:
                    updates["message"] = message
                if output_files:
                    updates["output_files"] = output_files
                if frame_metrics:
                    # Store frame metrics in database too
                    updates["frame_metrics"] = frame_metrics
                jobs_db.update_job(job_id, updates)
            except Exception as e:
                print(f"Error updating job in database: {e}")

        # Broadcast to WebSocket clients
        print(f"Broadcasting job update for {job_id}: progress={progress}, status={status}")
        await manager.broadcast({"type": "job_update", "data": active_jobs[job_id]})

def run_single_video_detection(job_id: str, request: SingleVideoRequest):
    """Run single video detection in background"""
    try:
        asyncio.run(update_job_status(job_id, "running", 10, "Starting video detection..."))

        # Load current configuration
        import yaml
        config_path = os.path.join(src_root, 'config', 'config_v1.yaml')
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)

        # Create temporary config for this job, respecting database settings
        temp_config = {
            "video": {
                "source_path": request.video_path,
                "use_webcam": False,
                "skip_frames": request.skip_frames,
                "display_output": request.display_output,
                "save_output_video": request.save_output_video,
                "output_video_path": f"src/output/web_job_{job_id}.mp4",
                "preserve_audio": True,
                "generate_report": True,
                "report_path": f"src/output/web_job_{job_id}_report.txt"
            },
            "detection": {
                "hand_model": request.hand_model,
                "pose_model": request.pose_model,
                "facemesh_model": request.facemesh_model,
                "emotion_model": request.emotion_model,
                "transcript_model": request.transcript_model
            },
            "database": current_config.get('database', {})  # Use actual database config
        }

        # Save temporary config
        import tempfile
        import cv2
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            temp_config_path = f.name

        asyncio.run(update_job_status(job_id, "running", 20, "Analyzing video..."))

        # Get video info for total frames
        cap = cv2.VideoCapture(request.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        frame_metrics = {
            "total_frames": total_frames,
            "frames_processed": 0,
            "fps": fps,
            "avg_processing_time_ms": 100  # Initial estimate
        }

        asyncio.run(update_job_status(job_id, "running", 30, "Initializing detector...",
                                      frame_metrics=frame_metrics))

        # Create a custom detector wrapper that reports progress
        class ProgressDetector(DetectorV2):
            def __init__(self, config_file):
                super().__init__(config_file)
                self.job_id = job_id
                self.frames_processed = 0
                self.processing_times = []
                self.total_frames = total_frames

            def process_frame(self, frame):
                import time
                start_time = time.time()
                result = super().process_frame(frame)
                processing_time = (time.time() - start_time) * 1000  # Convert to ms

                self.frames_processed += 1
                self.processing_times.append(processing_time)

                # Update progress every 10 frames
                if self.frames_processed % 10 == 0:
                    avg_time = sum(self.processing_times[-100:]) / len(self.processing_times[-100:])
                    progress = int((self.frames_processed / self.total_frames) * 100)
                    frame_metrics = {
                        "total_frames": self.total_frames,
                        "frames_processed": self.frames_processed,
                        "fps": fps,
                        "avg_processing_time_ms": avg_time
                    }
                    asyncio.run(update_job_status(self.job_id, "running", progress,
                                                 f"Processing frame {self.frames_processed}/{self.total_frames}",
                                                 frame_metrics=frame_metrics))
                return result

        # Run detection with progress tracking
        detector = ProgressDetector(temp_config_path)
        detector.process_video(request.video_path)

        # Get output files
        output_files = []
        if os.path.exists(temp_config["video"]["output_video_path"]):
            video_name = os.path.basename(request.video_path)
            output_files.append({
                "path": temp_config["video"]["output_video_path"],
                "type": "annotated",
                "name": f"annotated_{video_name}",
                "camera": video_name.replace(".mp4", "")
            })

        report_path = temp_config["video"].get("report_path")
        if report_path and os.path.exists(report_path):
            output_files.append({
                "path": report_path,
                "type": "report",
                "name": os.path.basename(report_path),
                "camera": "report"
            })

        asyncio.run(update_job_status(job_id, "completed", 100, "Video detection completed", output_files))

        # Cleanup
        os.unlink(temp_config_path)

    except Exception as e:
        asyncio.run(update_job_status(job_id, "failed", 0, f"Error: {str(e)}"))

def run_folder_processing(job_id: str, request: FolderProcessingRequest):
    """Run folder processing in background"""
    try:
        asyncio.run(update_job_status(job_id, "running", 10, "Starting folder processing..."))

        # Load current configuration
        import yaml
        config_path = os.path.join(src_root, 'config', 'config_v1.yaml')
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)

        # Create temporary config
        temp_config = {
            "integrated_processor": {
                "alignment_directory": request.folder_path,
                "processing_type": request.processing_type,
                "unified_videos": request.unified_videos,
                "limit_processing_duration": request.limit_processing_duration,
                "max_processing_duration": request.max_processing_duration,
                "aligned_videos_dir": "src/output/aligned_videos",
                "detection_videos_dir": "src/output/annotated_detection_videos",
                "unified_videos_dir": "src/output/unified_videos",
                "check_existing_alignment": True,
                "create_aligned_videos": True,
                "run_detection": True
            },
            "video": {
                "skip_frames": request.skip_frames,
                "save_output_video": True,
                "preserve_audio": True
            },
            "detection": {
                "hand_model": request.hand_model,
                "pose_model": request.pose_model,
                "facemesh_model": request.facemesh_model,
                "emotion_model": request.emotion_model,
                "transcript_model": request.transcript_model
            },
            "database": current_config.get('database', {})  # Use actual database config
        }

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            temp_config_path = f.name

        asyncio.run(update_job_status(job_id, "running", 30, "Processing videos..."))

        # Run integrated processor
        processor = IntegratedVideoProcessor(temp_config_path)

        asyncio.run(update_job_status(job_id, "running", 50, "Aligning and processing videos..."))

        processor.process()

        # Get output files
        output_files = []

        # Collect annotated detection videos (per camera)
        detection_dir = temp_config["integrated_processor"]["detection_videos_dir"]
        if os.path.exists(detection_dir):
            for file in os.listdir(detection_dir):
                if file.endswith('.mp4'):
                    full_path = os.path.join(detection_dir, file)
                    output_files.append({
                        "path": full_path,
                        "type": "annotated",
                        "name": file,
                        "camera": file.replace("annotated_", "").replace(".mp4", "")
                    })

        # Collect aligned videos (if created)
        aligned_dir = temp_config["integrated_processor"]["aligned_videos_dir"]
        if os.path.exists(aligned_dir):
            for file in os.listdir(aligned_dir):
                if file.endswith('.mp4'):
                    full_path = os.path.join(aligned_dir, file)
                    output_files.append({
                        "path": full_path,
                        "type": "aligned",
                        "name": file,
                        "camera": file.replace("aligned_", "").replace(".mp4", "")
                    })

        # Collect unified video output
        unified_dir = temp_config["integrated_processor"]["unified_videos_dir"]
        if os.path.exists(unified_dir):
            for file in os.listdir(unified_dir):
                if file.endswith('.mp4'):
                    full_path = os.path.join(unified_dir, file)
                    output_files.append({
                        "path": full_path,
                        "type": "unified",
                        "name": file,
                        "camera": "all"
                    })

        asyncio.run(update_job_status(job_id, "completed", 100, "Folder processing completed", output_files))

        # Cleanup
        os.unlink(temp_config_path)

    except Exception as e:
        asyncio.run(update_job_status(job_id, "failed", 0, f"Error: {str(e)}"))

@router.post("/single-video", response_model=JobResponse)
async def process_single_video(request: SingleVideoRequest, background_tasks: BackgroundTasks):
    """Process a single video file"""
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=400, detail="Video file not found")

    job_id = create_job("single_video", request.dict())
    background_tasks.add_task(run_single_video_detection, job_id, request)

    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Video processing job created",
        created_at=active_jobs[job_id]["created_at"]
    )

@router.post("/folder", response_model=JobResponse)
async def process_folder(request: FolderProcessingRequest, background_tasks: BackgroundTasks):
    """Process all videos in a folder"""
    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=400, detail="Folder not found")

    job_id = create_job("folder_processing", request.dict())
    background_tasks.add_task(run_folder_processing, job_id, request)

    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Folder processing job created",
        created_at=active_jobs[job_id]["created_at"]
    )

@router.get("/jobs")
async def get_all_jobs():
    """Get all processing jobs"""
    # First try to get from database
    if jobs_db:
        try:
            db_jobs = jobs_db.get_all_jobs()
            if db_jobs:
                # Merge with active_jobs to get latest progress data
                merged_jobs = []
                for db_job in db_jobs:
                    job_id = db_job.get('job_id')
                    if job_id in active_jobs:
                        # Use active job data (has latest progress)
                        merged_jobs.append(active_jobs[job_id])
                    else:
                        # Use database job data
                        merged_jobs.append(db_job)

                # Also add any active jobs not in database
                for job_id, job_data in active_jobs.items():
                    if not any(j.get('job_id') == job_id for j in db_jobs):
                        merged_jobs.append(job_data)

                return {"jobs": merged_jobs}
        except Exception as e:
            print(f"Error fetching jobs from database: {e}")

    # Fallback to in-memory jobs
    return {"jobs": list(active_jobs.values())}

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    # First try database
    if jobs_db:
        try:
            job = jobs_db.get_job(job_id)
            if job:
                return job
        except Exception as e:
            print(f"Error fetching job {job_id} from database: {e}")

    # Fallback to in-memory
    if job_id in active_jobs:
        return active_jobs[job_id]

    raise HTTPException(status_code=404, detail="Job not found")

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a processing job"""
    if job_id in active_jobs:
        await update_job_status(job_id, "cancelled", message="Job cancelled by user")
        return {"message": "Job cancelled"}

    raise HTTPException(status_code=404, detail="Job not found")

@router.delete("/jobs/{job_id}/delete")
async def delete_job(job_id: str):
    """Permanently delete a processing job from database"""
    try:
        # Delete from database if available
        if jobs_db:
            success = jobs_db.delete_job(job_id)
            if success:
                # Also remove from active jobs if present
                if job_id in active_jobs:
                    del active_jobs[job_id]
                return {"message": "Job deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Job not found in database")
        else:
            # Fallback: just remove from active jobs
            if job_id in active_jobs:
                del active_jobs[job_id]
                return {"message": "Job deleted from memory"}
            else:
                raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get available detection models"""
    return {
        "hand_models": ["mediapipe", "yolo"],
        "pose_models": ["mediapipe", "yolo"],
        "facemesh_models": ["mediapipe", "yolo+mediapipe", "yolo", "none"],
        "emotion_models": ["deepface", "ghostfacenet", "fer", "mediapipe", "none"],
        "transcript_models": ["whisper", "none"]
    }

@router.get("/jobs/stats")
async def get_job_statistics():
    """Get statistics about processing jobs"""
    if jobs_db:
        try:
            stats = jobs_db.get_job_statistics()
            return stats
        except Exception as e:
            print(f"Error getting job statistics: {e}")

    # Fallback to calculating from in-memory jobs
    stats = {
        "total": len(active_jobs),
        "by_status": {},
        "by_type": {}
    }

    for job in active_jobs.values():
        status = job.get("status", "unknown")
        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        job_type = job.get("type", "unknown")
        stats["by_type"][job_type] = stats["by_type"].get(job_type, 0) + 1

    return stats