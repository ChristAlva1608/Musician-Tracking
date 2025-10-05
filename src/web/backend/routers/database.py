"""
Database query and management endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add project src directory to path
# From routers/database.py -> routers -> backend -> web -> src
routers_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(routers_dir)
web_dir = os.path.dirname(backend_dir)
src_root = os.path.dirname(web_dir)
sys.path.insert(0, src_root)

from database.setup import MusicianDatabase, VideoAlignmentDatabase, ChunkVideoAlignmentDatabase
from supabase import create_client, Client
import yaml

# Load config from YAML
config_path = Path(src_root) / 'config' / 'config_v1.yaml'
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)
print(f"Loaded config from: {config_path}")

router = APIRouter()

class QueryParams(BaseModel):
    limit: int = 100
    offset: int = 0
    source: Optional[str] = None
    camera_prefix: Optional[str] = None

@router.get("/debug/config")
async def debug_config():
    """Debug endpoint to check if config is loaded"""
    supabase_config = CONFIG.get('database', {}).get('supabase', {})
    return {
        "config_file_path": str(Path(src_root) / 'config' / 'config_v1.yaml'),
        "database_type": CONFIG.get('database', {}).get('use_database_type', 'local'),
        "supabase_url_present": supabase_config.get('url') is not None,
        "supabase_key_present": supabase_config.get('anon_key') is not None,
        "supabase_url_length": len(supabase_config.get('url', '')),
        "supabase_key_length": len(supabase_config.get('anon_key', ''))
    }

@router.get("/tables")
async def get_tables():
    """Get list of available database tables from Supabase"""
    try:
        # Get Supabase credentials from environment
        url = os.getenv("SUPABASE_URL", "").strip().strip("'\"")
        key = os.getenv("SUPABASE_ANON_KEY", "").strip().strip("'\"")

        if not url or not key:
            print(f"Warning: Supabase credentials not found. URL: {url is not None}, KEY: {key is not None}")
            # Return predefined tables if no credentials
            return {
                "tables": [
                    {
                        "name": "processing_jobs",
                        "description": "Processing job tracking and status",
                        "type": "system"
                    },
                    {
                        "name": "musician_frame_analysis",
                        "description": "Frame-by-frame detection analysis data",
                        "type": "detection"
                    },
                    {
                        "name": "video_alignment_offset",
                        "description": "Video alignment offsets and metadata",
                        "type": "alignment"
                    },
                    {
                        "name": "chunk_video_alignment_offset",
                        "description": "Chunk video alignment data",
                        "type": "chunk_alignment"
                    },
                    {
                        "name": "transcript_video",
                        "description": "Video transcript segments",
                        "type": "transcript"
                    }
                ]
            }

        # Create Supabase client
        supabase: Client = create_client(url, key)

        # Get list of tables using raw SQL query
        result = supabase.rpc("get_tables").execute()

        # If RPC function doesn't exist, use predefined list
        tables = [
            {
                "name": "musician_frame_analysis",
                "description": "Frame-by-frame detection analysis data",
                "type": "detection"
            },
            {
                "name": "video_alignment_offset",
                "description": "Video alignment offsets and metadata",
                "type": "alignment"
            },
            {
                "name": "chunk_video_alignment_offset",
                "description": "Chunk video alignment data",
                "type": "chunk_alignment"
            },
            {
                "name": "transcript_video",
                "description": "Video transcript segments",
                "type": "transcript"
            }
        ]

        return {"tables": tables}

    except Exception as e:
        print(f"Error fetching tables: {e}")
        # Return predefined tables on error
        return {
            "tables": [
                {
                    "name": "musician_frame_analysis",
                    "description": "Frame-by-frame detection analysis data",
                    "type": "detection"
                },
                {
                    "name": "video_alignment_offset",
                    "description": "Video alignment offsets and metadata",
                    "type": "alignment"
                },
                {
                    "name": "chunk_video_alignment_offset",
                    "description": "Chunk video alignment data",
                    "type": "chunk_alignment"
                },
                {
                    "name": "transcript_video",
                    "description": "Video transcript segments",
                    "type": "transcript"
                }
            ]
        }

@router.get("/detection/summary")
async def get_detection_summary():
    """Get summary statistics for detection data"""
    try:
        db = MusicianDatabase()

        # Get basic counts (you'll need to implement these methods in MusicianDatabase)
        summary = {
            "total_frames": 0,
            "total_videos": 0,
            "detection_types": {
                "hand": 0,
                "pose": 0,
                "face": 0,
                "emotion": 0
            },
            "bad_gestures": {
                "low_wrists": 0,
                "turtle_neck": 0,
                "hunched_back": 0,
                "fingers_pointing_up": 0
            }
        }

        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/alignment/sources")
async def get_alignment_sources():
    """Get list of video sources with alignment data"""
    try:
        chunk_db = ChunkVideoAlignmentDatabase()

        # Get distinct sources from chunk alignment data
        sources = chunk_db.get_all_sources()

        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/alignment/{source}")
async def get_alignment_data(source: str):
    """Get alignment data for a specific source"""
    try:
        chunk_db = ChunkVideoAlignmentDatabase()

        alignment_data = chunk_db.get_chunk_alignments_by_source(source)

        return {
            "source": source,
            "alignment_data": alignment_data,
            "total_chunks": len(alignment_data) if alignment_data else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/detection/query")
async def query_detection_data(params: QueryParams):
    """Query detection data with filters"""
    try:
        db = MusicianDatabase()

        # Build query based on parameters
        # You'll need to implement a query method in MusicianDatabase
        results = []  # db.query_frames(limit=params.limit, offset=params.offset, ...)

        return {
            "results": results,
            "limit": params.limit,
            "offset": params.offset,
            "total": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/transcripts/{source}")
async def get_transcripts(source: str):
    """Get transcript data for a video source"""
    try:
        # You'll need to implement transcript querying
        transcripts = []

        return {
            "source": source,
            "transcripts": transcripts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.delete("/cleanup")
async def cleanup_old_data(days: int = 30):
    """Clean up old database records"""
    try:
        if days < 1:
            raise HTTPException(status_code=400, detail="Days must be at least 1")

        # Implement cleanup logic
        deleted_count = 0

        return {
            "message": f"Cleaned up records older than {days} days",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/table/{table_name}/data")
async def get_table_data(table_name: str, limit: int = 100, offset: int = 0):
    """Get data from a specific table"""
    try:
        # Get Supabase credentials from environment
        url = os.getenv("SUPABASE_URL", "").strip().strip("'\"")
        key = os.getenv("SUPABASE_ANON_KEY", "").strip().strip("'\"")

        if not url or not key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")

        # Create Supabase client
        supabase: Client = create_client(url, key)

        # Fetch data from the table
        response = supabase.table(table_name).select("*").limit(limit).offset(offset).execute()

        # Get total count
        count_response = supabase.table(table_name).select("*", count="exact", head=True).execute()
        total_count = count_response.count if hasattr(count_response, 'count') else 0

        return {
            "table_name": table_name,
            "data": response.data if response.data else [],
            "limit": limit,
            "offset": offset,
            "total_count": total_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching table data: {str(e)}")

@router.get("/table/{table_name}/columns")
async def get_table_columns(table_name: str):
    """Get column information for a specific table"""
    try:
        # Get Supabase credentials from environment
        url = os.getenv("SUPABASE_URL", "").strip().strip("'\"")
        key = os.getenv("SUPABASE_ANON_KEY", "").strip().strip("'\"")

        if not url or not key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")

        # Create Supabase client
        supabase: Client = create_client(url, key)

        # Fetch one row to get column names
        response = supabase.table(table_name).select("*").limit(1).execute()

        if response.data and len(response.data) > 0:
            columns = list(response.data[0].keys())
            return {
                "table_name": table_name,
                "columns": columns
            }
        else:
            return {
                "table_name": table_name,
                "columns": []
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching table columns: {str(e)}")

@router.get("/stats")
async def get_database_stats():
    """Get general database statistics"""
    try:
        stats = {
            "connections": {
                "musician_db": False,
                "alignment_db": False,
                "chunk_alignment_db": False
            },
            "table_sizes": {},
            "last_updated": {}
        }

        # Test connections
        try:
            db = MusicianDatabase()
            stats["connections"]["musician_db"] = True
        except:
            pass

        try:
            alignment_db = VideoAlignmentDatabase()
            stats["connections"]["alignment_db"] = True
        except:
            pass

        try:
            chunk_db = ChunkVideoAlignmentDatabase()
            stats["connections"]["chunk_alignment_db"] = True
        except:
            pass

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/heatmap")
async def get_heatmap_data(source: str, type: str = "pose", limit: int = 1000):
    """Get heatmap data for visualization"""
    try:
        # Get Supabase credentials
        url = os.getenv("SUPABASE_URL", "").strip().strip("'\"")
        key = os.getenv("SUPABASE_ANON_KEY", "").strip().strip("'\"")

        if not url or not key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")

        supabase: Client = create_client(url, key)

        # Query detection data based on type
        query = supabase.table("musician_frame_analysis").select("*")

        if source:
            query = query.eq("source", source)

        # Filter by detection type
        if type == "pose":
            query = query.not_.is_("body_pose", None)
        elif type == "hand":
            query = query.not_.is_("hand_landmarks", None)
        elif type == "face":
            query = query.not_.is_("facemesh", None)

        query = query.limit(limit).order("frame_number")
        response = query.execute()

        if not response.data:
            return {
                "timestamps": [],
                "keypoints": {},
                "frameWidth": 1920,
                "frameHeight": 1080
            }

        # Process data for heatmap visualization
        timestamps = []
        keypoints = {}

        for row in response.data:
            frame_num = row.get("frame_number", 0)
            timestamps.append(frame_num)

            # Extract keypoints based on type
            if type == "pose" and row.get("body_pose"):
                pose_data = row["body_pose"]
                if isinstance(pose_data, dict):
                    for kp_name, kp_data in pose_data.items():
                        if kp_name not in keypoints:
                            keypoints[kp_name] = {"x": [], "y": [], "confidence": []}

                        if isinstance(kp_data, dict):
                            keypoints[kp_name]["x"].append(kp_data.get("x", 0))
                            keypoints[kp_name]["y"].append(kp_data.get("y", 0))
                            keypoints[kp_name]["confidence"].append(kp_data.get("confidence", 0))

            elif type == "hand" and row.get("hand_landmarks"):
                hand_data = row["hand_landmarks"]
                if isinstance(hand_data, dict):
                    for hand_side, landmarks in hand_data.items():
                        if isinstance(landmarks, list):
                            for i, lm in enumerate(landmarks):
                                kp_name = f"{hand_side}_landmark_{i}"
                                if kp_name not in keypoints:
                                    keypoints[kp_name] = {"x": [], "y": [], "confidence": []}

                                if isinstance(lm, dict):
                                    keypoints[kp_name]["x"].append(lm.get("x", 0))
                                    keypoints[kp_name]["y"].append(lm.get("y", 0))
                                    keypoints[kp_name]["confidence"].append(1.0)

        return {
            "timestamps": timestamps,
            "keypoints": keypoints,
            "frameWidth": 1920,
            "frameHeight": 1080
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching heatmap data: {str(e)}")