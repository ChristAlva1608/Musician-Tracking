"""
Configuration management endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import yaml
import os
import sys

# Add project src directory to path
# From routers/config.py -> routers -> backend -> web -> src
routers_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(routers_dir)
web_dir = os.path.dirname(backend_dir)
src_root = os.path.dirname(web_dir)
sys.path.insert(0, src_root)

router = APIRouter()

CONFIG_PATH = os.path.join(src_root, "config", "config_v1.yaml")

class ConfigUpdate(BaseModel):
    section: str
    key: str
    value: Any

class ConfigSection(BaseModel):
    data: Dict[str, Any]

def load_config() -> Dict[str, Any]:
    """Load current configuration"""
    try:
        with open(CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file"""
    try:
        with open(CONFIG_PATH, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")

@router.get("/")
async def get_full_config():
    """Get the complete configuration"""
    return load_config()

@router.get("/{section}")
async def get_config_section(section: str):
    """Get a specific configuration section"""
    config = load_config()
    if section not in config:
        raise HTTPException(status_code=404, detail=f"Configuration section '{section}' not found")
    return {section: config[section]}

@router.put("/{section}")
async def update_config_section(section: str, section_data: ConfigSection):
    """Update an entire configuration section"""
    config = load_config()
    config[section] = section_data.data
    save_config(config)
    return {"message": f"Configuration section '{section}' updated successfully"}

@router.put("/{section}/{key}")
async def update_config_value(section: str, key: str, update: ConfigUpdate):
    """Update a specific configuration value"""
    config = load_config()

    if section not in config:
        raise HTTPException(status_code=404, detail=f"Configuration section '{section}' not found")

    # Handle nested keys (e.g., "video.source_path")
    keys = key.split('.')
    current = config[section]

    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set the value
    current[keys[-1]] = update.value

    save_config(config)
    return {"message": f"Configuration value '{section}.{key}' updated successfully"}

@router.get("/presets/list")
async def get_config_presets():
    """Get available configuration presets"""
    return {
        "presets": [
            {
                "name": "Quick Detection",
                "description": "Fast processing with basic models",
                "config": {
                    "video": {"skip_frames": 2},
                    "detection": {
                        "hand_model": "mediapipe",
                        "pose_model": "mediapipe",
                        "facemesh_model": "none",
                        "emotion_model": "none"
                    }
                }
            },
            {
                "name": "Full Analysis",
                "description": "Complete analysis with all models",
                "config": {
                    "video": {"skip_frames": 0},
                    "detection": {
                        "hand_model": "yolo",
                        "pose_model": "yolo",
                        "facemesh_model": "yolo+mediapipe",
                        "emotion_model": "deepface",
                        "transcript_model": "whisper"
                    }
                }
            },
            {
                "name": "Performance Testing",
                "description": "Short duration for testing",
                "config": {
                    "integrated_processor": {
                        "limit_processing_duration": True,
                        "max_processing_duration": 10.0
                    }
                }
            }
        ]
    }

@router.post("/presets/{preset_name}/apply")
async def apply_config_preset(preset_name: str):
    """Apply a configuration preset"""
    presets_response = await get_config_presets()
    presets = presets_response["presets"]

    preset = next((p for p in presets if p["name"] == preset_name), None)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

    config = load_config()

    # Apply preset configuration
    for section, values in preset["config"].items():
        if section not in config:
            config[section] = {}
        config[section].update(values)

    save_config(config)
    return {"message": f"Preset '{preset_name}' applied successfully"}

@router.post("/backup")
async def backup_config():
    """Create a backup of current configuration"""
    config = load_config()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CONFIG_PATH.replace(".yaml", f"_backup_{timestamp}.yaml")

    try:
        with open(backup_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        return {"message": "Configuration backed up successfully", "backup_path": backup_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to backup config: {str(e)}")

@router.post("/restore/{backup_name}")
async def restore_config(backup_name: str):
    """Restore configuration from backup"""
    backup_path = CONFIG_PATH.replace(".yaml", f"_backup_{backup_name}.yaml")

    if not os.path.exists(backup_path):
        raise HTTPException(status_code=404, detail="Backup file not found")

    try:
        with open(backup_path, 'r') as file:
            config = yaml.safe_load(file)
        save_config(config)
        return {"message": "Configuration restored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore config: {str(e)}")

@router.get("/schema")
async def get_config_schema():
    """Get configuration schema for frontend validation"""
    return {
        "database": {
            "enabled": {"type": "boolean", "description": "Enable database storage"},
            "table_name": {"type": "string", "description": "Database table name"},
            "batch_size": {"type": "integer", "min": 1, "max": 1000, "description": "Batch size for database operations"}
        },
        "video": {
            "source_path": {"type": "string", "description": "Path to video file"},
            "use_webcam": {"type": "boolean", "description": "Use webcam instead of file"},
            "skip_frames": {"type": "integer", "min": 0, "max": 10, "description": "Skip every N frames"},
            "display_output": {"type": "boolean", "description": "Display output window"}
        },
        "detection": {
            "hand_model": {"type": "enum", "options": ["mediapipe", "yolo"], "description": "Hand detection model"},
            "pose_model": {"type": "enum", "options": ["mediapipe", "yolo"], "description": "Pose detection model"},
            "facemesh_model": {"type": "enum", "options": ["mediapipe", "yolo+mediapipe", "yolo", "none"], "description": "Face mesh model"},
            "emotion_model": {"type": "enum", "options": ["deepface", "ghostfacenet", "fer", "mediapipe", "none"], "description": "Emotion detection model"}
        }
    }