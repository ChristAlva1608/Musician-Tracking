// Shared TypeScript types for the application

export interface VideoFile {
  name: string;
  path: string;
  size?: number;
  duration?: number;
  fps?: number;
  width?: number;
  height?: number;
}

export interface ProcessingJob {
  job_id: string; // Primary key - unique job identifier
  type: 'single_video' | 'folder_processing';
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  created_at: string;
  updated_at: string;
  output_files: string[];
  logs: LogEntry[];
  request_data?: any; // Original request data
  frame_metrics?: {
    total_frames: number;
    frames_processed: number;
    fps: number;
    avg_processing_time_ms: number;
  };
  estimated_finish_time?: string;
  estimated_seconds_remaining?: number;
}

export interface LogEntry {
  timestamp: string;
  message: string;
}

export interface SingleVideoRequest {
  video_path: string;
  skip_frames: number;
  hand_model: string;
  pose_model: string;
  facemesh_model: string;
  emotion_model: string;
  transcript_model: string;
  save_output_video: boolean;
  display_output: boolean;
}

export interface FolderProcessingRequest {
  folder_path: string;
  skip_frames: number;
  hand_model: string;
  pose_model: string;
  facemesh_model: string;
  emotion_model: string;
  transcript_model: string;
  processing_type: string;
  unified_videos: boolean;
  limit_processing_duration: boolean;
  max_processing_duration: number;
}

export interface ModelOptions {
  hand_models: string[];
  pose_models: string[];
  facemesh_models: string[];
  emotion_models: string[];
  transcript_models: string[];
  processing_types: string[];
}

export interface ConfigSection {
  [key: string]: any;
}

export interface Configuration {
  database: ConfigSection;
  video: ConfigSection;
  detection: ConfigSection;
  bad_gestures: ConfigSection;
  heatmap: ConfigSection;
  logging: ConfigSection;
  performance: ConfigSection;
  video_aligner: ConfigSection;
  integrated_processor: ConfigSection;
}

export interface FileItem {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  modified?: string;
  extension?: string;
  is_video: boolean;
  video_info?: {
    fps: number;
    frame_count: number;
    width: number;
    height: number;
    duration: number;
  };
}

export interface DirectoryListing {
  current_path: string;
  parent_path?: string;
  items: FileItem[];
}

export interface DatabaseTable {
  name: string;
  description: string;
  type: string;
}

export interface AlignmentData {
  source: string;
  alignment_data: any[];
  total_chunks: number;
}

export interface WebSocketMessage {
  type: string;
  data: any;
}