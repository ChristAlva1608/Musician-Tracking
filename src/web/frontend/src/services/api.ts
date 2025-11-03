// API service for communicating with the FastAPI backend

import axios from 'axios';
import {
  SingleVideoRequest,
  FolderProcessingRequest,
  VideoUploadRequest,
  ProcessingJob,
  ModelOptions,
  Configuration,
  DirectoryListing,
  FileItem,
  DatabaseTable,
  AlignmentData
} from '../types';

export const API_BASE_URL = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 30000,
});

// Request interceptor for logging
apiClient.interceptors.request.use((config) => {
  console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
  return config;
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Processing API
export const processingApi = {
  // Process single video
  processSingleVideo: async (request: SingleVideoRequest): Promise<ProcessingJob> => {
    const response = await apiClient.post('/processing/single-video', request);
    return response.data;
  },

  // Process folder
  processFolder: async (request: FolderProcessingRequest): Promise<ProcessingJob> => {
    const response = await apiClient.post('/processing/folder', request);
    return response.data;
  },

  // Get all jobs
  getAllJobs: async (): Promise<ProcessingJob[]> => {
    const response = await apiClient.get('/processing/jobs');
    return response.data.jobs;
  },

  // Get job status
  getJobStatus: async (jobId: string): Promise<ProcessingJob> => {
    const response = await apiClient.get(`/processing/jobs/${jobId}`);
    return response.data;
  },

  // Cancel job
  cancelJob: async (jobId: string): Promise<void> => {
    await apiClient.delete(`/processing/jobs/${jobId}`);
  },

  // Delete job (permanently remove from database)
  deleteJob: async (jobId: string): Promise<void> => {
    await apiClient.delete(`/processing/jobs/${jobId}/delete`);
  },

  // Get available models
  getModels: async (): Promise<ModelOptions> => {
    const response = await apiClient.get('/processing/models');
    return response.data;
  },

  // Upload and process video
  uploadVideo: async (request: VideoUploadRequest): Promise<ProcessingJob> => {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('hand_model', request.hand_model);
    formData.append('pose_model', request.pose_model);
    formData.append('facemesh_model', request.facemesh_model);
    formData.append('emotion_model', request.emotion_model);
    formData.append('transcript_model', request.transcript_model);
    formData.append('num_poses', request.num_poses.toString());
    formData.append('num_hands', request.num_hands.toString());
    formData.append('num_faces', request.num_faces.toString());
    if (request.target_person !== null) {
      formData.append('target_person', request.target_person.toString());
    }
    formData.append('skip_frames', request.skip_frames.toString());
    formData.append('save_output_video', request.save_output_video.toString());
    formData.append('display_output', request.display_output.toString());

    const response = await apiClient.post('/processing/upload-video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 60 second timeout for uploads
    });
    return response.data;
  },
};

// Configuration API
export const configApi = {
  // Get full configuration
  getConfig: async (): Promise<Configuration> => {
    const response = await apiClient.get('/config/');
    return response.data;
  },

  // Get config section
  getConfigSection: async (section: string): Promise<any> => {
    const response = await apiClient.get(`/config/${section}`);
    return response.data;
  },

  // Update config section
  updateConfigSection: async (section: string, data: any): Promise<void> => {
    await apiClient.put(`/config/${section}`, { data });
  },

  // Update config value
  updateConfigValue: async (section: string, key: string, value: any): Promise<void> => {
    await apiClient.put(`/config/${section}/${key}`, { value });
  },

  // Get presets
  getPresets: async (): Promise<any> => {
    const response = await apiClient.get('/config/presets/list');
    return response.data;
  },

  // Apply preset
  applyPreset: async (presetName: string): Promise<void> => {
    await apiClient.post(`/config/presets/${presetName}/apply`);
  },

  // Backup config
  backupConfig: async (): Promise<{ backup_path: string }> => {
    const response = await apiClient.post('/config/backup');
    return response.data;
  },

  // Get config schema
  getConfigSchema: async (): Promise<any> => {
    const response = await apiClient.get('/config/schema');
    return response.data;
  },
};

// Files API
export const filesApi = {
  // Browse directory
  browseDirectory: async (path?: string): Promise<DirectoryListing> => {
    const response = await apiClient.get('/files/browse', {
      params: path ? { path } : {}
    });
    return response.data;
  },

  // Get video folders
  getVideoFolders: async (): Promise<{ folders: any[] }> => {
    const response = await apiClient.get('/files/video-folders');
    return response.data;
  },

  // Search files
  searchFiles: async (query: string, path?: string, fileType: string = 'video'): Promise<{ results: FileItem[] }> => {
    const response = await apiClient.get('/files/search', {
      params: { query, path, file_type: fileType }
    });
    return response.data;
  },

  // Get file info
  getFileInfo: async (path: string): Promise<FileItem> => {
    const response = await apiClient.get('/files/info', {
      params: { path }
    });
    return response.data;
  },

  // Get output files
  getOutputFiles: async (): Promise<{ output_files: FileItem[] }> => {
    const response = await apiClient.get('/files/outputs');
    return response.data;
  },

  // Download file
  downloadFile: (path: string): string => {
    return `${API_BASE_URL}/api/files/download?path=${encodeURIComponent(path)}`;
  },
};

// Database API
export const databaseApi = {
  // Get tables
  getTables: async (): Promise<{ tables: DatabaseTable[] }> => {
    const response = await apiClient.get('/database/tables');
    return response.data;
  },

  // Get detection summary
  getDetectionSummary: async (): Promise<any> => {
    const response = await apiClient.get('/database/detection/summary');
    return response.data;
  },

  // Get alignment sources
  getAlignmentSources: async (): Promise<{ sources: string[] }> => {
    const response = await apiClient.get('/database/alignment/sources');
    return response.data;
  },

  // Get alignment data
  getAlignmentData: async (source: string): Promise<AlignmentData> => {
    const response = await apiClient.get(`/database/alignment/${source}`);
    return response.data;
  },

  // Get database stats
  getDatabaseStats: async (): Promise<any> => {
    const response = await apiClient.get('/database/stats');
    return response.data;
  },

  // Cleanup old data
  cleanupOldData: async (days: number): Promise<{ deleted_count: number }> => {
    const response = await apiClient.delete('/database/cleanup', {
      params: { days }
    });
    return response.data;
  },

  // Get table data
  getTableData: async (tableName: string, limit = 100, offset = 0) => {
    const response = await apiClient.get(`/database/table/${tableName}/data`, {
      params: { limit, offset }
    });
    return response.data;
  },

  // Get table columns
  getTableColumns: async (tableName: string) => {
    const response = await apiClient.get(`/database/table/${tableName}/columns`);
    return response.data;
  },

  // Get heatmap data
  getHeatmapData: async (source: string, type: string = 'pose', limit: number = 1000) => {
    const response = await apiClient.get('/database/heatmap', {
      params: { source, type, limit }
    });
    return response.data;
  },
};

// Health check
export const healthCheck = async (): Promise<any> => {
  const response = await axios.get(`${API_BASE_URL}/`);
  return response.data;
};

export default {
  processingApi,
  configApi,
  filesApi,
  databaseApi,
  healthCheck,
};