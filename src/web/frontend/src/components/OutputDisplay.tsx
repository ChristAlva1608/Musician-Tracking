import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  LinearProgress,
  Paper,
  IconButton,
  Chip
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';
import { processingApi } from '../services/api';

interface OutputDisplayProps {
  jobId: string | null;
  open: boolean;
  onClose: () => void;
}

export const OutputDisplay: React.FC<OutputDisplayProps> = ({ jobId, open, onClose }) => {
  const [jobData, setJobData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    if (!open || !jobId) return;

    const fetchJobStatus = async () => {
      try {
        setLoading(true);
        const data = await processingApi.getJobStatus(jobId);
        setJobData(data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch job status');
      } finally {
        setLoading(false);
      }
    };

    fetchJobStatus();

    // Auto-refresh while job is running
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(() => {
        if (jobData?.status === 'running' || jobData?.status === 'queued') {
          fetchJobStatus();
        }
      }, 2000); // Refresh every 2 seconds
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId, open, autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'primary';
      case 'failed': return 'error';
      case 'cancelled': return 'warning';
      default: return 'default';
    }
  };

  const formatLog = (log: any) => {
    if (typeof log === 'string') return log;
    return `[${log.timestamp}] ${log.message}`;
  };

  if (!open) return null;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { minHeight: '60vh', maxHeight: '80vh' }
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h6">Processing Output</Typography>
          {jobData && (
            <Chip
              label={jobData.status}
              color={getStatusColor(jobData.status) as any}
              size="small"
            />
          )}
        </Box>
        <Box>
          <IconButton
            onClick={() => setAutoRefresh(!autoRefresh)}
            color={autoRefresh ? 'primary' : 'default'}
            title={autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          >
            <RefreshIcon />
          </IconButton>
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {loading && !jobData && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <Typography>Loading job details...</Typography>
          </Box>
        )}

        {error && (
          <Box sx={{ mb: 2 }}>
            <Typography color="error">{error}</Typography>
          </Box>
        )}

        {jobData && (
          <>
            {/* Job Information */}
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Job Information</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                <Typography variant="body2">
                  <strong>Job ID:</strong> {jobData.job_id || jobId}
                </Typography>
                <Typography variant="body2">
                  <strong>Type:</strong> {jobData.type}
                </Typography>
                <Typography variant="body2">
                  <strong>Created:</strong> {new Date(jobData.created_at).toLocaleString()}
                </Typography>
                <Typography variant="body2">
                  <strong>Status:</strong> {jobData.status}
                </Typography>
              </Box>
            </Paper>

            {/* Progress Bar */}
            {jobData.progress !== undefined && (
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2">Progress</Typography>
                  <Typography variant="body2">{jobData.progress}%</Typography>
                </Box>
                <LinearProgress variant="determinate" value={jobData.progress} />
                {jobData.message && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {jobData.message}
                  </Typography>
                )}
                {jobData.frame_metrics && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2">
                      Frames: {jobData.frame_metrics.frames_processed} / {jobData.frame_metrics.total_frames}
                    </Typography>
                    <Typography variant="body2">
                      Avg Processing Time: {jobData.frame_metrics.avg_processing_time_ms?.toFixed(1)}ms per frame
                    </Typography>
                    {jobData.estimated_finish_time && (
                      <Typography variant="body2" color="primary">
                        Expected Finish: {new Date(jobData.estimated_finish_time).toLocaleTimeString()}
                        {jobData.estimated_seconds_remaining && (
                          <span> ({Math.ceil(jobData.estimated_seconds_remaining / 60)} min remaining)</span>
                        )}
                      </Typography>
                    )}
                  </Box>
                )}
              </Box>
            )}

            {/* Processing Logs */}
            {jobData.logs && jobData.logs.length > 0 && (
              <Paper sx={{ p: 2, mb: 2, maxHeight: 300, overflow: 'auto' }}>
                <Typography variant="subtitle2" gutterBottom>Processing Logs</Typography>
                <Box
                  component="pre"
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    m: 0
                  }}
                >
                  {jobData.logs.map((log: any, index: number) => (
                    <div key={index}>{formatLog(log)}</div>
                  ))}
                </Box>
              </Paper>
            )}

            {/* Output Files */}
            {jobData.output_files && jobData.output_files.length > 0 && (
              <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>Output Files</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {/* Group files by type */}
                  {['unified', 'annotated', 'aligned', 'report'].map(fileType => {
                    const filesOfType = jobData.output_files.filter((file: any) =>
                      typeof file === 'object' ? file.type === fileType : fileType === 'report'
                    );

                    if (filesOfType.length === 0) return null;

                    return (
                      <Box key={fileType}>
                        <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
                          {fileType === 'unified' ? 'Unified Video (All Cameras)' :
                           fileType === 'annotated' ? 'Annotated Detection Videos' :
                           fileType === 'aligned' ? 'Aligned Videos' : 'Reports'}
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 0.5 }}>
                          {filesOfType.map((file: any, index: number) => {
                            const isObject = typeof file === 'object';
                            const filePath = isObject ? file.path : file;
                            const fileName = isObject ? file.name : filePath.split('/').pop();
                            const camera = isObject ? file.camera : '';

                            return (
                              <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="body2" sx={{ flex: 1 }}>
                                  {camera && camera !== 'all' && camera !== 'report' ? `Camera ${camera}: ` : ''}
                                  {fileName}
                                </Typography>
                                <Button
                                  size="small"
                                  variant="outlined"
                                  href={`http://localhost:8000/api/files/download?path=${encodeURIComponent(filePath)}`}
                                  target="_blank"
                                >
                                  Download
                                </Button>
                              </Box>
                            );
                          })}
                        </Box>
                      </Box>
                    );
                  })}
                </Box>
              </Paper>
            )}

            {/* Request Data (for debugging) */}
            {jobData.request_data && (
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>Request Parameters</Typography>
                <Box
                  component="pre"
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    m: 0,
                    maxHeight: 200,
                    overflow: 'auto'
                  }}
                >
                  {JSON.stringify(jobData.request_data, null, 2)}
                </Box>
              </Paper>
            )}
          </>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        {jobData?.status === 'running' && (
          <Button
            variant="contained"
            color="error"
            onClick={async () => {
              if (jobId) {
                await processingApi.cancelJob(jobId);
                onClose();
              }
            }}
          >
            Cancel Job
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};