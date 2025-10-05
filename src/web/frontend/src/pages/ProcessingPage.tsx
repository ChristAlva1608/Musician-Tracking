import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Button,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Grid
} from '@mui/material';
import { Refresh, Cancel, Download, Delete, Visibility } from '@mui/icons-material';
import { processingApi } from '../services/api';
import { ProcessingJob } from '../types';
import { useWebSocket } from '../hooks/useWebSocket';
import { OutputDisplay } from '../components/OutputDisplay';

export const ProcessingPage: React.FC = () => {
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [outputDisplayOpen, setOutputDisplayOpen] = useState(false);
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    loadJobs();
  }, []);

  useEffect(() => {
    // Update jobs from WebSocket messages
    if (lastMessage?.type === 'job_update') {
      const updatedJob = lastMessage.data;
      console.log('Received job update:', updatedJob);

      setJobs(prev => {
        // Check if job exists in current list
        const jobExists = prev.some(job => job.job_id === updatedJob.job_id);

        if (jobExists) {
          // Update existing job
          return prev.map(job =>
            job.job_id === updatedJob.job_id
              ? { ...job, ...updatedJob }
              : job
          );
        } else {
          // Add new job if it doesn't exist
          console.log('Adding new job to list:', updatedJob);
          return [updatedJob, ...prev];
        }
      });
    }
  }, [lastMessage]);

  const loadJobs = async () => {
    setLoading(true);
    try {
      const jobList = await processingApi.getAllJobs();
      setJobs(jobList);
    } catch (err) {
      console.error('Failed to load jobs');
    } finally {
      setLoading(false);
    }
  };

  const cancelJob = async (jobId: string) => {
    try {
      await processingApi.cancelJob(jobId);
      await loadJobs();
    } catch (err) {
      console.error('Failed to cancel job');
    }
  };

  const deleteJob = async (jobId: string) => {
    if (window.confirm('Are you sure you want to delete this job?')) {
      try {
        await processingApi.deleteJob(jobId);
        await loadJobs();
      } catch (err) {
        console.error('Failed to delete job');
      }
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'running': return 'primary';
      case 'cancelled': return 'default';
      default: return 'default';
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <Typography variant="h4" component="h1" gutterBottom>
            Processing Status
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Monitor and manage video processing jobs
          </Typography>
        </div>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={loadJobs}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      <Grid container spacing={3}>
        {jobs.map((job) => (
          <Grid item xs={12} key={job.job_id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    {job.type === 'single_video' ? 'Single Video' : 'Folder Processing'}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <Chip
                      label={job.status}
                      color={getStatusColor(job.status) as any}
                      size="small"
                    />
                    <IconButton
                      size="small"
                      onClick={() => {
                        setSelectedJobId(job.job_id);
                        setOutputDisplayOpen(true);
                      }}
                      color="primary"
                      title="View Details"
                    >
                      <Visibility />
                    </IconButton>
                    {job.status === 'running' && (
                      <IconButton
                        size="small"
                        onClick={() => cancelJob(job.job_id)}
                        color="error"
                        title="Cancel Job"
                      >
                        <Cancel />
                      </IconButton>
                    )}
                    {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
                      <IconButton
                        size="small"
                        onClick={() => deleteJob(job.job_id)}
                        color="error"
                        title="Delete Job"
                      >
                        <Delete />
                      </IconButton>
                    )}
                  </Box>
                </Box>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Job ID: {job.job_id}
                </Typography>

                <Typography variant="body2" sx={{ mb: 2 }}>
                  {job.message}
                </Typography>

                {job.status === 'running' && (
                  <Box sx={{ mb: 2 }}>
                    <LinearProgress
                      variant="determinate"
                      value={job.progress}
                    />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                      <Typography variant="caption" color="text.secondary">
                        {job.progress}% complete
                      </Typography>
                      {job.estimated_finish_time && (
                        <Typography variant="caption" color="primary">
                          ETA: {new Date(job.estimated_finish_time).toLocaleTimeString()}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                )}

                {job.output_files && job.output_files.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Output Files ({job.output_files.length}):
                    </Typography>
                    <List dense>
                      {job.output_files.slice(0, 3).map((file: any, index: number) => {
                        const isObject = typeof file === 'object';
                        const filePath = isObject ? file.path : file;
                        const fileName = isObject ? file.name : filePath.split('/').pop();
                        const fileType = isObject ? file.type : 'file';

                        return (
                          <ListItem key={index} sx={{ pl: 0 }}>
                            <ListItemText
                              primary={fileName}
                              secondary={fileType}
                            />
                            <IconButton
                              size="small"
                              href={`http://localhost:8000/api/files/download?path=${encodeURIComponent(filePath)}`}
                              target="_blank"
                            >
                              <Download />
                            </IconButton>
                          </ListItem>
                        );
                      })}
                      {job.output_files.length > 3 && (
                        <Typography variant="caption" color="text.secondary">
                          ...and {job.output_files.length - 3} more files
                        </Typography>
                      )}
                    </List>
                  </Box>
                )}

                <Typography variant="caption" color="text.secondary">
                  Created: {new Date(job.created_at).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}

        {jobs.length === 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="body1" color="text.secondary" textAlign="center">
                  No processing jobs found. Start a new job from the Home page.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Output Display Modal */}
      <OutputDisplay
        jobId={selectedJobId}
        open={outputDisplayOpen}
        onClose={() => {
          setOutputDisplayOpen(false);
          setSelectedJobId(null);
        }}
      />
    </Container>
  );
};