import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Box,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  Chip
} from '@mui/material';
import {
  VideoFile as VideoFileIcon,
  Folder,
  PlayArrow,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { processingApi, filesApi } from '../services/api';
import { ModelOptions, SingleVideoRequest, FolderProcessingRequest } from '../types';
import { FileBrowser } from '../components/FileBrowser';
import { ProcessingForm } from '../components/ProcessingForm';
import { OutputDisplay } from '../components/OutputDisplay';

export const HomePage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [selectedFolder, setSelectedFolder] = useState<string>('');
  const [models, setModels] = useState<ModelOptions | null>(null);
  const [showFileBrowser, setShowFileBrowser] = useState(false);
  const [browserMode, setBrowserMode] = useState<'file' | 'folder'>('file');
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [showOutputDisplay, setShowOutputDisplay] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const modelOptions = await processingApi.getModels();
      setModels(modelOptions);
    } catch (err) {
      setError('Failed to load model options');
    }
  };

  const handleFileSelect = (path: string) => {
    setSelectedFile(path);
    setShowFileBrowser(false);
  };

  const handleFolderSelect = (path: string) => {
    setSelectedFolder(path);
    setShowFileBrowser(false);
  };

  const handleSingleVideoSubmit = async (formData: SingleVideoRequest) => {
    try {
      setError('');
      const job = await processingApi.processSingleVideo(formData);
      setSuccess(`Processing job created: ${job.job_id}`);
      setCurrentJobId(job.job_id);

      // Show output display if display_output is enabled
      if (formData.display_output) {
        setShowOutputDisplay(true);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start processing');
    }
  };

  const handleFolderSubmit = async (formData: FolderProcessingRequest) => {
    try {
      setError('');
      const job = await processingApi.processFolder(formData);
      setSuccess(`Folder processing job created: ${job.job_id}`);
      setCurrentJobId(job.job_id);
      setShowOutputDisplay(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start processing');
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Musician Tracking System
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Analyze musician posture and gestures using computer vision
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccess('')}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Single Video Processing */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <VideoFileIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h5">Single Video Analysis</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Process a single video file with configurable detection models
              </Typography>

              <TextField
                fullWidth
                label="Video File Path"
                value={selectedFile}
                onChange={(e) => setSelectedFile(e.target.value)}
                placeholder="Select or enter video file path"
                sx={{ mb: 2 }}
                InputProps={{
                  readOnly: true,
                }}
              />

              <Button
                variant="outlined"
                startIcon={<Folder />}
                onClick={() => {
                  setBrowserMode('file');
                  setShowFileBrowser(true);
                }}
                sx={{ mb: 2 }}
              >
                Browse Files
              </Button>

              {selectedFile && (
                <Box sx={{ mt: 2 }}>
                  <Chip
                    label={selectedFile.split('/').pop()}
                    onDelete={() => setSelectedFile('')}
                    sx={{ mb: 2 }}
                  />

                  {models && (
                    <ProcessingForm
                      type="single_video"
                      models={models}
                      initialData={{ video_path: selectedFile }}
                      onSubmit={handleSingleVideoSubmit}
                    />
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Folder Processing */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Folder color="primary" sx={{ mr: 1 }} />
                <Typography variant="h5">Folder Processing</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Process multiple videos in a folder with alignment and unified output
              </Typography>

              <TextField
                fullWidth
                label="Folder Path"
                value={selectedFolder}
                onChange={(e) => setSelectedFolder(e.target.value)}
                placeholder="Select or enter folder path"
                sx={{ mb: 2 }}
                InputProps={{
                  readOnly: true,
                }}
              />

              <Button
                variant="outlined"
                startIcon={<Folder />}
                onClick={() => {
                  setBrowserMode('folder');
                  setShowFileBrowser(true);
                }}
                sx={{ mb: 2 }}
              >
                Browse Folders
              </Button>

              {selectedFolder && (
                <Box sx={{ mt: 2 }}>
                  <Chip
                    label={selectedFolder.split('/').pop()}
                    onDelete={() => setSelectedFolder('')}
                    sx={{ mb: 2 }}
                  />

                  {models && (
                    <ProcessingForm
                      type="folder_processing"
                      models={models}
                      initialData={{ folder_path: selectedFolder }}
                      onSubmit={handleFolderSubmit}
                    />
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="outlined"
                  startIcon={<PlayArrow />}
                  href="/processing"
                >
                  View Processing Status
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<SettingsIcon />}
                  href="/settings"
                >
                  Configure Settings
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* File Browser Modal */}
      {showFileBrowser && (
        <FileBrowser
          open={showFileBrowser}
          mode={browserMode}
          onClose={() => setShowFileBrowser(false)}
          onSelect={browserMode === 'file' ? handleFileSelect : handleFolderSelect}
        />
      )}

      {/* Output Display Modal */}
      <OutputDisplay
        jobId={currentJobId}
        open={showOutputDisplay}
        onClose={() => setShowOutputDisplay(false)}
      />
    </Container>
  );
};