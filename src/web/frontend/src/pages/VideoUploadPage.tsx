import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  Grid,
  Paper,
  ToggleButtonGroup,
  ToggleButton,
  LinearProgress,
  Chip,
  FormControlLabel,
  Switch,
  Divider,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  CloudUpload,
  PersonAdd,
  People,
  PlayArrow,
  Info,
  Settings as SettingsIcon,
  VideoFile
} from '@mui/icons-material';
import { processingApi } from '../services/api';
import { ModelOptions, VideoUploadRequest } from '../types';
import { useNavigate } from 'react-router-dom';

export const VideoUploadPage: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string>('');
  const [models, setModels] = useState<ModelOptions | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');

  // Person selection state
  const [detectionMode, setDetectionMode] = useState<'single' | 'both'>('both');
  const [targetPerson, setTargetPerson] = useState<number | null>(null);

  // Model configuration
  const [handModel, setHandModel] = useState('mediapipe');
  const [poseModel, setPoseModel] = useState('mediapipe');
  const [facemeshModel, setFacemeshModel] = useState('yolo+mediapipe');
  const [emotionModel, setEmotionModel] = useState('none');
  const [transcriptModel, setTranscriptModel] = useState('none');

  // Processing options
  const [skipFrames, setSkipFrames] = useState(0);
  const [saveOutputVideo, setSaveOutputVideo] = useState(true);

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

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('video/')) {
        setError('Please select a valid video file');
        return;
      }
      setSelectedFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setError('');
    }
  };

  const handleDetectionModeChange = (_: React.MouseEvent<HTMLElement>, newMode: 'single' | 'both' | null) => {
    if (newMode !== null) {
      setDetectionMode(newMode);
      if (newMode === 'both') {
        setTargetPerson(null);
      } else {
        // Default to left person (0) when switching to single mode
        setTargetPerson(0);
      }
    }
  };

  const handleTargetPersonChange = (_: React.MouseEvent<HTMLElement>, person: number | null) => {
    setTargetPerson(person);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a video file');
      return;
    }

    setUploading(true);
    setError('');
    setSuccess('');

    try {
      // Calculate num_poses, num_hands, num_faces based on detection mode
      const numPoses = detectionMode === 'both' ? 2 : 1;
      const numHands = detectionMode === 'both' ? 4 : 2;
      const numFaces = detectionMode === 'both' ? 2 : 1;

      const request: VideoUploadRequest = {
        file: selectedFile,
        hand_model: handModel,
        pose_model: poseModel,
        facemesh_model: facemeshModel,
        emotion_model: emotionModel,
        transcript_model: transcriptModel,
        num_poses: numPoses,
        num_hands: numHands,
        num_faces: numFaces,
        target_person: detectionMode === 'single' ? targetPerson : null,
        skip_frames: skipFrames,
        save_output_video: saveOutputVideo,
        display_output: false
      };

      const job = await processingApi.uploadVideo(request);
      setSuccess(`Video uploaded successfully! Job ID: ${job.job_id}`);

      // Navigate to processing page after 2 seconds
      setTimeout(() => {
        navigate('/processing');
      }, 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setVideoPreview('');
    setError('');
    setSuccess('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Upload Video for Analysis
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Upload a video and choose whether to detect one specific person or multiple people
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
        {/* Video Upload Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <VideoFile color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Video Upload</Typography>
              </Box>

              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
              />

              <Box sx={{ mb: 3 }}>
                <Button
                  variant="contained"
                  component="span"
                  startIcon={<CloudUpload />}
                  onClick={() => fileInputRef.current?.click()}
                  fullWidth
                  size="large"
                  disabled={uploading}
                >
                  Choose Video File
                </Button>
              </Box>

              {selectedFile && (
                <Box sx={{ mb: 2 }}>
                  <Chip
                    label={`${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`}
                    onDelete={handleReset}
                    color="primary"
                    sx={{ mb: 2 }}
                  />

                  {videoPreview && (
                    <Box sx={{
                      width: '100%',
                      backgroundColor: '#000',
                      borderRadius: 1,
                      overflow: 'hidden'
                    }}>
                      <video
                        src={videoPreview}
                        controls
                        style={{
                          width: '100%',
                          maxHeight: '300px',
                          objectFit: 'contain'
                        }}
                      />
                    </Box>
                  )}
                </Box>
              )}

              {!selectedFile && (
                <Paper
                  sx={{
                    p: 4,
                    textAlign: 'center',
                    backgroundColor: '#f5f5f5',
                    border: '2px dashed #ccc'
                  }}
                >
                  <CloudUpload sx={{ fontSize: 60, color: '#999', mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    No video selected
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Click the button above to upload a video file
                  </Typography>
                </Paper>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Person Selection Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <People color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Person Detection</Typography>
                <Tooltip title="Choose whether to detect one specific person or multiple people in the video">
                  <IconButton size="small" sx={{ ml: 1 }}>
                    <Info fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>

              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Choose detection mode:
              </Typography>

              <ToggleButtonGroup
                value={detectionMode}
                exclusive
                onChange={handleDetectionModeChange}
                fullWidth
                sx={{ mb: 3 }}
              >
                <ToggleButton value="single">
                  <PersonAdd sx={{ mr: 1 }} />
                  Single Person
                </ToggleButton>
                <ToggleButton value="both">
                  <People sx={{ mr: 1 }} />
                  Multiple People
                </ToggleButton>
              </ToggleButtonGroup>

              {detectionMode === 'single' && (
                <Box sx={{ mb: 3, p: 2, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Select which person to track:
                  </Typography>
                  <ToggleButtonGroup
                    value={targetPerson}
                    exclusive
                    onChange={handleTargetPersonChange}
                    fullWidth
                  >
                    <ToggleButton value={0}>
                      Left Person
                    </ToggleButton>
                    <ToggleButton value={1}>
                      Right Person
                    </ToggleButton>
                  </ToggleButtonGroup>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    People are ordered left-to-right based on their position in the frame
                  </Typography>
                </Box>
              )}

              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  <strong>Single Person Mode:</strong> Faster processing, focuses on one musician
                  <br />
                  <strong>Multiple People Mode:</strong> Tracks up to 2 people simultaneously
                </Typography>
              </Alert>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                Detection Summary:
              </Typography>
              <Box sx={{ pl: 2 }}>
                <Typography variant="body2">
                  • People to detect: <strong>{detectionMode === 'both' ? '2' : '1'}</strong>
                </Typography>
                <Typography variant="body2">
                  • Hands to detect: <strong>{detectionMode === 'both' ? '4' : '2'}</strong>
                </Typography>
                <Typography variant="body2">
                  • Faces to detect: <strong>{detectionMode === 'both' ? '2' : '1'}</strong>
                </Typography>
                {detectionMode === 'single' && targetPerson !== null && (
                  <Typography variant="body2">
                    • Target: <strong>{targetPerson === 0 ? 'Left' : 'Right'} person</strong>
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Model Configuration */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SettingsIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Model Configuration</Typography>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Hand Model</InputLabel>
                    <Select
                      value={handModel}
                      label="Hand Model"
                      onChange={(e) => setHandModel(e.target.value)}
                    >
                      {models?.hand_models.map((model) => (
                        <MenuItem key={model} value={model}>
                          {model}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Pose Model</InputLabel>
                    <Select
                      value={poseModel}
                      label="Pose Model"
                      onChange={(e) => setPoseModel(e.target.value)}
                    >
                      {models?.pose_models.map((model) => (
                        <MenuItem key={model} value={model}>
                          {model}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Face Model</InputLabel>
                    <Select
                      value={facemeshModel}
                      label="Face Model"
                      onChange={(e) => setFacemeshModel(e.target.value)}
                    >
                      {models?.facemesh_models.map((model) => (
                        <MenuItem key={model} value={model}>
                          {model}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Emotion Model</InputLabel>
                    <Select
                      value={emotionModel}
                      label="Emotion Model"
                      onChange={(e) => setEmotionModel(e.target.value)}
                    >
                      {models?.emotion_models.map((model) => (
                        <MenuItem key={model} value={model}>
                          {model}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Transcript Model</InputLabel>
                    <Select
                      value={transcriptModel}
                      label="Transcript Model"
                      onChange={(e) => setTranscriptModel(e.target.value)}
                    >
                      {models?.transcript_models.map((model) => (
                        <MenuItem key={model} value={model}>
                          {model}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Skip Frames"
                    type="number"
                    value={skipFrames}
                    onChange={(e) => setSkipFrames(parseInt(e.target.value) || 0)}
                    inputProps={{ min: 0, max: 10 }}
                  />
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={saveOutputVideo}
                        onChange={(e) => setSaveOutputVideo(e.target.checked)}
                      />
                    }
                    label="Save Output Video"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<PlayArrow />}
                  onClick={handleUpload}
                  disabled={!selectedFile || uploading}
                  sx={{ minWidth: 200 }}
                >
                  {uploading ? 'Uploading...' : 'Upload & Process'}
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={handleReset}
                  disabled={!selectedFile || uploading}
                >
                  Reset
                </Button>
              </Box>
              {uploading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" align="center" sx={{ mt: 1 }}>
                    Uploading video and starting processing...
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};
