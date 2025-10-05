import React, { useState } from 'react';
import {
  Box,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Grid,
  Typography,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  Chip
} from '@mui/material';
import { ExpandMore, PlayArrow } from '@mui/icons-material';
import { ModelOptions, SingleVideoRequest, FolderProcessingRequest } from '../types';

interface ProcessingFormProps {
  type: 'single_video' | 'folder_processing';
  models: ModelOptions;
  initialData: Partial<SingleVideoRequest | FolderProcessingRequest>;
  onSubmit: (data: any) => void;
}

export const ProcessingForm: React.FC<ProcessingFormProps> = ({
  type,
  models,
  initialData,
  onSubmit
}) => {
  const [formData, setFormData] = useState({
    skip_frames: 0,
    hand_model: 'mediapipe',
    pose_model: 'mediapipe',
    facemesh_model: 'yolo+mediapipe',
    emotion_model: 'none',
    transcript_model: 'whisper',
    save_output_video: true,
    display_output: false,
    processing_type: 'full_frames',
    unified_videos: true,
    limit_processing_duration: false,
    max_processing_duration: 10.0,
    ...initialData
  });

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = () => {
    onSubmit(formData);
  };

  return (
    <Box>
      <Grid container spacing={2}>
        {/* Performance Settings */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Performance Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography gutterBottom>Skip Frames: {formData.skip_frames}</Typography>
                  <Slider
                    value={formData.skip_frames}
                    onChange={(_, value) => handleInputChange('skip_frames', value)}
                    min={0}
                    max={10}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                  />
                  <Typography variant="caption" color="text.secondary">
                    0 = process all frames, higher = faster but less detailed
                  </Typography>
                </Grid>

                {type === 'folder_processing' && (
                  <>
                    <Grid item xs={12} sm={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={formData.limit_processing_duration}
                            onChange={(e) => handleInputChange('limit_processing_duration', e.target.checked)}
                          />
                        }
                        label="Limit Processing Duration"
                      />
                      {formData.limit_processing_duration && (
                        <TextField
                          fullWidth
                          type="number"
                          label="Max Duration (seconds)"
                          value={formData.max_processing_duration}
                          onChange={(e) => handleInputChange('max_processing_duration', parseFloat(e.target.value))}
                          sx={{ mt: 1 }}
                        />
                      )}
                    </Grid>
                  </>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Model Selection */}
        <Grid item xs={12}>
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Detection Models</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Hand Model</InputLabel>
                    <Select
                      value={formData.hand_model}
                      onChange={(e) => handleInputChange('hand_model', e.target.value)}
                    >
                      {models.hand_models.map(model => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Pose Model</InputLabel>
                    <Select
                      value={formData.pose_model}
                      onChange={(e) => handleInputChange('pose_model', e.target.value)}
                    >
                      {models.pose_models.map(model => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Face Mesh Model</InputLabel>
                    <Select
                      value={formData.facemesh_model}
                      onChange={(e) => handleInputChange('facemesh_model', e.target.value)}
                    >
                      {models.facemesh_models.map(model => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Emotion Model</InputLabel>
                    <Select
                      value={formData.emotion_model}
                      onChange={(e) => handleInputChange('emotion_model', e.target.value)}
                    >
                      {models.emotion_models.map(model => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Transcript Model</InputLabel>
                    <Select
                      value={formData.transcript_model}
                      onChange={(e) => handleInputChange('transcript_model', e.target.value)}
                    >
                      {models.transcript_models.map(model => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Output Settings */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Output Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {type === 'single_video' && (
                  <>
                    <Grid item xs={12} sm={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={formData.save_output_video}
                            onChange={(e) => handleInputChange('save_output_video', e.target.checked)}
                          />
                        }
                        label="Save Output Video"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={formData.display_output}
                            onChange={(e) => handleInputChange('display_output', e.target.checked)}
                          />
                        }
                        label="Display Output Window"
                      />
                    </Grid>
                  </>
                )}

                {type === 'folder_processing' && (
                  <>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Processing Type</InputLabel>
                        <Select
                          value={formData.processing_type}
                          onChange={(e) => handleInputChange('processing_type', e.target.value)}
                        >
                          {models.processing_types.map(type => (
                            <MenuItem key={type} value={type}>{type}</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={formData.unified_videos}
                            onChange={(e) => handleInputChange('unified_videos', e.target.checked)}
                          />
                        }
                        label="Create Unified Videos"
                      />
                    </Grid>
                  </>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Submit Button */}
        <Grid item xs={12}>
          <Button
            variant="contained"
            size="large"
            startIcon={<PlayArrow />}
            onClick={handleSubmit}
            fullWidth
          >
            Start Processing
          </Button>
        </Grid>

        {/* Model Summary */}
        <Grid item xs={12}>
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Selected Models:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip label={`Hand: ${formData.hand_model}`} size="small" />
              <Chip label={`Pose: ${formData.pose_model}`} size="small" />
              <Chip label={`Face: ${formData.facemesh_model}`} size="small" />
              <Chip label={`Emotion: ${formData.emotion_model}`} size="small" />
              <Chip label={`Transcript: ${formData.transcript_model}`} size="small" />
            </Box>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};