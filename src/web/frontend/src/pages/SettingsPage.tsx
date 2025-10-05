import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Tabs,
  Tab,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Grid,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  IconButton
} from '@mui/material';
import {
  ExpandMore,
  Save,
  Refresh,
  Backup,
  Restore
} from '@mui/icons-material';
import { configApi } from '../services/api';
import { Configuration } from '../types';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index}>
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

export const SettingsPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [config, setConfig] = useState<Configuration | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [presets, setPresets] = useState<any>(null);

  useEffect(() => {
    loadConfig();
    loadPresets();
  }, []);

  const loadConfig = async () => {
    setLoading(true);
    try {
      const configData = await configApi.getConfig();
      setConfig(configData);
    } catch (err: any) {
      setError('Failed to load configuration');
    } finally {
      setLoading(false);
    }
  };

  const loadPresets = async () => {
    try {
      const presetsData = await configApi.getPresets();
      setPresets(presetsData);
    } catch (err) {
      console.error('Failed to load presets');
    }
  };

  const handleConfigChange = (section: string, key: string, value: any) => {
    if (!config) return;

    setConfig(prev => ({
      ...prev!,
      [section]: {
        ...prev![section as keyof Configuration],
        [key]: value
      }
    }));
  };

  const saveConfig = async () => {
    if (!config) return;

    try {
      setError('');
      // Save each section
      for (const [section, data] of Object.entries(config)) {
        await configApi.updateConfigSection(section, data);
      }
      setSuccess('Configuration saved successfully');
    } catch (err: any) {
      setError('Failed to save configuration');
    }
  };

  const applyPreset = async (presetName: string) => {
    try {
      setError('');
      await configApi.applyPreset(presetName);
      await loadConfig(); // Reload config
      setSuccess(`Preset "${presetName}" applied successfully`);
    } catch (err: any) {
      setError('Failed to apply preset');
    }
  };

  const backupConfig = async () => {
    try {
      const result = await configApi.backupConfig();
      setSuccess(`Configuration backed up: ${result.backup_path}`);
    } catch (err: any) {
      setError('Failed to backup configuration');
    }
  };

  if (!config) {
    return (
      <Container>
        <Typography>Loading configuration...</Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Settings
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Configure detection models, processing parameters, and system settings
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

      <Paper>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
            <Tab label="Detection Models" />
            <Tab label="Video Processing" />
            <Tab label="Database" />
            <Tab label="Performance" />
            <Tab label="Presets" />
          </Tabs>
        </Box>

        {/* Detection Models Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Hand Model"
                select
                SelectProps={{ native: true }}
                value={config.detection.hand_model}
                onChange={(e) => handleConfigChange('detection', 'hand_model', e.target.value)}
              >
                <option value="mediapipe">MediaPipe</option>
                <option value="yolo">YOLO</option>
              </TextField>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Pose Model"
                select
                SelectProps={{ native: true }}
                value={config.detection.pose_model}
                onChange={(e) => handleConfigChange('detection', 'pose_model', e.target.value)}
              >
                <option value="mediapipe">MediaPipe</option>
                <option value="yolo">YOLO</option>
              </TextField>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Face Mesh Model"
                select
                SelectProps={{ native: true }}
                value={config.detection.facemesh_model}
                onChange={(e) => handleConfigChange('detection', 'facemesh_model', e.target.value)}
              >
                <option value="mediapipe">MediaPipe</option>
                <option value="yolo+mediapipe">YOLO + MediaPipe</option>
                <option value="yolo">YOLO</option>
                <option value="none">None</option>
              </TextField>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Emotion Model"
                select
                SelectProps={{ native: true }}
                value={config.detection.emotion_model}
                onChange={(e) => handleConfigChange('detection', 'emotion_model', e.target.value)}
              >
                <option value="deepface">DeepFace</option>
                <option value="ghostfacenet">GhostFaceNet</option>
                <option value="fer">FER</option>
                <option value="mediapipe">MediaPipe</option>
                <option value="none">None</option>
              </TextField>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Video Processing Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Skip Frames"
                value={config.video.skip_frames}
                onChange={(e) => handleConfigChange('video', 'skip_frames', parseInt(e.target.value))}
                helperText="0 = process all frames, higher = faster processing"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.video.display_output}
                    onChange={(e) => handleConfigChange('video', 'display_output', e.target.checked)}
                  />
                }
                label="Display Output Window"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.video.save_output_video}
                    onChange={(e) => handleConfigChange('video', 'save_output_video', e.target.checked)}
                  />
                }
                label="Save Output Video"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.video.preserve_audio}
                    onChange={(e) => handleConfigChange('video', 'preserve_audio', e.target.checked)}
                  />
                }
                label="Preserve Audio"
              />
            </Grid>
          </Grid>
        </TabPanel>

        {/* Database Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.database.enabled}
                    onChange={(e) => handleConfigChange('database', 'enabled', e.target.checked)}
                  />
                }
                label="Enable Database Storage"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Batch Size"
                value={config.database.batch_size}
                onChange={(e) => handleConfigChange('database', 'batch_size', parseInt(e.target.value))}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                Select Tables to Store Data:
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.database.store_musician_frame_analysis ?? true}
                      onChange={(e) => handleConfigChange('database', 'store_musician_frame_analysis', e.target.checked)}
                      disabled={!config.database.enabled}
                    />
                  }
                  label="musician_frame_analysis - Frame-by-frame detection data"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.database.store_chunk_video_alignment ?? true}
                      onChange={(e) => handleConfigChange('database', 'store_chunk_video_alignment', e.target.checked)}
                      disabled={!config.database.enabled}
                    />
                  }
                  label="chunk_video_alignment_offset - Video alignment offsets"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.database.store_transcript_video ?? true}
                      onChange={(e) => handleConfigChange('database', 'store_transcript_video', e.target.checked)}
                      disabled={!config.database.enabled}
                    />
                  }
                  label="transcript_video - Video transcriptions"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.database.store_processing_jobs ?? true}
                      onChange={(e) => handleConfigChange('database', 'store_processing_jobs', e.target.checked)}
                      disabled={!config.database.enabled}
                    />
                  }
                  label="processing_jobs - Processing job tracking"
                />
              </Box>
              {!config.database.enabled && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  Enable database storage to select tables
                </Typography>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Performance Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Max Workers"
                value={config.performance.max_workers}
                onChange={(e) => handleConfigChange('performance', 'max_workers', parseInt(e.target.value))}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.performance.gpu_enabled}
                    onChange={(e) => handleConfigChange('performance', 'gpu_enabled', e.target.checked)}
                  />
                }
                label="GPU Enabled"
              />
            </Grid>
          </Grid>
        </TabPanel>

        {/* Presets Tab */}
        <TabPanel value={tabValue} index={4}>
          {presets?.presets.map((preset: any) => (
            <Accordion key={preset.name}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="h6">{preset.name}</Typography>
                  <Chip label="Preset" size="small" />
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {preset.description}
                </Typography>
                <Button
                  variant="contained"
                  onClick={() => applyPreset(preset.name)}
                >
                  Apply Preset
                </Button>
              </AccordionDetails>
            </Accordion>
          ))}
        </TabPanel>

        {/* Action Buttons */}
        <Box sx={{ p: 3, borderTop: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                startIcon={<Save />}
                onClick={saveConfig}
              >
                Save Configuration
              </Button>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={loadConfig}
              >
                Reload
              </Button>
            </Box>

            <Button
              variant="outlined"
              startIcon={<Backup />}
              onClick={backupConfig}
            >
              Backup Config
            </Button>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
};