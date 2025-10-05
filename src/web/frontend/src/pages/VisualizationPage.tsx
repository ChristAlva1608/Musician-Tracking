import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Tabs,
  Tab,
  Alert
} from '@mui/material';
import { HeatmapVisualization } from '../components/HeatmapVisualization';
import { databaseApi } from '../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`visualization-tabpanel-${index}`}
      aria-labelledby={`visualization-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export const VisualizationPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [sources, setSources] = useState<string[]>([]);
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [detectionType, setDetectionType] = useState<'pose' | 'hand' | 'face'>('pose');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    loadSources();
  }, []);

  const loadSources = async () => {
    try {
      setLoading(true);
      const response = await databaseApi.getAlignmentSources();
      setSources(response.sources || []);
      if (response.sources && response.sources.length > 0) {
        setSelectedSource(response.sources[0]);
      }
    } catch (err) {
      setError('Failed to load video sources');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Data Visualization
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Visualize detection data with heatmaps and analytics
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="visualization tabs">
            <Tab label="Heatmap Visualization" />
            <Tab label="Statistics" />
            <Tab label="Timeline" />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2, mb: 2 }}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth>
                      <InputLabel>Video Source</InputLabel>
                      <Select
                        value={selectedSource}
                        label="Video Source"
                        onChange={(e) => setSelectedSource(e.target.value)}
                      >
                        {sources.map(source => (
                          <MenuItem key={source} value={source}>
                            {source}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth>
                      <InputLabel>Detection Type</InputLabel>
                      <Select
                        value={detectionType}
                        label="Detection Type"
                        onChange={(e) => setDetectionType(e.target.value as 'pose' | 'hand' | 'face')}
                      >
                        <MenuItem value="pose">Pose Detection</MenuItem>
                        <MenuItem value="hand">Hand Detection</MenuItem>
                        <MenuItem value="face">Face Detection</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Button
                      variant="contained"
                      fullWidth
                      onClick={() => {
                        // Trigger reload by changing key
                        setSelectedSource('');
                        setTimeout(() => setSelectedSource(sources[0] || ''), 100);
                      }}
                      disabled={!selectedSource}
                    >
                      Refresh Visualization
                    </Button>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>

            <Grid item xs={12}>
              <HeatmapVisualization
                source={selectedSource}
                detectionType={detectionType}
                width={1200}
                height={675}
              />
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Detection Statistics
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Statistical analysis of detection data will be displayed here.
            </Typography>
            {/* Add statistics visualization components here */}
          </Paper>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Timeline View
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Timeline visualization of detections and events will be displayed here.
            </Typography>
            {/* Add timeline visualization components here */}
          </Paper>
        </TabPanel>
      </Paper>
    </Container>
  );
};