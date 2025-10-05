import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Grid,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  CircularProgress
} from '@mui/material';
import { useTheme } from '@mui/material/styles';

interface HeatmapData {
  timestamps: number[];
  keypoints: {
    [key: string]: {
      x: number[];
      y: number[];
      confidence: number[];
    };
  };
  frameWidth: number;
  frameHeight: number;
}

interface HeatmapVisualizationProps {
  source?: string;
  detectionType?: 'pose' | 'hand' | 'face';
  width?: number;
  height?: number;
}

export const HeatmapVisualization: React.FC<HeatmapVisualizationProps> = ({
  source,
  detectionType = 'pose',
  width = 800,
  height = 600
}) => {
  const theme = useTheme();
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedKeypoint, setSelectedKeypoint] = useState<string>('all');
  const [intensity, setIntensity] = useState(50);
  const [viewMode, setViewMode] = useState<'heatmap' | 'trajectory'>('heatmap');
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (source) {
      fetchHeatmapData();
    }
  }, [source, detectionType]);

  useEffect(() => {
    if (heatmapData && canvasRef.current) {
      renderVisualization();
    }
  }, [heatmapData, selectedKeypoint, intensity, viewMode]);

  const fetchHeatmapData = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `/api/database/heatmap?source=${encodeURIComponent(source || '')}&type=${detectionType}`
      );
      const data = await response.json();
      setHeatmapData(data);
    } catch (error) {
      console.error('Failed to fetch heatmap data:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas || !heatmapData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set canvas size
    canvas.width = width;
    canvas.height = height;

    // Calculate scale factors
    const scaleX = width / heatmapData.frameWidth;
    const scaleY = height / heatmapData.frameHeight;

    if (viewMode === 'heatmap') {
      renderHeatmap(ctx, scaleX, scaleY);
    } else {
      renderTrajectory(ctx, scaleX, scaleY);
    }
  };

  const renderHeatmap = (ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) => {
    if (!heatmapData) return;

    // Create a 2D heatmap grid
    const gridSize = 20;
    const heatGrid: number[][] = Array(Math.ceil(height / gridSize))
      .fill(null)
      .map(() => Array(Math.ceil(width / gridSize)).fill(0));

    // Accumulate keypoint positions
    const keypointsToRender = selectedKeypoint === 'all' 
      ? Object.keys(heatmapData.keypoints)
      : [selectedKeypoint];

    keypointsToRender.forEach(kp => {
      const keypoint = heatmapData.keypoints[kp];
      if (!keypoint) return;

      keypoint.x.forEach((x, i) => {
        const y = keypoint.y[i];
        const conf = keypoint.confidence[i];
        
        if (conf > 0.3) {  // Only consider confident detections
          const gridX = Math.floor((x * scaleX) / gridSize);
          const gridY = Math.floor((y * scaleY) / gridSize);
          
          if (gridX >= 0 && gridX < heatGrid[0].length && 
              gridY >= 0 && gridY < heatGrid.length) {
            heatGrid[gridY][gridX] += conf * (intensity / 50);
          }
        }
      });
    });

    // Find max value for normalization
    const maxValue = Math.max(...heatGrid.flat());

    // Render heatmap
    heatGrid.forEach((row, y) => {
      row.forEach((value, x) => {
        if (value > 0) {
          const normalizedValue = value / maxValue;
          const alpha = Math.min(normalizedValue, 1);
          
          // Create gradient from blue to red
          const red = Math.floor(255 * normalizedValue);
          const blue = Math.floor(255 * (1 - normalizedValue));
          
          ctx.fillStyle = `rgba(${red}, 0, ${blue}, ${alpha * 0.7})`;
          ctx.fillRect(x * gridSize, y * gridSize, gridSize, gridSize);
        }
      });
    });
  };

  const renderTrajectory = (ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) => {
    if (!heatmapData) return;

    const keypointsToRender = selectedKeypoint === 'all' 
      ? Object.keys(heatmapData.keypoints)
      : [selectedKeypoint];

    // Set line style
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.7;

    keypointsToRender.forEach((kp, kpIndex) => {
      const keypoint = heatmapData.keypoints[kp];
      if (!keypoint) return;

      // Generate unique color for each keypoint
      const hue = (kpIndex * 360) / keypointsToRender.length;
      ctx.strokeStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;

      // Draw trajectory path
      ctx.beginPath();
      let prevX = -1;
      let prevY = -1;

      keypoint.x.forEach((x, i) => {
        const y = keypoint.y[i];
        const conf = keypoint.confidence[i];
        
        if (conf > 0.3) {
          const scaledX = x * scaleX;
          const scaledY = y * scaleY;
          
          if (prevX >= 0 && prevY >= 0) {
            ctx.lineTo(scaledX, scaledY);
          } else {
            ctx.moveTo(scaledX, scaledY);
          }
          
          prevX = scaledX;
          prevY = scaledY;
        }
      });
      
      ctx.stroke();

      // Draw points
      keypoint.x.forEach((x, i) => {
        const y = keypoint.y[i];
        const conf = keypoint.confidence[i];
        
        if (conf > 0.3) {
          const scaledX = x * scaleX;
          const scaledY = y * scaleY;
          
          ctx.beginPath();
          ctx.arc(scaledX, scaledY, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    });
  };

  const availableKeypoints = useMemo(() => {
    if (!heatmapData) return [];
    return Object.keys(heatmapData.keypoints);
  }, [heatmapData]);

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Detection Heatmap Visualization
      </Typography>

      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={3}>
          <FormControl fullWidth size="small">
            <InputLabel>Keypoint</InputLabel>
            <Select
              value={selectedKeypoint}
              label="Keypoint"
              onChange={(e) => setSelectedKeypoint(e.target.value)}
            >
              <MenuItem value="all">All Keypoints</MenuItem>
              {availableKeypoints.map(kp => (
                <MenuItem key={kp} value={kp}>{kp}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={3}>
          <Typography gutterBottom>Intensity</Typography>
          <Slider
            value={intensity}
            onChange={(_, value) => setIntensity(value as number)}
            min={10}
            max={100}
            marks
            valueLabelDisplay="auto"
          />
        </Grid>

        <Grid item xs={12} md={3}>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, value) => value && setViewMode(value)}
            size="small"
          >
            <ToggleButton value="heatmap">Heatmap</ToggleButton>
            <ToggleButton value="trajectory">Trajectory</ToggleButton>
          </ToggleButtonGroup>
        </Grid>

        <Grid item xs={12} md={3}>
          {source && (
            <Chip label={`Source: ${source}`} color="primary" />
          )}
        </Grid>
      </Grid>

      <Box sx={{ position: 'relative', width, height, border: '1px solid', borderColor: 'divider' }}>
        {loading ? (
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <CircularProgress />
          </Box>
        ) : (
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{ width: '100%', height: '100%' }}
          />
        )}
      </Box>

      {!heatmapData && !loading && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
          No data available. Select a source video to visualize.
        </Typography>
      )}
    </Paper>
  );
};