import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Pause, RotateCcw } from 'lucide-react';

const MulticamChangeDetector = () => {
  const [video, setVideo] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [changes, setChanges] = useState([]);
  const [currentState, setCurrentState] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const previousStateRef = useRef(null);
  const frameCountRef = useRef(0);
  const stableStateRef = useRef(null);
  const stableFramesRef = useRef(0);

  const STABILITY_THRESHOLD = 5; // Frames needed to confirm a change
  const BRIGHTNESS_THRESHOLD = 30; // Threshold for black screen detection
  const EDGE_THRESHOLD = 0.1; // Threshold for edge density

  // Detect if screen is in grid or fullscreen mode
  const detectLayout = (canvas, ctx, imageData) => {
    const width = canvas.width;
    const height = canvas.height;
    const data = imageData.data;
    
    // Check for vertical divider in the middle
    const centerX = Math.floor(width / 2);
    const centerY = Math.floor(height / 2);
    
    let verticalEdges = 0;
    let horizontalEdges = 0;
    
    // Sample vertical center line
    for (let y = Math.floor(height * 0.3); y < Math.floor(height * 0.7); y++) {
      const idx = (y * width + centerX) * 4;
      const idx1 = (y * width + (centerX - 1)) * 4;
      const idx2 = (y * width + (centerX + 1)) * 4;
      
      const diff = Math.abs(
        (data[idx] + data[idx + 1] + data[idx + 2]) / 3 -
        (data[idx1] + data[idx1 + 1] + data[idx1 + 2]) / 3
      );
      
      if (diff > 20) verticalEdges++;
    }
    
    // Sample horizontal center line
    for (let x = Math.floor(width * 0.3); x < Math.floor(width * 0.7); x++) {
      const idx = (centerY * width + x) * 4;
      const idx1 = ((centerY - 1) * width + x) * 4;
      const idx2 = ((centerY + 1) * width + x) * 4;
      
      const diff = Math.abs(
        (data[idx] + data[idx + 1] + data[idx + 2]) / 3 -
        (data[idx1] + data[idx1 + 1] + data[idx1 + 2]) / 3
      );
      
      if (diff > 20) horizontalEdges++;
    }
    
    const verticalEdgeRatio = verticalEdges / (height * 0.4);
    const horizontalEdgeRatio = horizontalEdges / (width * 0.4);
    
    // If both dividers are strong, it's grid mode
    return (verticalEdgeRatio > 0.3 && horizontalEdgeRatio > 0.3) ? 'grid' : 'fullscreen';
  };

  // Check if a quadrant is active (has video content) or inactive (black/disconnected)
  const isQuadrantActive = (imageData, x, y, w, h) => {
    const data = imageData.data;
    const width = imageData.width;
    
    let totalBrightness = 0;
    let edgeCount = 0;
    let sampleCount = 0;
    
    // Sample points in the quadrant (every 10 pixels to speed up)
    for (let py = y; py < y + h; py += 10) {
      for (let px = x; px < x + w; px += 10) {
        if (py >= imageData.height || px >= imageData.width) continue;
        
        const idx = (py * width + px) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        totalBrightness += brightness;
        
        // Check for edges (content variation)
        if (px < x + w - 1 && py < y + h - 1) {
          const idx2 = (py * width + (px + 1)) * 4;
          const brightness2 = (data[idx2] + data[idx2 + 1] + data[idx2 + 2]) / 3;
          if (Math.abs(brightness - brightness2) > 30) edgeCount++;
        }
        
        sampleCount++;
      }
    }
    
    const avgBrightness = totalBrightness / sampleCount;
    const edgeDensity = edgeCount / sampleCount;
    
    // Active if: reasonable brightness AND has content variation (edges)
    return avgBrightness > BRIGHTNESS_THRESHOLD || edgeDensity > EDGE_THRESHOLD;
  };

  // Get current state fingerprint
  const getStateFingerprint = (canvas, ctx) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const layout = detectLayout(canvas, ctx, imageData);
    
    const width = canvas.width;
    const height = canvas.height;
    
    let quadrants = [0, 0, 0, 0];
    
    if (layout === 'grid') {
      // Check all 4 quadrants
      const halfW = Math.floor(width / 2);
      const halfH = Math.floor(height / 2);
      
      // Top-left, Top-right, Bottom-left, Bottom-right
      quadrants[0] = isQuadrantActive(imageData, 0, 0, halfW, halfH) ? 1 : 0;
      quadrants[1] = isQuadrantActive(imageData, halfW, 0, halfW, halfH) ? 1 : 0;
      quadrants[2] = isQuadrantActive(imageData, 0, halfH, halfW, halfH) ? 1 : 0;
      quadrants[3] = isQuadrantActive(imageData, halfW, halfH, halfW, halfH) ? 1 : 0;
    } else {
      // Fullscreen - only check if there's content
      quadrants[0] = isQuadrantActive(imageData, 0, 0, width, height) ? 1 : 0;
    }
    
    return {
      layout,
      quadrants,
      fingerprint: `${layout}-${quadrants.join('')}`
    };
  };

  // Process video frame by frame
  const processFrame = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (video.paused || video.ended) {
      setIsPlaying(false);
      return;
    }
    
    // Draw current frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get state fingerprint
    const state = getStateFingerprint(canvas, ctx);
    setCurrentState(state);
    
    // Check for changes
    if (previousStateRef.current && state.fingerprint !== previousStateRef.current.fingerprint) {
      // State has changed - but wait for stability
      if (stableStateRef.current === state.fingerprint) {
        stableFramesRef.current++;
        
        // If stable for enough frames, register the change
        if (stableFramesRef.current >= STABILITY_THRESHOLD) {
          const changeTime = video.currentTime;
          setChanges(prev => [...prev, {
            time: changeTime,
            from: previousStateRef.current,
            to: state,
            frameNumber: frameCountRef.current
          }]);
          previousStateRef.current = state;
          stableFramesRef.current = 0;
        }
      } else {
        // New potential state
        stableStateRef.current = state.fingerprint;
        stableFramesRef.current = 1;
      }
    } else {
      // State unchanged
      stableFramesRef.current = 0;
      stableStateRef.current = null;
    }
    
    if (!previousStateRef.current) {
      previousStateRef.current = state;
    }
    
    frameCountRef.current++;
    setProgress((video.currentTime / video.duration) * 100);
    
    requestAnimationFrame(processFrame);
  };

  const handleVideoUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideo(url);
      setChanges([]);
      setCurrentState(null);
      previousStateRef.current = null;
      frameCountRef.current = 0;
      setProgress(0);
    }
  };

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
        processFrame();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleReset = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.pause();
      setIsPlaying(false);
      setChanges([]);
      setCurrentState(null);
      previousStateRef.current = null;
      frameCountRef.current = 0;
      setProgress(0);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      video.addEventListener('loadedmetadata', () => {
        canvasRef.current.width = video.videoWidth;
        canvasRef.current.height = video.videoHeight;
      });
    }
  }, [video]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Multicam Screen Change Detector</h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Video Display */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Video Preview</h2>
            
            {!video ? (
              <label className="flex flex-col items-center justify-center h-96 border-2 border-dashed border-gray-600 rounded-lg cursor-pointer hover:border-gray-500">
                <Upload className="w-12 h-12 mb-4 text-gray-400" />
                <span className="text-gray-400">Click to upload video</span>
                <input type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
              </label>
            ) : (
              <div>
                <video
                  ref={videoRef}
                  src={video}
                  className="w-full rounded-lg mb-4"
                  onEnded={() => setIsPlaying(false)}
                />
                <canvas ref={canvasRef} className="hidden" />
                
                <div className="flex gap-2 mb-4">
                  <button
                    onClick={handlePlayPause}
                    className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    {isPlaying ? 'Pause' : 'Play'}
                  </button>
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded-lg"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Reset
                  </button>
                </div>
                
                <div className="mb-2">
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
                
                {currentState && (
                  <div className="bg-gray-700 p-3 rounded-lg">
                    <div className="text-sm">
                      <span className="font-semibold">Current State:</span> {currentState.layout}
                    </div>
                    <div className="text-sm">
                      <span className="font-semibold">Active Cameras:</span>{' '}
                      {currentState.layout === 'grid' 
                        ? currentState.quadrants.filter(q => q === 1).length + '/4'
                        : currentState.quadrants[0] ? 'Active' : 'Inactive'}
                    </div>
                    {currentState.layout === 'grid' && (
                      <div className="grid grid-cols-2 gap-1 mt-2">
                        {currentState.quadrants.map((active, i) => (
                          <div
                            key={i}
                            className={`text-xs p-2 rounded text-center ${
                              active ? 'bg-green-600' : 'bg-red-600'
                            }`}
                          >
                            Q{i + 1}: {active ? 'ON' : 'OFF'}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
          
          {/* Change Log */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">
              Detected Changes ({changes.length})
            </h2>
            
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {changes.length === 0 ? (
                <p className="text-gray-400">No changes detected yet. Upload and play a video.</p>
              ) : (
                changes.map((change, index) => (
                  <div key={index} className="bg-gray-700 p-3 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-semibold text-blue-400">Change #{index + 1}</span>
                      <span className="text-sm text-gray-400">{formatTime(change.time)}</span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <div className="text-gray-400 mb-1">From:</div>
                        <div className="bg-gray-600 p-2 rounded">
                          <div>{change.from.layout}</div>
                          {change.from.layout === 'grid' && (
                            <div className="text-xs mt-1">
                              Active: {change.from.quadrants.filter(q => q === 1).length}/4
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div>
                        <div className="text-gray-400 mb-1">To:</div>
                        <div className="bg-gray-600 p-2 rounded">
                          <div>{change.to.layout}</div>
                          {change.to.layout === 'grid' && (
                            <div className="text-xs mt-1">
                              Active: {change.to.quadrants.filter(q => q === 1).length}/4
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
        
        <div className="mt-6 bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-2">Algorithm Summary</h3>
          <div className="text-sm text-gray-300 space-y-1">
            <p><strong>Layout Detection:</strong> Analyzes center dividers to distinguish grid (2x2) vs fullscreen mode</p>
            <p><strong>Activity Detection:</strong> Checks each quadrant's brightness and edge density to determine if camera is active</p>
            <p><strong>Change Detection:</strong> Compares state fingerprints between frames, requires {STABILITY_THRESHOLD} stable frames to confirm change</p>
            <p><strong>Motion Filtering:</strong> Ignores content movement within frames, only tracks structural state changes</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MulticamChangeDetector;