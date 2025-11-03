import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { Navigation } from './components/Navigation';
import { HomePage } from './pages/HomePage';
import { VideoUploadPage } from './pages/VideoUploadPage';
import { SettingsPage } from './pages/SettingsPage';
import { ProcessingPage } from './pages/ProcessingPage';
import { ResultsPage } from './pages/ResultsPage';
import { DatabasePage } from './pages/DatabasePage';
import { VisualizationPage } from './pages/VisualizationPage';
import { WebSocketProvider } from './hooks/useWebSocket';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <WebSocketProvider>
        <Router>
          <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            <Navigation />
            <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/upload" element={<VideoUploadPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="/processing" element={<ProcessingPage />} />
                <Route path="/results" element={<ResultsPage />} />
                <Route path="/database" element={<DatabasePage />} />
                <Route path="/visualization" element={<VisualizationPage />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </WebSocketProvider>
    </ThemeProvider>
  );
}

export default App;