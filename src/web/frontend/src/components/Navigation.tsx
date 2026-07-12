import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Box,
  Divider
} from '@mui/material';
import {
  Home,
  Settings,
  PlayArrow,
  Assessment,
  Storage,
  MusicNote,
  Timeline,
  CloudUpload
} from '@mui/icons-material';

const drawerWidth = 240;

const menuItems = [
  { path: '/', label: 'Home', icon: <Home /> },
  { path: '/upload', label: 'Upload Video', icon: <CloudUpload /> },
  { path: '/processing', label: 'Processing', icon: <PlayArrow /> },
  { path: '/results', label: 'Results', icon: <Assessment /> },
  { path: '/visualization', label: 'Visualization', icon: <Timeline /> },
  { path: '/database', label: 'Database', icon: <Storage /> },
  { path: '/settings', label: 'Settings', icon: <Settings /> },
];

export const Navigation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
    >
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <MusicNote color="primary" />
          <Typography variant="h6" noWrap component="div">
            Musician Tracking
          </Typography>
        </Box>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};