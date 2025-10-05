import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Breadcrumbs,
  Link,
  Typography,
  Box,
  CircularProgress,
  Alert
} from '@mui/material';
import { Folder, VideoFile, ArrowBack, Home } from '@mui/icons-material';
import { filesApi } from '../services/api';
import { DirectoryListing, FileItem } from '../types';

interface FileBrowserProps {
  open: boolean;
  mode: 'file' | 'folder';
  onClose: () => void;
  onSelect: (path: string) => void;
}

export const FileBrowser: React.FC<FileBrowserProps> = ({
  open,
  mode,
  onClose,
  onSelect
}) => {
  const [currentPath, setCurrentPath] = useState<string>('');
  const [listing, setListing] = useState<DirectoryListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    if (open) {
      loadDirectory();
    }
  }, [open]);

  const loadDirectory = async (path?: string) => {
    setLoading(true);
    setError('');
    try {
      const result = await filesApi.browseDirectory(path);
      setListing(result);
      setCurrentPath(result.current_path);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load directory');
    } finally {
      setLoading(false);
    }
  };

  const handleItemClick = (item: FileItem) => {
    if (item.type === 'directory') {
      loadDirectory(item.path);
    } else if (mode === 'file' && item.is_video) {
      onSelect(item.path);
    }
  };

  const handleFolderSelect = (path: string) => {
    if (mode === 'folder') {
      onSelect(path);
    }
  };

  const navigateUp = () => {
    if (listing?.parent_path) {
      loadDirectory(listing.parent_path);
    }
  };

  const pathParts = currentPath.split('/').filter(Boolean);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {mode === 'file' ? 'Select Video File' : 'Select Folder'}
      </DialogTitle>

      <DialogContent>
        {/* Breadcrumbs */}
        <Box sx={{ mb: 2 }}>
          <Breadcrumbs>
            <Link
              component="button"
              variant="body2"
              onClick={() => loadDirectory()}
              sx={{ display: 'flex', alignItems: 'center' }}
            >
              <Home sx={{ mr: 0.5 }} fontSize="inherit" />
              Root
            </Link>
            {pathParts.map((part, index) => {
              const path = '/' + pathParts.slice(0, index + 1).join('/');
              const isLast = index === pathParts.length - 1;

              return isLast ? (
                <Typography key={path} color="text.primary">
                  {part}
                </Typography>
              ) : (
                <Link
                  key={path}
                  component="button"
                  variant="body2"
                  onClick={() => loadDirectory(path)}
                >
                  {part}
                </Link>
              );
            })}
          </Breadcrumbs>
        </Box>

        {/* Navigation */}
        <Box sx={{ mb: 2, display: 'flex', gap: 1 }}>
          {listing?.parent_path && (
            <Button
              variant="outlined"
              size="small"
              startIcon={<ArrowBack />}
              onClick={navigateUp}
            >
              Back
            </Button>
          )}

          {mode === 'folder' && currentPath && (
            <Button
              variant="contained"
              size="small"
              onClick={() => handleFolderSelect(currentPath)}
            >
              Select This Folder
            </Button>
          )}
        </Box>

        {/* Content */}
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error">{error}</Alert>
        ) : (
          <List sx={{ maxHeight: 400, overflow: 'auto' }}>
            {listing?.items.map((item) => (
              <ListItem key={item.path} disablePadding>
                <ListItemButton
                  onClick={() => handleItemClick(item)}
                  disabled={mode === 'file' && item.type === 'file' && !item.is_video}
                >
                  <ListItemIcon>
                    {item.type === 'directory' ? (
                      <Folder color="primary" />
                    ) : (
                      <VideoFile
                        color={item.is_video ? "primary" : "disabled"}
                      />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.name}
                    secondary={
                      item.type === 'file' && item.size
                        ? `${(item.size / 1024 / 1024).toFixed(1)} MB`
                        : undefined
                    }
                  />
                </ListItemButton>
              </ListItem>
            ))}

            {listing?.items.length === 0 && (
              <ListItem>
                <ListItemText primary="No items found" />
              </ListItem>
            )}
          </List>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
      </DialogActions>
    </Dialog>
  );
};