import React from 'react';
import { Container, Typography, Box } from '@mui/material';

export const ResultsPage: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Results Dashboard
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          View and analyze processing results
        </Typography>
      </Box>

      <Typography variant="body1" color="text.secondary">
        Results dashboard coming soon...
      </Typography>
    </Container>
  );
};