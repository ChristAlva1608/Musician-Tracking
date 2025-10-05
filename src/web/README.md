# Musician Tracking Web Interface

A modern web interface for the musician tracking system built with FastAPI and React + TypeScript.

## Features

### 🏠 **Homepage**
- **Single Video Processing**: Select individual video files for analysis
- **Folder Processing**: Process multiple videos with alignment and unified output
- **Model Configuration**: Configure detection models (hand, pose, face, emotion, transcript)
- **Performance Settings**: Adjust skip frames, processing duration limits
- **File Browser**: Easy navigation and selection of video files/folders

### ⚙️ **Settings Page**
- **Model Selection**: Configure all detection models
- **Processing Parameters**: Video settings, performance tuning
- **Database Configuration**: Enable/disable database storage
- **Configuration Presets**: Quick preset applications
- **Backup/Restore**: Configuration management

### 📊 **Processing Status**
- **Real-time Updates**: WebSocket-based live progress tracking
- **Job Management**: View, cancel, and monitor processing jobs
- **Output Files**: Download and manage generated videos
- **Processing Logs**: Detailed job execution logs

### 📈 **Results Dashboard** (Coming Soon)
- Video analysis visualization
- Detection statistics and charts
- Processing history and trends

### 🗃️ **Database Viewer** (Coming Soon)
- Browse alignment and detection data
- Query and filter database records
- Export data for analysis

## Architecture

```
web/
├── backend/                # FastAPI Backend
│   ├── main.py            # FastAPI application
│   ├── routers/           # API route modules
│   │   ├── processing.py  # Video processing endpoints
│   │   ├── config.py      # Configuration management
│   │   ├── files.py       # File system navigation
│   │   └── database.py    # Database queries
│   └── requirements.txt   # Python dependencies
├── frontend/              # React + TypeScript Frontend
│   ├── src/
│   │   ├── components/    # Reusable React components
│   │   ├── pages/         # Page components
│   │   ├── hooks/         # Custom React hooks
│   │   ├── types/         # TypeScript type definitions
│   │   └── services/      # API client services
│   ├── package.json       # Node.js dependencies
│   └── tsconfig.json      # TypeScript configuration
└── README.md              # This file
```

## Setup Instructions

### Backend Setup

1. **Install Python Dependencies**:
   ```bash
   cd src/web/backend
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Server**:
   ```bash
   python main.py
   ```

   The API will be available at: `http://localhost:8000`
   API documentation: `http://localhost:8000/docs`

### Frontend Setup

1. **Install Node.js Dependencies**:
   ```bash
   cd src/web/frontend
   npm install
   ```

2. **Start the React Development Server**:
   ```bash
   npm start
   ```

   The web interface will be available at: `http://localhost:3000`

### Production Deployment

1. **Build the Frontend**:
   ```bash
   cd src/web/frontend
   npm run build
   ```

2. **Serve with FastAPI**:
   The FastAPI backend will automatically serve the built React app from the `/` route.

## API Endpoints

### Processing Endpoints
- `POST /api/processing/single-video` - Process single video
- `POST /api/processing/folder` - Process folder of videos
- `GET /api/processing/jobs` - Get all processing jobs
- `GET /api/processing/jobs/{job_id}` - Get job status
- `DELETE /api/processing/jobs/{job_id}` - Cancel job
- `GET /api/processing/models` - Get available models

### Configuration Endpoints
- `GET /api/config/` - Get full configuration
- `GET /api/config/{section}` - Get config section
- `PUT /api/config/{section}` - Update config section
- `GET /api/config/presets/list` - Get configuration presets
- `POST /api/config/presets/{name}/apply` - Apply preset

### File System Endpoints
- `GET /api/files/browse` - Browse directories
- `GET /api/files/video-folders` - Get common video folders
- `GET /api/files/search` - Search for files
- `GET /api/files/outputs` - Get output files

### Database Endpoints
- `GET /api/database/tables` - Get database tables
- `GET /api/database/alignment/sources` - Get alignment sources
- `GET /api/database/alignment/{source}` - Get alignment data

### WebSocket
- `ws://localhost:8000/ws` - Real-time updates for processing status

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **WebSockets**: Real-time communication
- **Pydantic**: Data validation and serialization
- **PyYAML**: Configuration file handling

### Frontend
- **React 18**: Modern React with hooks
- **TypeScript**: Type-safe JavaScript
- **Material-UI**: Professional UI component library
- **Axios**: HTTP client for API calls
- **Socket.IO**: WebSocket client

## Key Features

### 🔄 **Real-time Updates**
- WebSocket connection for live processing updates
- Automatic job status synchronization
- Progress tracking with detailed logs

### 📁 **File Management**
- Safe file system navigation
- Video file detection and validation
- Common folder shortcuts
- File search functionality

### ⚙️ **Configuration Management**
- Live configuration editing
- Preset system for common configurations
- Configuration backup and restore
- Schema validation

### 🎛️ **Processing Control**
- Background job management
- Configurable processing parameters
- Output file management
- Error handling and reporting

## Development

### Adding New Features

1. **Backend**: Add new endpoints in `routers/`
2. **Frontend**: Create components in `components/` or pages in `pages/`
3. **Types**: Update TypeScript types in `types/index.ts`
4. **API Client**: Add new API calls in `services/api.ts`

### WebSocket Messages

The system uses WebSocket for real-time updates:

```typescript
interface WebSocketMessage {
  type: 'job_update' | 'status_update' | 'error';
  data: any;
}
```

## Troubleshooting

### Common Issues

1. **Backend Import Errors**: Ensure the project root is in Python path
2. **Database Connection**: Check database credentials and connection
3. **File Access**: Verify file permissions and safe path restrictions
4. **WebSocket Connection**: Check firewall and proxy settings

### Configuration

The web interface uses the same configuration system as the CLI tools:
- Configuration file: `src/config/config_v1.yaml`
- Backup files: `src/config/config_v1_backup_*.yaml`

## Security Notes

- File browser restricts access to safe directories
- Configuration changes require proper validation
- Processing jobs run in isolated environments
- Database queries use parameterized statements