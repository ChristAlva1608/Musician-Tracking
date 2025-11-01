# Musician Tracking Web Interface

Web-based interface for the Musician Tracking System with video upload and person selection capabilities.

## Quick Start Guide

### Prerequisites
- Python 3.8+ with pip
- Node.js 16+ with npm
- PostgreSQL (optional, for database features)

---

## Running the Application

### Step 1: Start the Backend (FastAPI)

```bash
# Navigate to the backend directory
cd "/Volumes/Extreme_Pro/Mitou Project/Musician-Tracking/src/web/backend"

# Install Python dependencies (first time only)
pip install -r requirements.txt

# Start the backend server
python main.py
```

✅ Backend will run on **http://localhost:8000**
📚 API Documentation available at **http://localhost:8000/docs**

---

### Step 2: Start the Frontend (React)

Open a **new terminal window** and run:

```bash
# Navigate to the frontend directory
cd "/Volumes/Extreme_Pro/Mitou Project/Musician-Tracking/src/web/frontend"

# Install Node.js dependencies (first time only)
npm install

# Start the development server
npm start
```

✅ Frontend will run on **http://localhost:3000**
🌐 Your browser will automatically open to the app

---

## Features

### 📹 Video Upload Page (NEW!)
- Upload videos directly from browser
- Choose between **single person** or **multiple people** detection
- Select specific person (left/right) for focused tracking
- Configure detection models
- Real-time upload progress

### Other Pages
- **Home**: Process videos from file system
- **Processing**: Monitor jobs in real-time
- **Results**: View processed videos
- **Database**: Query detection data
- **Visualization**: Heatmap analysis
- **Settings**: Configure models and thresholds

---

## Troubleshooting

### Backend won't start
```bash
# Make sure you're in the correct directory
cd src/web/backend

# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend won't start
```bash
# Make sure you're in the correct directory
cd src/web/frontend

# Check Node.js version
node --version  # Should be 16+

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Port already in use
If port 8000 or 3000 is already in use, you can change the port:

**Backend**: Edit `main.py` and change the port number
**Frontend**: Run `PORT=3001 npm start`

---

## API Endpoints

### Processing
- `POST /api/processing/upload-video` - Upload and process video
- `GET /api/processing/jobs` - Get all jobs
- `GET /api/processing/jobs/{job_id}` - Get job status

### WebSocket
- `ws://localhost:8000/ws` - Real-time job updates

For full API documentation, visit **http://localhost:8000/docs** when the backend is running.
