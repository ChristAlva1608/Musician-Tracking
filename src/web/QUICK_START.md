# Quick Start - Musician Tracking Web Interface

## 🚀 The Easiest Way to Run

### Option 1: Using Startup Scripts (Recommended)

#### Start Backend (Terminal 1):
```bash
cd "/Volumes/Extreme_Pro/Mitou Project/Musician-Tracking/src/web"
./start-backend.sh
```

#### Start Frontend (Terminal 2):
```bash
cd "/Volumes/Extreme_Pro/Mitou Project/Musician-Tracking/src/web"
./start-frontend.sh
```

---

### Option 2: Manual Start

#### Backend (Terminal 1):
```bash
cd "/Volumes/Extreme_Pro/Mitou Project/Musician-Tracking/src/web/backend"
pip install -r requirements.txt  # First time only
python main.py
```

#### Frontend (Terminal 2):
```bash
cd "/Volumes/Extreme_Pro/Mitou Project/Musician-Tracking/src/web/frontend"
npm install  # First time only
npm start
```

---

## 📍 Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🎯 Using the New Video Upload Feature

1. Open http://localhost:3000
2. Click **"Upload Video"** in the navigation menu
3. Upload a video file
4. Choose detection mode:
   - **Single Person**: Faster, focuses on one musician (select Left or Right)
   - **Multiple People**: Tracks up to 2 people
5. Configure models (optional)
6. Click **"Upload & Process"**
7. Monitor progress in the **"Processing"** page

---

## ⚠️ Common Issues

### Backend Error: "ModuleNotFoundError"
```bash
cd "/Volumes/Extreme_Pro/Mitou Project/Musician-Tracking/src/web/backend"
pip install -r requirements.txt
```

### Frontend Error: "react-scripts not found"
```bash
cd "/Volumes/Extreme_Pro/Mitou Project/Musician-Tracking/src/web/frontend"
npm install
```

### Port Already in Use
- Backend: Edit `backend/main.py` and change port to 8001
- Frontend: Run `PORT=3001 npm start`

---

## 📝 Notes

- **Backend** must be running before **Frontend** can make API calls
- Keep both terminals open while using the application
- Press **Ctrl+C** in each terminal to stop the servers
- The frontend automatically proxies API requests to the backend

---

## 🆘 Need Help?

Check the full README.md for detailed documentation and troubleshooting.
