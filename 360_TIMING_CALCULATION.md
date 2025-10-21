# 360-Degree Video Timing Calculation Guide

**Last Updated:** 2025-10-21
**Purpose:** Reference for calculating actual recording duration from Insta360 file chunks for visualization

---

## Understanding Insta360 Filename Format

### Filename Structure:
```
VID_YYYYMMDD_HHMMSS_LL_CCC.insv
    │        │       │   │
    │        │       │   └── Chunk number (001, 002, 003...)
    │        │       └────── Lens ID (00=front/single, 10=back for X3)
    │        └────────────── Time: Hour Minute Second (SESSION START)
    └─────────────────────── Date: Year Month Day
```

### **CRITICAL INSIGHT:**

**The timestamp (HHMMSS) represents the SESSION START TIME, NOT the individual chunk start time!**

All chunks from a single recording session will have the **SAME timestamp**.

---

## Duration Calculation Formula

### Method 1: File Size-Based Estimation

**Rule of Thumb:**
- **~20 GB file** ≈ **~29-30 minutes** of recording
- **Pro-rated:** Smaller files = proportional duration

**Formula:**
```python
duration_minutes = (file_size_GB / 20) * 30
```

**Example:**
```
File: VID_20250709_132654_00_001.insv (20 GB)
Duration: (20 / 20) × 30 = 30 minutes

File: VID_20250709_132654_00_003.insv (9.7 GB)
Duration: (9.7 / 20) × 30 = 14.55 minutes
```

### Method 2: Total Session Duration

**For continuous recordings (same timestamp across chunks):**

1. Sum all chunk file sizes for ONE camera
2. Apply the formula:

```python
total_duration_minutes = (total_size_GB / 20) * 30
```

**Example: Wangling's Session**
```
Camera 2 - X5 1:
  Chunk 001: 20 GB    → ~30 min
  Chunk 002: 20 GB    → ~30 min
  Chunk 003: 9.7 GB   → ~15 min
  ────────────────────────────
  Total: 49.7 GB      → ~74.55 minutes (1 hr 15 min)
```

---

## Timeline Reconstruction

### Step-by-Step Process:

**Given:** Chunks with same timestamp `VID_20250709_132654_00_XXX.insv`

**Reconstruct:**
```
Session Start: 13:26:54 (from filename)

Chunk 001 (20 GB):
  Start: 13:26:54
  End:   13:56:54 (add ~30 min)

Chunk 002 (20 GB):
  Start: 13:56:54
  End:   14:26:54 (add ~30 min)

Chunk 003 (9.7 GB):
  Start: 14:26:54
  End:   14:41:54 (add ~15 min)

Session End: 14:41:54
Total Duration: ~75 minutes
```

---

## Real Examples from Dataset

### Example 1: Wangling (July 9, 2025)

**Files:**
- `VID_20250709_132654_00_001.insv` - 20 GB
- `VID_20250709_132654_00_002.insv` - 20 GB
- `VID_20250709_132654_00_003.insv` - 9.7 GB

**Calculation:**
```
Session started: 13:26:54
Total size: 49.7 GB
Duration: (49.7 / 20) × 30 = 74.55 minutes

Timeline:
  13:26:54 - 13:56:54  Chunk 001 (30 min)
  13:56:54 - 14:26:54  Chunk 002 (30 min)
  14:26:54 - 14:41:54  Chunk 003 (15 min)

Actual recording time: ~1 hour 15 minutes
```

---

### Example 2: Christina (September 19, 2025)

**Setup:** 3 × Insta360 X5 cameras

**Camera 1 (X5 1):**
- `VID_20250919_140543_00_006.insv` - 17.5 GB
- `VID_20250919_140543_00_007.insv` - 17.5 GB

**Camera 2 (X5 2):**
- `VID_20250919_140534_00_007.insv` - 17.5 GB
- `VID_20250919_140534_00_008.insv` - 17.5 GB

**Camera 3 (X5 3):**
- `VID_20250919_140532_00_005.insv` - 17 GB
- `VID_20250919_140532_00_006.insv` - 17 GB

**Calculation (per camera):**
```
Each camera: ~35 GB total
Duration per camera: (35 / 20) × 30 = 52.5 minutes

Session started: 14:05:32 (earliest timestamp)
Session ended: ~14:58:02 (add ~52.5 min)

Actual recording time: ~53 minutes
```

**Note:** Slight timestamp variations (140532, 140534, 140543) across cameras are due to manual start timing, but all are part of the same session.

---

### Example 3: Lina (September 19, 2025)

**Setup:** 3 × Insta360 X5 cameras

**Multiple chunks per camera (007-013)**

**Estimated per camera:** ~50-60 GB total
**Duration:** (55 / 20) × 30 ≈ 82.5 minutes (~1 hour 23 minutes)

```
Session started: 15:23:23
Session ended: ~16:45:53
```

---

## Calculation Reference Table

### File Size → Duration Conversion

| File Size (GB) | Duration (minutes) | Duration (h:mm) |
|----------------|-------------------|-----------------|
| 5 GB           | ~7.5 min          | 0:07            |
| 10 GB          | ~15 min           | 0:15            |
| 15 GB          | ~22.5 min         | 0:22            |
| **20 GB**      | **~30 min**       | **0:30**        |
| 25 GB          | ~37.5 min         | 0:37            |
| 30 GB          | ~45 min           | 0:45            |
| 40 GB          | ~60 min           | 1:00            |
| 50 GB          | ~75 min           | 1:15            |
| 60 GB          | ~90 min           | 1:30            |
| 100 GB         | ~150 min          | 2:30            |

---

## Python Code for Automation

### Calculate Single File Duration
```python
def calculate_duration_minutes(file_size_gb):
    """
    Calculate recording duration from Insta360 file size.

    Args:
        file_size_gb (float): File size in gigabytes

    Returns:
        float: Duration in minutes
    """
    return (file_size_gb / 20.0) * 30.0

# Example
file_size = 9.7  # GB
duration = calculate_duration_minutes(file_size)
print(f"Duration: {duration:.2f} minutes ({duration/60:.2f} hours)")
# Output: Duration: 14.55 minutes (0.24 hours)
```

### Calculate Session Duration
```python
import os
import re
from datetime import datetime, timedelta

def calculate_session_duration(chunk_files):
    """
    Calculate total session duration from chunk files.

    Args:
        chunk_files: List of (filename, size_in_gb) tuples

    Returns:
        dict: Session info including duration and timeline
    """
    # Extract session start time from first chunk
    first_file = chunk_files[0][0]
    match = re.search(r'VID_(\d{8})_(\d{6})', first_file)

    if match:
        date_str = match.group(1)
        time_str = match.group(2)

        # Parse session start
        session_start = datetime.strptime(
            f"{date_str} {time_str}",
            "%Y%m%d %H%M%S"
        )

        # Calculate total duration
        total_size = sum(size for _, size in chunk_files)
        total_minutes = (total_size / 20.0) * 30.0

        session_end = session_start + timedelta(minutes=total_minutes)

        return {
            'session_start': session_start,
            'session_end': session_end,
            'duration_minutes': total_minutes,
            'duration_formatted': f"{int(total_minutes//60)}h {int(total_minutes%60)}m",
            'total_size_gb': total_size,
            'chunk_count': len(chunk_files)
        }

    return None

# Example usage
chunks = [
    ('VID_20250709_132654_00_001.insv', 20.0),
    ('VID_20250709_132654_00_002.insv', 20.0),
    ('VID_20250709_132654_00_003.insv', 9.7)
]

session = calculate_session_duration(chunks)
print(f"Session Start: {session['session_start']}")
print(f"Session End: {session['session_end']}")
print(f"Duration: {session['duration_formatted']}")
# Output:
# Session Start: 2025-07-09 13:26:54
# Session End: 2025-07-09 14:41:28
# Duration: 1h 14m
```

---

## Important Notes for Visualization

### 1. **Multiple Cameras → Same Timeline**
When a participant uses multiple cameras:
- Each camera has its own chunk files
- All cameras record the SAME session
- Use ANY one camera to calculate timeline
- All cameras should have similar total size

### 2. **Chunk Number Continuity**
Chunk numbers may NOT be continuous if:
- Earlier chunks were deleted
- Recording was stopped and restarted
- Different cameras have different chunk sequences

**Example:** Christina has chunks 005, 006, 007, 008 (not starting from 001)

### 3. **Timestamp Precision**
- Filename timestamp: **SECONDS only**
- Actual duration: Calculated from file size
- For precise frame-level timing: Use video metadata (ffprobe)

### 4. **Back-to-Back Sessions**
When participants record back-to-back:
- Look at file timestamps AND file modification times
- Session gap = difference between timestamps
- Each participant gets their own timeline

**Example: Christina & Lina (Sept 19, 2025)**
```
Christina: 14:05:32 - 14:58:02 (~53 min)
Gap: ~25 minutes
Lina:      15:23:23 - 16:45:53 (~82 min)
```

---

## Validation Checklist

Before using calculated timings for visualization:

- [ ] Verify chunk files are from the SAME session (same timestamp)
- [ ] Check file sizes are reasonable (~17-20 GB for full chunks)
- [ ] Confirm last chunk is smaller (indicates end of session)
- [ ] Account for multiple cameras (use one camera's timeline)
- [ ] Validate against actual session notes if available

---

## Common Pitfalls

### ❌ **WRONG: Using timestamp as chunk start time**
```
Chunk 001: 13:26:54 - 13:26:54 (0 minutes) ← WRONG!
```

### ✅ **CORRECT: Using timestamp as session start, calculating from size**
```
Session start: 13:26:54
Chunk 001: 20 GB → 30 minutes
Chunk 001 timeline: 13:26:54 - 13:56:54 ✓
```

---

## References

- **Insta360 File Specs:** Auto-split at ~20 GB
- **Typical Bitrate:** ~120-150 Mbps for X4/X5
- **Recording Format:** Dual-fisheye MP4 container (.insv)
- **Resolution:** 5.7K-8K depending on model

---

**For Questions:** Refer to this document when calculating session durations for:
- Timeline visualization
- Multi-camera synchronization
- Session planning
- Storage estimation
