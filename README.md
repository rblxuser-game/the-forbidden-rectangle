# ESP — Live Visual Analyzer

A Python desktop application for live desktop capture and generic motion-region tracking.

## What it does

- Captures a selected desktop display using MSS
- Uses OpenCV background subtraction to detect moving image regions
- Assigns temporary IDs using centroid tracking
- Draws red bounding boxes and center-to-region lines
- Displays approximate processing FPS and current track count
- Provides Start and Stop controls and a minimum-area setting

**Important:** This is motion-region analysis, not person recognition. It does not identify players, read game memory, inject into applications, or connect to game servers. Static objects and camera movement can cause missed detections or false positives.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10+
- A desktop session that permits screen capture

## Setup (Windows PowerShell)

```powershell
cd ESP
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Notes

- The first seconds establish a background model; movement may initially be classified as foreground.
- Large desktop changes, scrolling, camera motion, and animations can produce large regions.
- Increase minimum motion area to suppress small noisy detections.
- The displayed FPS is the approximate processing rate, not a guaranteed capture refresh rate.
