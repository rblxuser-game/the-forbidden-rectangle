
# The Rectangle — Live Visual Analyzer

A Python desktop application for screen capture, computer-vision analysis, and motion-region tracking

## Features

- Desktop capture using MSS
- OpenCV-based motion detection
- Background subtraction
- Centroid-based object tracking
- Temporary tracking IDs
- Configurable motion-region filtering
- big red rectangle for overlay and yes

## Requirements

- Windows 10/11, macOS, or Linux
- IDLE Python 3.10+ or my version IDLE Python 3.14
- Desktop environment with screen-capture support

## Setup (Windows PowerShell)

```powershell
cd TheRectangle
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Notes

- Initial frames are used to establish the background model
- Large screen changes and animations may trigger detections
- Increase the minimum motion area to reduce noisy detections
- Detection results are visual estimates, not guaranteed object classifications
  
## Known Issues
- False detection
- Slow

## Privacy & Security

- Screen capture may include sensitive information
- Review the source code before running
- Only use screen analysis where you have permission
- May/Will violate game rules
- AntiVirus may flag as malicous, it's not malicous its a python executeable
