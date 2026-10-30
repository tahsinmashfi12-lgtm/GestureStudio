
Try a Demo video : 
https://github.com/user-attachments/assets/e817153d-f50a-4e17-816c-d8ee8361848e

GestureStudio
Control your computer and draw in the air using only your hand. A webcam feeds MediaPipe hand tracking, and gestures are turned into real-time mouse, scroll, volume and drawing actions.

Built with Python, OpenCV, MediaPipe and PyAutoGUI.

Features
Virtual mouse: move, click, drag, right-click
Scroll and volume control by gesture
Air drawing with eraser, color switching and PNG export
Smooth strokes using a One Euro filter and a pen-lift grace period
Switch between mouse and draw modes by gesture or key
Lock gesture to pause control
Requirements
Python 3.9 to 3.12 (MediaPipe's solutions API is not available on newer Python versions)
A webcam (or a phone camera via DroidCam)
Windows recommended (volume keys work best there)
Setup
bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python gestureos.py


Gestures
Gesture	Mouse mode	Draw mode
Index finger only	Move cursor	Draw
Index + middle	Scroll (move hand up/down)	Move without drawing
Three fingers	Volume (move hand up/down)	Change color
Open palm	Idle	Erase
Thumb + index pinch	Click (hold to drag)	n/a
Thumb + middle pinch	Right click	n/a
Fist, hold 1 s	Lock / unlock	n/a
Pinky only, hold 1 s	Switch mode	Switch mode
Keys (click the camera window first)
Key	Action
q	Quit
d	Switch mouse / draw mode
l	Lock / unlock
c	Clear drawing
s	Save drawing as PNG
Keep your hand inside the purple box in the preview; it maps to the whole screen.

Tuning
Settings are at the top of gestureos.py:

SMOOTHING: cursor smoothness (higher = smoother, laggier)
PINCH_RATIO: pinch sensitivity
STABLE_FRAMES: frames a gesture must hold before it counts
SMOOTH_MIN_CUTOFF, SMOOTH_BETA: drawing smoothness vs lag
PEN_UP_GRACE_FRAMES: how long the pen survives tracking flicker
BRUSH_SIZE, ERASER_RADIUS: drawing sizes
Challenges I ran into
MediaPipe version and Python compatibility (needs Python 3.12 and an older MediaPipe)
A fist being detected as a pinch, which needed threshold and check-order tuning
Shaky handwriting, fixed with adaptive smoothing and a pen grace period
Balancing responsiveness against cursor jitter

Safety
The mouse is controlled by your hand. Press q in the preview window, or hold a fist to lock, if the cursor misbehaves.


