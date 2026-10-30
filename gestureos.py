"""
GestureOS - control your computer with hand gestures.

Webcam -> MediaPipe hand landmarks -> gesture classifier -> OS actions (pyautogui).

Keys (in the preview window):  q = quit,  l = lock/unlock
"""

import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ----------------------------- Configuration -----------------------------
CAM_INDEX = 0            # change if you have several cameras
FRAME_W, FRAME_H = 640, 480
MARGIN = 100             # px border of the camera frame that is NOT used (active box)
SMOOTHING = 5.0          # higher = smoother but laggier cursor
PINCH_RATIO = 0.28       # thumb-finger distance / hand size below this = pinch
DRAG_HOLD_S = 0.35       # hold a pinch longer than this to start a drag
RIGHT_CLICK_COOLDOWN_S = 0.6
LOCK_HOLD_S = 1.0        # hold a fist this long to lock/unlock
STABLE_FRAMES = 3        # a gesture must be seen this many frames in a row
SCROLL_GAIN = 2.0        # scroll amount per pixel of hand movement
SCROLL_DEADZONE_PX = 4
VOLUME_STEP_PX = 25      # hand travel (px) per volume key press

# Air drawing
BRUSH_SIZE = 4           # line thickness in px (thinner = nicer handwriting)
ERASER_RADIUS = 40       # eraser size (open palm) in px
COLOR_CHANGE_COOLDOWN_S = 0.8
# Line smoothing (One Euro filter):
SMOOTH_MIN_CUTOFF = 1.5  # lower = smoother when moving slowly (less shake), more lag
SMOOTH_BETA = 0.03       # higher = less lag when moving fast
PEN_UP_GRACE_FRAMES = 4  # pen stays down this many frames if tracking flickers

pyautogui.FAILSAFE = False   # we map to the screen corners, so disable the corner failsafe
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ----------------------------- Gesture logic -----------------------------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(lm):
    """Return [thumb, index, middle, ring, pinky] booleans (rotation independent)."""
    thumb = dist(lm[4], lm[17]) > dist(lm[3], lm[17])
    others = [
        dist(lm[tip], lm[0]) > dist(lm[pip], lm[0])
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    ]
    return [thumb] + others


def classify(lm):
    """Map hand landmarks to a gesture name."""
    thumb, idx, mid, ring, pinky = fingers_up(lm)
    size = dist(lm[0], lm[9]) or 1e-6  # wrist -> middle-finger knuckle

    if dist(lm[4], lm[8]) / size < PINCH_RATIO:
        return "PINCH"          # left click / drag
    if dist(lm[4], lm[12]) / size < PINCH_RATIO:
        return "RIGHT_CLICK"
    if idx and not mid and not ring and not pinky:
        return "MOVE"
    if idx and mid and not ring and not pinky:
        return "SCROLL"
    if idx and mid and ring and not pinky:
        return "VOLUME"
    if idx and mid and ring and pinky:
        return "PALM"           # neutral / idle
    if not (idx or mid or ring or pinky):
        return "FIST"           # hold to lock/unlock
    if pinky and not (idx or mid or ring):
        return "PINKY"          # hold 1 s to switch mouse mode <-> draw mode
    return "IDLE"


class GestureOS:
    def __init__(self):
        self.locked = False
        self.prev_cursor = None
        self.history = deque(maxlen=STABLE_FRAMES)
        self.gesture = "IDLE"
        # pinch / drag state
        self.pinch_start = None
        self.dragging = False
        # other state
        self.last_right_click = 0.0
        self.fist_start = None
        self.fist_armed = True
        self.anchor_y = None
        self.vol_accum = 0.0

    # --- helpers ---
    def stable(self, raw):
        self.history.append(raw)
        if len(self.history) == STABLE_FRAMES and len(set(self.history)) == 1:
            self.gesture = raw
        return self.gesture

    def move_cursor(self, x_px, y_px):
        tx = np.interp(x_px, (MARGIN, FRAME_W - MARGIN), (0, SCREEN_W))
        ty = np.interp(y_px, (MARGIN, FRAME_H - MARGIN), (0, SCREEN_H))
        if self.prev_cursor is None:
            cx, cy = tx, ty
        else:
            px, py = self.prev_cursor
            cx = px + (tx - px) / SMOOTHING
            cy = py + (ty - py) / SMOOTHING
        cx = float(np.clip(cx, 0, SCREEN_W - 1))
        cy = float(np.clip(cy, 0, SCREEN_H - 1))
        pyautogui.moveTo(cx, cy)
        self.prev_cursor = (cx, cy)

    def release_all(self):
        if self.dragging:
            pyautogui.mouseUp()
            self.dragging = False
        self.pinch_start = None

    # --- per-frame update ---
    def update(self, lm):
        raw = classify(lm)
        g = self.stable(raw)
        tip_x, tip_y = lm[8].x * FRAME_W, lm[8].y * FRAME_H
        now = time.time()

        # Lock / unlock: hold a fist
        if g == "FIST":
            if self.fist_start is None:
                self.fist_start = now
            elif self.fist_armed and now - self.fist_start >= LOCK_HOLD_S:
                self.locked = not self.locked
                self.fist_armed = False
                self.release_all()
        else:
            self.fist_start = None
            self.fist_armed = True

        if self.locked:
            return g

        # Left click / drag
        if g == "PINCH":
            if self.pinch_start is None:
                self.pinch_start = now
            elif not self.dragging and now - self.pinch_start >= DRAG_HOLD_S:
                pyautogui.mouseDown()
                self.dragging = True
            if self.dragging:
                self.move_cursor(tip_x, tip_y)
        else:
            if self.pinch_start is not None:
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                else:
                    pyautogui.click()
                self.pinch_start = None

        # Right click
        if g == "RIGHT_CLICK" and now - self.last_right_click > RIGHT_CLICK_COOLDOWN_S:
            pyautogui.rightClick()
            self.last_right_click = now

        # Cursor
        if g == "MOVE":
            self.move_cursor(tip_x, tip_y)
        elif g != "PINCH":
            self.prev_cursor = None if g not in ("SCROLL", "VOLUME") else self.prev_cursor

        # Scroll
        if g == "SCROLL":
            if self.anchor_y is None:
                self.anchor_y = tip_y
            dy = self.anchor_y - tip_y
            if abs(dy) > SCROLL_DEADZONE_PX:
                pyautogui.scroll(int(dy * SCROLL_GAIN))
                self.anchor_y = tip_y
        # Volume
        elif g == "VOLUME":
            if self.anchor_y is None:
                self.anchor_y = tip_y
            self.vol_accum += self.anchor_y - tip_y
            self.anchor_y = tip_y
            while self.vol_accum >= VOLUME_STEP_PX:
                pyautogui.press("volumeup")
                self.vol_accum -= VOLUME_STEP_PX
            while self.vol_accum <= -VOLUME_STEP_PX:
                pyautogui.press("volumedown")
                self.vol_accum += VOLUME_STEP_PX
        else:
            self.anchor_y = None
            self.vol_accum = 0.0

        return g


# ------------------------------ Air drawing ------------------------------
class OneEuro:
    """Adaptive low-pass filter: steady when slow, responsive when fast."""

    def __init__(self, min_cutoff=SMOOTH_MIN_CUTOFF, beta=SMOOTH_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t_prev = None
        self.x_prev = 0.0
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, 0.0
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


class AirCanvas:
    """Draw in the air with your index finger.

    Index only      -> draw
    Index + middle  -> hover (pen up, no line)
    Open palm       -> erase around your palm
    3 fingers       -> change colour
    Keys: d = back to mouse mode, c = clear, s = save PNG
    """

    COLORS = [
        ((0, 0, 255), "RED"),
        ((0, 255, 0), "GREEN"),
        ((255, 0, 0), "BLUE"),
        ((0, 255, 255), "YELLOW"),
        ((255, 0, 255), "MAGENTA"),
        ((0, 165, 255), "ORANGE"),
    ]

    def __init__(self):
        self.canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.prev_pt = None
        self.tip = None
        self.palm = None
        self.color_i = 0
        self.last_color_change = 0.0
        self.history = deque(maxlen=STABLE_FRAMES)
        self.gesture = "IDLE"
        self.pen_down = False
        self.missed = 0
        self.fx = OneEuro()
        self.fy = OneEuro()

    def clear(self):
        self.canvas[:] = 0
        self.prev_pt = None

    def reset_pen(self):
        """Call when the hand disappears: lift the pen."""
        self.prev_pt = None
        self.tip = None
        self.palm = None
        self.history.clear()
        self.gesture = "NO HAND"
        self.pen_down = False
        self.missed = 0
        self.fx.reset()
        self.fy.reset()

    def update(self, lm):
        raw = classify(lm)
        self.history.append(raw)
        if len(self.history) == STABLE_FRAMES and len(set(self.history)) == 1:
            self.gesture = raw
        g = self.gesture
        now = time.time()

        x, y = int(lm[8].x * FRAME_W), int(lm[8].y * FRAME_H)
        self.tip = (x, y)
        self.palm = (int(lm[9].x * FRAME_W), int(lm[9].y * FRAME_H))
        color = self.COLORS[self.color_i][0]

        # Pen state: starts after a stable "index only" gesture, and stays down
        # through short tracking flickers so strokes are not broken.
        if self.pen_down:
            if raw == "MOVE":
                self.missed = 0
            else:
                self.missed += 1
                if self.missed > PEN_UP_GRACE_FRAMES:
                    self.pen_down = False
        elif g == "MOVE":
            self.pen_down = True
            self.missed = 0
            self.fx.reset()
            self.fy.reset()

        if self.pen_down:
            pt = (int(self.fx(x, now)), int(self.fy(y, now)))
            if self.prev_pt is not None and pt != self.prev_pt:
                cv2.line(self.canvas, self.prev_pt, pt, color, BRUSH_SIZE, cv2.LINE_AA)
            self.prev_pt = pt
        else:
            self.prev_pt = None
            if g == "PALM":  # eraser
                cv2.circle(self.canvas, self.palm, ERASER_RADIUS, (0, 0, 0), -1)
            elif g == "VOLUME" and now - self.last_color_change > COLOR_CHANGE_COOLDOWN_S:
                self.color_i = (self.color_i + 1) % len(self.COLORS)
                self.last_color_change = now
        return g

    def render(self, frame):
        """Overlay the drawing, pen cursor and colour swatch on the camera frame."""
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY) > 0
        frame[mask] = self.canvas[mask]
        color, name = self.COLORS[self.color_i]
        if self.tip is not None:
            if self.gesture == "PALM" and self.palm is not None:
                cv2.circle(frame, self.palm, ERASER_RADIUS, (255, 255, 255), 2)
            elif self.pen_down:
                cv2.circle(frame, self.prev_pt or self.tip, 8, color, -1)
            else:
                cv2.circle(frame, self.tip, 10, color, 2)
        cv2.rectangle(frame, (FRAME_W - 120, 10), (FRAME_W - 10, 40), color, -1)
        cv2.putText(frame, name, (FRAME_W - 115, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "c: clear   s: save", (10, FRAME_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

    def save(self):
        out = np.full_like(self.canvas, 255)  # white background
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY) > 0
        out[mask] = self.canvas[mask]
        name = f"drawing_{int(time.time())}.png"
        cv2.imwrite(name, out)
        return name


# --------------------------------- HUD ---------------------------------
def draw_hud(frame, gesture, locked, fps, draw_mode=False):
    cv2.rectangle(frame, (MARGIN, MARGIN), (FRAME_W - MARGIN, FRAME_H - MARGIN), (255, 0, 255), 2)
    status = "LOCKED" if locked else gesture
    color = (0, 0, 255) if locked else (0, 255, 0)
    cv2.putText(frame, f"Gesture: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Big mode banner so you always know which mode you are in
    banner = "MODE: DRAW" if draw_mode else "MODE: MOUSE"
    bg = (180, 60, 0) if draw_mode else (0, 120, 0)
    cv2.rectangle(frame, (FRAME_W - 200, 50), (FRAME_W - 10, 85), bg, -1)
    cv2.putText(frame, banner, (FRAME_W - 190, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if draw_mode:
        hint = "D key or pinky 1s: MOUSE mode   q: quit"
    else:
        hint = "D key or pinky 1s: DRAW mode   l: lock   q: quit"
    cv2.putText(frame, hint, (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Try changing CAM_INDEX.")

    os_ctl = GestureOS()
    air = AirCanvas()
    draw_mode = False
    prev_t = time.time()
    pinky_start = None
    pinky_armed = True

    def toggle_mode():
        nonlocal draw_mode
        draw_mode = not draw_mode
        os_ctl.release_all()
        os_ctl.history.clear()
        air.reset_pen()
        print("Switched to", "DRAW mode" if draw_mode else "MOUSE mode")

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))  # needed for phone cameras
                frame = cv2.flip(frame, 1)  # mirror so it feels natural
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    if draw_mode:
                        air.update(hand.landmark)
                    else:
                        os_ctl.update(hand.landmark)
                else:
                    os_ctl.release_all()
                    os_ctl.history.clear()
                    os_ctl.gesture = "NO HAND"
                    os_ctl.prev_cursor = None
                    air.reset_pen()

                if draw_mode:
                    frame = air.render(frame)

                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now

                # Hold only the pinky finger up for 1 second to switch modes
                current = air.gesture if draw_mode else os_ctl.gesture
                if current == "PINKY":
                    if pinky_start is None:
                        pinky_start = now
                    elif pinky_armed and now - pinky_start >= 1.0:
                        toggle_mode()
                        pinky_armed = False
                else:
                    pinky_start = None
                    pinky_armed = True

                draw_hud(frame, current, os_ctl.locked and not draw_mode, fps, draw_mode)
                cv2.imshow("GestureOS", frame)

                key = cv2.waitKey(1) & 0xFF
                k = chr(key).lower() if key != 255 else ""
                if k == "q":
                    break
                if k == "l":
                    os_ctl.locked = not os_ctl.locked
                    os_ctl.release_all()
                if k == "d":  # toggle air-drawing mode
                    toggle_mode()
                if k == "c":
                    air.clear()
                if k == "s":
                    print("Saved", air.save())
        finally:
            os_ctl.release_all()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()