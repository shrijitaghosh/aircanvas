import os
import sys
import time
import threading

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. MEDIAPIPE HAND LANDMARKER (VIDEO mode = tracking, not
#    a cold detection every single frame -> faster + steadier)
# =========================================================

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    sys.exit(
        f"Could not find '{MODEL_PATH}' in the current folder.\n"
        "Download it from: "
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

detector = vision.HandLandmarker.create_from_options(options)


# =========================================================
# 2. THREADED CAMERA
#    cap.read() blocks the main thread until a new frame arrives.
#    Grabbing frames on a background thread and always working
#    with the most recent one removes that wait from the hot loop,
#    which is the single biggest remaining source of pointer lag.
# =========================================================

class ThreadedCamera:
    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            sys.exit(
                f"Could not open camera index {index}. "
                "Check that it's connected and not in use by another app."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        # Prime the first frame synchronously so callers never see None.
        ok, frame = self.cap.read()
        if not ok:
            sys.exit("Camera opened but did not return a frame.")
        self.frame = frame

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


camera = ThreadedCamera(index=0, width=1280, height=720, fps=30)

canvas = None

# Width we actually run detection on. Landmarks are normalized
# (0-1), so detecting on a smaller frame and scaling the result
# back up to full resolution loses no accuracy but is much faster.
DETECT_WIDTH = 640


# =========================================================
# 3. DRAWING VARIABLES
# =========================================================

prev_x = None
prev_y = None

# Current drawing points
drawing_points = []

# Last recognized shape
recognized_shape = ""

# Frames after drawing stops
stop_counter = 0

# Pinch state with hysteresis (see PINCH_ENTER / PINCH_EXIT below)
is_pinching_state = False

# Completed strokes, kept separately so a single stroke can be
# undone without wiping the whole canvas. Each entry is
# (points, color, thickness).
strokes = []
current_stroke = []

# Selectable colors (BGR) and brush thickness
COLORS = {
    "1": (255, 0, 255),   # magenta
    "2": (0, 255, 255),   # yellow
    "3": (255, 255, 0),   # cyan
    "4": (255, 255, 255), # white
}
draw_color = COLORS["1"]
brush_thickness = 6

# FPS tracking
fps_display = 0.0
fps_last_time = time.time()
fps_frame_count = 0


# =========================================================
# 4. ONE EURO FILTER
#    This replaces the old fixed-alpha smoothing. A single fixed
#    smoothing constant forces a trade-off: high alpha = responsive
#    but jittery, low alpha = smooth but laggy. One Euro adapts its
#    own cutoff to how fast the signal is moving, so it stays calm
#    when your finger is nearly still and snaps to fast motion
#    instead of trailing behind it. This is the standard fix for
#    exactly the "laggy pointer / messy line" symptom you're seeing.
# =========================================================

class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.2, beta=0.02, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x

        dt = t - self.t_prev
        if dt > 0:
            self.freq = 1.0 / dt
        self.t_prev = t

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


filter_x = OneEuroFilter(mincutoff=1.2, beta=0.03)
filter_y = OneEuroFilter(mincutoff=1.2, beta=0.03)


# =========================================================
# 5. THRESHOLDS
# =========================================================

# Ignore impossible jumps when drawing lines (tracking glitches)
MAX_JUMP = 120

# Hysteresis on the pinch gesture: enter drawing mode at a tighter
# distance, exit at a looser one. A single threshold flickers
# true/false when your fingers hover right at the boundary, which
# is what was breaking your lines into messy fragments.
PINCH_ENTER = 40
PINCH_EXIT = 55


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def redraw_canvas(shape):
    """Rebuild the canvas from the strokes list. Only called on
    undo/clear, not every frame, so the extra cost is negligible."""
    new_canvas = np.zeros(shape, dtype=np.uint8)
    for points, color, thickness in strokes:
        for i in range(1, len(points)):
            cv2.line(new_canvas, points[i - 1], points[i], color, thickness, cv2.LINE_AA)
    return new_canvas


def recognize_shape(points):
    if len(points) < 15:
        return "Too small"

    pts = np.array(points, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(pts)
    if w < 30 or h < 30:
        return "Too small"

    contour = pts.reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Unknown"

    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    ratio = w / float(h)

    if vertices >= 7:
        return "Circle"
    if vertices == 3:
        return "Triangle"
    if vertices == 4:
        if 0.80 <= ratio <= 1.20:
            return "Square"
        return "Rectangle"
    if vertices >= 5:
        return "Circle"
    return "Unknown"


# =========================================================
# 7. MAIN LOOP
# =========================================================

start_time = time.time()

try:
  while True:

    frame = camera.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    height, width, _ = frame.shape

    # FPS bookkeeping
    fps_frame_count += 1
    if time.time() - fps_last_time >= 0.5:
        fps_display = fps_frame_count / (time.time() - fps_last_time)
        fps_frame_count = 0
        fps_last_time = time.time()

    # -----------------------------------------------------
    # DOWNSCALE ONLY FOR DETECTION (drawing stays full-res)
    # -----------------------------------------------------

    scale = DETECT_WIDTH / float(width)
    small_frame = cv2.resize(
        frame,
        (DETECT_WIDTH, int(height * scale)),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    now = time.time()

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # ---------------------------------------------
        # INDEX FINGER TIP (normalized -> full-res pixels)
        # ---------------------------------------------

        index_tip = hand[8]
        raw_x = index_tip.x * width
        raw_y = index_tip.y * height

        x = int(filter_x.filter(raw_x, now))
        y = int(filter_y.filter(raw_y, now))

        # ---------------------------------------------
        # THUMB TIP
        # ---------------------------------------------

        thumb_tip = hand[4]
        thumb_x = int(thumb_tip.x * width)
        thumb_y = int(thumb_tip.y * height)

        # ---------------------------------------------
        # PINCH DETECTION (with hysteresis)
        # ---------------------------------------------

        pinch_distance = distance((x, y), (thumb_x, thumb_y))

        if is_pinching_state:
            if pinch_distance > PINCH_EXIT:
                is_pinching_state = False
        else:
            if pinch_distance < PINCH_ENTER:
                is_pinching_state = True

        is_pinching = is_pinching_state

        # ---------------------------------------------
        # OPEN HAND
        # ---------------------------------------------

        middle_tip = hand[12]
        ring_tip = hand[16]
        pinky_tip = hand[20]

        open_hand = (
            index_tip.y < hand[5].y
            and middle_tip.y < hand[9].y
            and ring_tip.y < hand[13].y
            and pinky_tip.y < hand[17].y
        )

        # ===============================================
        # ERASE
        # ===============================================

        if open_hand and not is_pinching:

            cv2.putText(
                frame, "ERASER", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            )

            palm = hand[9]
            palm_x = int(palm.x * width)
            palm_y = int(palm.y * height)

            cv2.circle(canvas, (palm_x, palm_y), 65, (0, 0, 0), -1)

            if current_stroke:
                strokes.append((current_stroke, draw_color, brush_thickness))
                current_stroke = []

            prev_x = None
            prev_y = None
            drawing_points = []

        # ===============================================
        # DRAW
        # ===============================================

        elif is_pinching:

            cv2.putText(
                frame, "DRAWING", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

            cv2.circle(frame, (x, y), 8, draw_color, -1)

            if prev_x is not None:

                movement = distance((prev_x, prev_y), (x, y))

                if movement < MAX_JUMP:
                    cv2.line(
                        canvas, (prev_x, prev_y), (x, y),
                        draw_color, brush_thickness, cv2.LINE_AA,
                    )
                    drawing_points.append((x, y))

                    if not current_stroke:
                        current_stroke.append((prev_x, prev_y))
                    current_stroke.append((x, y))

            prev_x = x
            prev_y = y
            stop_counter = 0

        # ===============================================
        # POINTER MODE
        # ===============================================

        else:

            cv2.putText(
                frame, "PINCH TO DRAW", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            )

            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

            if current_stroke:
                strokes.append((current_stroke, draw_color, brush_thickness))
                current_stroke = []

            if len(drawing_points) > 15:
                stop_counter += 1
                if stop_counter == 8:
                    recognized_shape = recognize_shape(drawing_points)
                    drawing_points = []

            prev_x = None
            prev_y = None

    else:
        prev_x = None
        prev_y = None
        is_pinching_state = False
        if current_stroke:
            strokes.append((current_stroke, draw_color, brush_thickness))
            current_stroke = []
        # Don't reset the One Euro filters here — losing hand
        # tracking for a single flickery frame shouldn't force the
        # pointer to "forget" where it was and jump on reacquire.

    # =====================================================
    # RECOGNIZED SHAPE
    # =====================================================

    if recognized_shape:
        cv2.rectangle(frame, (20, 75), (380, 130), (0, 0, 0), -1)
        cv2.putText(
            frame, "Shape: " + recognized_shape, (35, 112),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )

    # =====================================================
    # CAMERA + CANVAS
    # =====================================================

    output = cv2.add(frame, canvas)

    # Current color swatch
    cv2.rectangle(output, (width - 50, 10), (width - 10, 50), draw_color, -1)
    cv2.rectangle(output, (width - 50, 10), (width - 10, 50), (0, 0, 0), 2)

    cv2.putText(
        output, f"FPS: {fps_display:.0f}", (20, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
    )
    cv2.putText(
        output,
        "Pinch=Draw  Palm=Erase  1-4=Color  +/-=Size  Z=Undo  S=Save  C=Clear  Q=Quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    cv2.imshow("AirSketch AI", output)

    key = cv2.waitKey(1) & 0xFF
    key_char = chr(key) if 0 <= key < 256 else ""

    if key_char == "q":
        break

    elif key_char == "c":
        canvas = np.zeros_like(frame)
        strokes = []
        current_stroke = []
        drawing_points = []
        recognized_shape = ""
        stop_counter = 0

    elif key_char == "z":
        # Undo the last completed stroke
        if strokes:
            strokes.pop()
            canvas = redraw_canvas(frame.shape)
        recognized_shape = ""

    elif key_char == "s":
        filename = f"airsketch_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved drawing to {filename}")

    elif key_char in COLORS:
        draw_color = COLORS[key_char]

    elif key_char in ("+", "="):
        brush_thickness = min(brush_thickness + 1, 30)

    elif key_char == "-":
        brush_thickness = max(brush_thickness - 1, 1)

finally:
    camera.release()
    cv2.destroyAllWindows()