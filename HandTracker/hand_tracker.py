"""
Hand tracker (MediaPipe + OpenCV) with hand-made filter frames.

Every visible fingertip (up to 5 per hand, 10 total) becomes a corner of a
shape. Whatever is inside the shape gets a filter applied. Works with one
hand or two. Spread your fingers to stretch the shape, curl them to shrink it.

Setup:
    pip install opencv-python mediapipe
    The hand-landmarker model is downloaded automatically on first run.

Run:
    python hand_tracker_filter.py

Keys:
    q = quit
    f = next filter
    x = filter on/off
    m = shape mode: "all" (every fingertip is a corner) / "hull" (tight outline)
    t = toggle fingertip trails
    g = toggle glow
    b = toggle per-hand bounding box
    s = save screenshot
"""

import time
from collections import deque
from pathlib import Path
import os
import shutil
import tempfile
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

mp_vision = mp.tasks.vision
HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

GREEN = (0, 255, 0)      # BGR
RED = (0, 0, 255)
WHITE = (255, 255, 255)

TIP_IDS = [4, 8, 12, 16, 20]
THUMB_TIP, INDEX_TIP = 4, 8

# one trail per fingertip per hand (max 2 hands)
trails = [[deque(maxlen=25) for _ in TIP_IDS] for _ in range(2)]


# ---------------------------------------------------------------- filters
def f_pixelate(roi):
    h, w = roi.shape[:2]
    block = 16
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def f_blur(roi):
    return cv2.GaussianBlur(roi, (0, 0), 12)


def f_invert(roi):
    return cv2.bitwise_not(roi)


def f_edges(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def f_thermal(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def f_gray(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


FILTERS = [
    ("pixelate", f_pixelate),
    ("blur", f_blur),
    ("invert", f_invert),
    ("edges", f_edges),
    ("thermal", f_thermal),
    ("grayscale", f_gray),
]


def apply_filter_in_quad(frame, quad, filter_fn):
    """Apply filter_fn only inside the 4-point polygon `quad`."""
    h, w = frame.shape[:2]
    quad = quad.copy()
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)

    x, y, bw, bh = cv2.boundingRect(quad)
    if bw < 10 or bh < 10:
        return

    roi = frame[y:y + bh, x:x + bw]
    filtered = filter_fn(roi.copy())

    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [quad - np.array([x, y])], 255)
    roi[mask > 0] = filtered[mask > 0]


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def extended_tips(pts):
    """
    Fingertips of fingers that are actually sticking out (not curled).
    Works at any hand rotation by comparing distances, not up/down.
    """
    wrist = pts[0]
    tips = []

    # thumb: extended if its tip is farther from the pinky base than its joint
    if _dist(pts[4], pts[17]) > _dist(pts[3], pts[17]) * 1.05:
        tips.append(pts[4])

    # other fingers: extended if the tip is farther from the wrist than the
    # middle joint (a curled finger folds back toward the palm)
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if _dist(pts[tip], wrist) > _dist(pts[pip], wrist) * 1.05:
            tips.append(pts[tip])

    return tips


def get_frame_quad(all_pts, mode="all"):
    """
    Build the filter shape from EVERY fingertip (5 per hand, up to 10 points).

    mode "all"  : connect every fingertip, ordered by angle around the center,
                  so each finger is a corner of the shape.
    mode "hull" : wrap a tight outline around the fingertips (curled fingers
                  that sit inside the outline get ignored).

    Returns an (N,2) int array, or None if fewer than 3 fingertips are visible.
    """
    tips = [t for p in all_pts for t in extended_tips(p)]
    if len(tips) < 3:
        return None

    pts = np.array(tips, dtype=np.int32)
    if mode == "hull":
        return cv2.convexHull(pts).reshape(-1, 2)

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


# ---------------------------------------------------------------- model
def get_model_path():
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    model_dir = cache_root / "HandTracker"
    model_path = model_dir / "hand_landmarker.task"
    if model_path.is_file():
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=model_dir, suffix=".task", delete=False) as f:
        temp_path = Path(f.name)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with temp_path.open("wb") as model_file:
                shutil.copyfileobj(response, model_file)
        os.replace(temp_path, model_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return model_path


# ---------------------------------------------------------------- drawing
def count_fingers(lm, handedness):
    """Rough finger count assuming the hand is held upright."""
    up = 0
    if handedness == "Right":
        up += lm[4].x > lm[3].x
    else:
        up += lm[4].x < lm[3].x
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        up += lm[tip].y < lm[pip].y
    return int(up)


def draw_hand(frame, lm, pts, hand_idx, handedness, opts):
    h, w = frame.shape[:2]

    layer = frame.copy() if opts["glow"] else None
    target = layer if layer is not None else frame

    for connection in HAND_CONNECTIONS:
        a, b = connection.start, connection.end
        cv2.line(target, pts[a], pts[b], GREEN, 3, cv2.LINE_AA)

    for i, p in enumerate(pts):
        r = 7 if i in TIP_IDS else 5
        cv2.circle(target, p, r, RED, -1, cv2.LINE_AA)

    if layer is not None:
        blur = cv2.GaussianBlur(layer, (0, 0), 9)
        frame[:] = cv2.addWeighted(frame, 1.0, blur, 0.6, 0)
        for connection in HAND_CONNECTIONS:
            a, b = connection.start, connection.end
            cv2.line(frame, pts[a], pts[b], GREEN, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 7 if i in TIP_IDS else 5, RED, -1, cv2.LINE_AA)

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 25
    x1, y1 = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
    x2, y2 = min(max(xs) + pad, w - 1), min(max(ys) + pad, h - 1)
    if opts["box"]:
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)

    n = count_fingers(lm, handedness)
    label = f"{handedness} hand | fingers: {n}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

    if opts["trails"] and hand_idx < 2:
        for t_i, tip in enumerate(TIP_IDS):
            trails[hand_idx][t_i].append(pts[tip])
            trail = trails[hand_idx][t_i]
            for k in range(1, len(trail)):
                thickness = max(1, int(k / len(trail) * 5))
                cv2.line(frame, trail[k - 1], trail[k], GREEN, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------- camera
def open_camera(camera_index=0):
    backends = (
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("automatic", cv2.CAP_ANY),
    )
    errors = []
    for backend_name, backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            errors.append(f"{backend_name}: could not open")
            cap.release()
            continue

        for _ in range(15):
            ok, frame = cap.read()
            if ok:
                print(f"Camera {camera_index} opened with {backend_name}")
                return cap, frame
            time.sleep(0.1)

        errors.append(f"{backend_name}: opened but returned no frames")
        cap.release()

    raise RuntimeError(
        f"Could not start camera {camera_index}. "
        "Check Windows camera privacy permissions and close other apps using "
        f"the camera. Attempts: {'; '.join(errors)}"
    )


# ---------------------------------------------------------------- main
def main():
    opts = {"trails": True, "glow": True, "box": False, "filter_on": True,
            "shape": "all"}
    filter_idx = 0
    prev = time.time()
    cap, frame = open_camera()
    try:
        model_path = get_model_path()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        with mp_vision.HandLandmarker.create_from_options(options) as hands:
            cv2.namedWindow("Hand Tracker", cv2.WINDOW_NORMAL)
            start_ms = time.monotonic_ns() // 1_000_000
            previous_timestamp_ms = -1
            while True:
                if frame is None:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera stopped returning frames")
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000 - start_ms,
                    previous_timestamp_ms + 1,
                )
                previous_timestamp_ms = timestamp_ms
                result = hands.detect_for_video(image, timestamp_ms)

                if result.hand_landmarks:
                    # pixel coordinates for every detected hand
                    all_pts = [
                        [(int(p.x * w), int(p.y * h)) for p in lms]
                        for lms in result.hand_landmarks
                    ]

                    # 1) filter the region between the hands (on the clean frame)
                    quad = get_frame_quad(all_pts, opts["shape"])
                    if quad is not None:
                        if opts["filter_on"]:
                            apply_filter_in_quad(frame, quad, FILTERS[filter_idx][1])
                        cv2.polylines(frame, [quad.reshape(-1, 1, 2)], True,
                                      WHITE, 2, cv2.LINE_AA)

                    # 2) draw hand skeletons on top
                    for i, (lms, pts, handedness) in enumerate(
                        zip(result.hand_landmarks, all_pts, result.handedness)
                    ):
                        side = handedness[0].category_name
                        draw_hand(frame, lms, pts, i, side, opts)
                else:
                    for hand in trails:
                        for trail in hand:
                            trail.clear()

                now = time.time()
                fps = 1 / max(now - prev, 1e-6)
                prev = now
                status = FILTERS[filter_idx][0] if opts["filter_on"] else "off"
                cv2.putText(frame,
                            f"FPS: {fps:.0f} | filter: {status} | shape: {opts['shape']}",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

                cv2.imshow("Hand Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("f"):
                    filter_idx = (filter_idx + 1) % len(FILTERS)
                elif key == ord("x"):
                    opts["filter_on"] = not opts["filter_on"]
                elif key == ord("m"):
                    opts["shape"] = "hull" if opts["shape"] == "all" else "all"
                elif key == ord("t"):
                    opts["trails"] = not opts["trails"]
                elif key == ord("g"):
                    opts["glow"] = not opts["glow"]
                elif key == ord("b"):
                    opts["box"] = not opts["box"]
                elif key == ord("s"):
                    name = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(name, frame)
                    print("Saved", name)
                frame = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()