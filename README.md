# Hand_Tracker
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
