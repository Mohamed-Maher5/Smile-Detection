"""Live webcam demo: classify on a fixed decision cadence and overlay the
result + latency plus the 5 face landmarks (kps) the model used.

This is the SINGLE live-camera entry point. It contains the two
previously separate behaviors behind one flag set:

  * DEFAULT — the GUIDED view: a green centered guide (fixed rectangle, or the
    inscribed oval with --guide-oval) is drawn and NEVER moves, independent of
    the detected face. The smile verdict is only produced while a single face is
    centered inside the guide AND its bounding-box area covers between
    --guide-min and --guide-max of the guide area — i.e. the subject is close
    enough that the geometric features are reliable but not so close the face
    fills the frame. Otherwise the screen says how to fix it. The guide adapts
    to the WINDOW size (it is drawn as a fraction of the display width), so the
    same seating distance works at any window scale.
  * `--no-square` — the classic whole-frame view: the smile check runs on any
    single face anywhere in the frame, and the detected face bounding box IS
    drawn.

Both views run the IDENTICAL inference pipeline — detection (on the camera's
NATIVE resolution) -> similarity face alignment (align_kps) -> the same 3
geometric features (mouth_width, mouth_vertical_lift, mouth_nose_ratio) -> the
same StandardScaler normalization -> the same LogisticRegression as the model
was trained on. Scaling for display happens only for drawing, never before
detection; upscaling the image before detection measurably degrades SCRFD
landmarks. There is no code duplication between the views — they are two
branches of one loop.

DECISION CADENCE (default 5 predictions per second): the camera captures at
~30 fps but the detect/classify step only runs on frames spaced ~1/--fps apart
(default every 200 ms, i.e. every 6th frame). The displayed verdict is a
rolling average of the last 5 predictions (~1 second of history) decided with
hysteresis (bands T+0.05 / T-0.05 around the cut), so borderline expressions
do not flicker.

DECISION LAYER (the model and features are untouched):
  * Rolling 5-frame average + hysteresis (default; see --debounce / --raw).
  * Default cut T = the production operating point (0.5 for the sklearn bundle).
  * --threshold raises/lowers the cut; --debounce K switches to debounced
    switching (flip only after K consecutive predictions agree); --raw removes
    EVERYTHING and labels the instantaneous probability each checked frame.
  * Low-confidence HOLD: when the single-face detection confidence drops below
    0.6 the fresh verdict is withheld (unreliable landmarks) and the last
    stable label stays on screen with a yellow border saying it is not from the
    current frame.

SAFETY (no guessing): when there is no single face properly centered inside the
guide at an acceptable size the verdict is withheld and the screen explains why
(NO FACE / MULTIPLE FACES / TOO FAR / TOO CLOSE / CENTER YOUR FACE) instead of
producing a label for an out-of-distribution frame.

LOGGING (optional): `--log PATH` appends one CSV row per captured frame with
frame_idx, n_faces, det_score, guide/gate state, bbox-area fraction, raw &
smoothed probabilities, the verdict, latency and the bbox/kps arrays — flushed
per frame, so an interrupted run loses nothing.

Keys: 'q' quit, 'c' reset the smoothed/verdict state.
"""

import argparse
import csv
import signal
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from features import align_kps
from smile_detector import detect_and_score_detailed, _BUNDLE

# The default decision threshold follows the loaded bundle: for an sklearn
# bundle it is the production 0.5 probability cut; for a threshold bundle the
# rule decides on a single aligned feature directly (e.g. mouth_eye_ratio >=
# 0.8357). --threshold overrides it.
SMILE_THRESHOLD = (
    float(_BUNDLE["model"])
    if _BUNDLE["type"] == "threshold"
    else 0.5
)

KPS_TAGS = ["LE", "RE", "N", "LM", "RM"]
KPS_COLOR = (255, 255, 0)        # keypoint dots (cyan)
BBOX_COLOR = (255, 255, 200)     # face bbox (light cyan-white) — whole-frame view
BBOX_THICKNESS = 2
SQUARE_COLOR = (0, 255, 0)       # guide box (green)
SQUARE_FILL = (0, 255, 0)        # face inside -> faint green wash over the guide
SQUARE_FILLED_ALPHA = 0.12
GATE_COLOR = (0, 165, 255)       # orange — waiting for the face to enter the guide
TOO_FAR_COLOR = (0, 165, 255)    # orange — face too far / too small in the guide
TOO_CLOSE_COLOR = (0, 0, 255)    # red — face too close / too big for the guide

# Fixed-guide defaults — calibrated on the real 14.4k-frame webcam log
# (data/exper/square_detect_log.csv; SCRFD bbox height ~ 2.84x inter-eye and
# width ~ 0.76x height, so bbox_area ~ 6.13 x inter_eye^2). With a 0.42-of-width
# guide on a 1280-wide display, the band min=0.15 / max=0.95 of guide area
# accepts inter-eye roughly 85..210 px — i.e. roughly 1x..2.5x the distance at
# which the face fills the guide. The user does NOT need to be very close: the
# far end of the band is deliberately generous. Trim --guide-max / raise
# --guide-min if you want a much closer/certainer operating distance.
GUIDE_SIZE_DEFAULT = 0.42    # static guide side as a fraction of display width
GUIDE_MIN_DEFAULT = 0.15     # lower bound: face bbox area / guide area (far end)
GUIDE_MAX_DEFAULT = 0.95     # upper bound: face bbox area / guide area (close end)
PREDICT_PER_SEC_DEFAULT = 5.0  # decision cadence (predictions per second)

# Stabilisation knobs (the underlying model/classification is not modified).
SMILE_HYSTERESIS = 0.05    # hold band: keep label inside [T-0.05, T+0.05]
WINDOW_SIZE = 5            # rolling average length for the smoothed decision
DET_CONF_THRESHOLD = 0.6   # hold result when the single-face det_score drops below this
HOLD_COLOR = (0, 255, 255)  # yellow (BGR) frame border for the hold indicator

# Requested capture resolution (webcams may not honour it exactly).
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# Display is decoupled from the capture resolution (see module docstring).
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720
WINDOW_TITLE = "Smile Detection"

EXP_DIR = Path(__file__).resolve().parents[1] / "data" / "exper"
DEFAULT_LOG = EXP_DIR / "webcam_demo_log.csv"
LOG_HEADER = [
    "frame_idx", "timestamp_ms", "checked", "n_faces", "det_score",
    "face_in_square", "gate_open", "bbox_frac", "raw", "smoothed", "verdict",
    "latency_ms", "bbox", "kps",
]


def _hysteresis_verdict(smoothed: float | None, prev: bool,
                        threshold: float) -> bool:
    """Map the smoothed probability to a label with a hold band.

    Flipping to True only when ``smoothed > T + 0.05``, to False only when
    ``smoothed < T - 0.05``, and otherwise keeping the previous label — so
    borderline values no longer toggle on a prediction-by-prediction basis.
    """
    if smoothed is None:
        return prev
    if smoothed > threshold + SMILE_HYSTERESIS:
        return True
    if smoothed < threshold - SMILE_HYSTERESIS:
        return False
    return prev


def _bbox_in_guide(bbox: np.ndarray, x1s: int, y1s: int, x2s: int, y2s: int,
                   strict: bool) -> bool:
    """True iff *bbox* (x1,y1,x2,y2) satisfies the centered-guide rule."""
    if bbox is None or len(bbox) != 4:
        return False
    bx1, by1, bx2, by2 = (float(v) for v in bbox)
    if not all(np.isfinite(v) for v in (bx1, by1, bx2, by2)):
        return False
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    if strict:
        return bx1 >= x1s and by1 >= y1s and bx2 <= x2s and by2 <= y2s
    return (x1s <= cx <= x2s) and (y1s <= cy <= y2s)


def _bbox_area(bbox: np.ndarray) -> float | None:
    """Face bounding-box area (display pixels) or None when unusable."""
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    return area if area > 0.0 else None


def _guide_area(side: int, oval: bool) -> float:
    """Guide area in display pixels (ellipse uses its inscribed area)."""
    if oval:
        return (np.pi / 4.0) * side * side
    return float(side * side)


def _draw_guide(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                face_in: bool, oval: bool) -> None:
    """Draw the fixed centered guide; faint green fill while a face is inside."""
    def shape(img, color, thickness):
        if oval:
            c = ((x1 + x2) // 2, (y1 + y2) // 2)
            r = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(img, c, r, 0, 0, 360, color, thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if face_in:
        overlay = frame.copy()
        shape(overlay, SQUARE_FILL, -1)
        cv2.addWeighted(overlay, SQUARE_FILLED_ALPHA, frame, 1 - SQUARE_FILLED_ALPHA,
                        0, frame)
    shape(frame, SQUARE_COLOR, 2)


def _draw_bbox(frame: np.ndarray, bbox: np.ndarray) -> None:
    """Overlay the detected face bounding box as a thin rectangle."""
    if bbox is None or len(bbox) != 4:
        return
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return
    pt1 = (int(round(x1)), int(round(y1)))
    pt2 = (int(round(x2)), int(round(y2)))
    if not (0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0]):
        return
    cv2.rectangle(frame, pt1, pt2, BBOX_COLOR, BBOX_THICKNESS, cv2.LINE_AA)


def _draw_kps(frame: np.ndarray, kps: np.ndarray) -> None:
    """Overlay each landmark as a small cyan dot with its tag next to it."""
    for tag, (px, py) in zip(KPS_TAGS, kps):
        px, py = float(px), float(py)
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        xi, yi = int(round(px)), int(round(py))
        if not (0 <= xi < frame.shape[1] and 0 <= yi < frame.shape[0]):
            continue
        cv2.circle(frame, (xi, yi), 4, KPS_COLOR, -1)
        cv2.putText(frame, tag, (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, KPS_COLOR, 1, cv2.LINE_AA)


def _draw_aligned(frame: np.ndarray, aligned: np.ndarray) -> None:
    """Overlay the aligned landmark coordinates (canonical eye frame) top-right."""
    if aligned is None:
        return
    h, w = frame.shape[:2]
    lines = [f"{tag}: " + ",".join(f"{c:+.3f}" for c in pt)
             for tag, pt in zip(KPS_TAGS, aligned)]
    y0 = 40
    line_h = 22
    panel_h = line_h * len(lines) + 10
    x0 = w - 240
    cv2.rectangle(frame, (x0 - 6, y0 - 34), (w - 4, y0 + panel_h - 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 200), 1, cv2.LINE_AA)


def _draw_status(frame: np.ndarray, text: str, color: tuple[int, int, int],
                 y: int = 40) -> None:
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


def _draw_face_label(frame: np.ndarray, bbox: np.ndarray, label: str,
                     color: tuple[int, int, int], scale: float = 1.0) -> None:
    """Draw *label* just above the top edge of *bbox*, outlined for readability."""
    if bbox is None or len(bbox) != 4:
        return
    x1 = float(bbox[0])
    y1 = float(bbox[1])
    if not (np.isfinite(x1) and np.isfinite(y1)):
        return
    thickness = 3 if scale >= 1.0 else 2
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tx = int(round(max(1, min(x1, frame.shape[1] - tw - 1))))
    ty = int(round(max(th + 4, y1 - 8)))
    cv2.rectangle(frame, (tx - 3, ty - th - 3),
                  (tx + tw + 3, ty + 3), (0, 0, 0), -1)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _guide_reason(n_faces: int, in_square: bool, r: float | None,
                  guide_min: float, guide_max: float) -> tuple[str, tuple[int, int, int]]:
    """Human reason + color for a closed gate, or (None, None) when open-capable."""
    if n_faces == 0:
        return "NO FACE", (128, 128, 128)
    if n_faces >= 2:
        return "MULTIPLE FACES", (128, 128, 128)
    if r is None:
        return "NO FACE", (128, 128, 128)
    if r < guide_min:
        return "TOO FAR  (move closer / enlarge window)", TOO_FAR_COLOR
    if r > guide_max:
        return "TOO CLOSE  (step back)", TOO_CLOSE_COLOR
    if not in_square:
        return "CENTER YOUR FACE", GATE_COLOR
    return "POSITION YOUR FACE IN THE BOX", GATE_COLOR


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live smile demo (single file): fixed centered guide by "
                    "default; --no-square for the classic whole-frame view. "
                    "Decisions at ~--fps predictions/second (default 5).")
    ap.add_argument("--no-square", action="store_true",
                    help="classic view: check any single face anywhere in the frame "
                         "and draw the face bounding box (no centered guide)")
    ap.add_argument("--guide-size", type=float, default=GUIDE_SIZE_DEFAULT,
                    help="fixed guide side as a fraction of display width "
                         "(default %(default)s)")
    ap.add_argument("--guide-min", type=float, default=GUIDE_MIN_DEFAULT,
                    help="acceptable lower bound: face bbox area / guide area — "
                         "below this is TOO FAR (default %(default)s)")
    ap.add_argument("--guide-max", type=float, default=GUIDE_MAX_DEFAULT,
                    help="acceptable upper bound: face bbox area / guide area — "
                         "above this is TOO CLOSE (default %(default)s)")
    ap.add_argument("--guide-oval", action="store_true",
                    help="draw the inscribed oval guide; center + area-band rules "
                         "then use the oval's inscribed area")
    ap.add_argument("--strict", action="store_true",
                    help="require the WHOLE face bbox inside the guide, not just its center")
    ap.add_argument("--fps", type=float, default=PREDICT_PER_SEC_DEFAULT,
                    help="decision cadence in predictions per second (default "
                         "%(default)s); 0 = classify every captured frame")
    ap.add_argument("--no-refine", action="store_true",
                    help="single-pass landmarks (skip the crop-and-refine stage): "
                         "faster (~4.5 vs ~7.5 ms/call) but ~2pp less accurate")
    ap.add_argument("--square-frac", type=float, default=None,
                    help="DEPRECATED alias for --guide-size (kept for back-compat)")
    ap.add_argument("--fixed", action="store_true",
                    help="DEPRECATED no-op: the guide is always fixed (kept for back-compat)")
    ap.add_argument("--raw", action="store_true",
                    help="remove stabilization: verdict = the instantaneous prediction "
                         "per checked frame (A/B control)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the decision threshold (default: the bundle's "
                         "production operating point, e.g. 0.5)")
    ap.add_argument("--debounce", type=int, default=0,
                    help="flip only after this many consecutive predictions agree "
                         "(0 = hysteresis, the default)")
    ap.add_argument("--log", nargs="?", const=str(DEFAULT_LOG), default=None,
                    help="append per-frame rows to <path> for analysis "
                         "(default path: %s)" % DEFAULT_LOG)
    args = ap.parse_args()

    guide_size = args.guide_size
    if args.no_refine:
        import smile_detector as _sd
        _sd.ENABLE_CROP_REFINE = False
    if args.square_frac is not None:
        print("NOTE: --square-frac is deprecated; use --guide-size.",
              file=sys.stderr)
        guide_size = args.square_frac
    if args.fixed:
        print("NOTE: --fixed is deprecated (the guide is always fixed).",
              file=sys.stderr)
    if not 0.1 <= guide_size <= 0.9:
        ap.error("guide size must be in [0.1, 0.9] (--guide-size / deprecated --square-frac)")
    if not 0.0 < args.guide_min < args.guide_max <= 2.0:
        ap.error("--guide-min and --guide-max must satisfy 0 < min < max <= 2.0")
    if args.fps < 0:
        ap.error("--fps must be >= 0")
    if args.debounce < 0:
        ap.error("--debounce must be >= 0")
    threshold = SMILE_THRESHOLD if args.threshold is None else args.threshold

    stop = False

    def _request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (cv2.VideoCapture(0))")
        return 1

    # Most webcams only expose 1280x720 under MJPEG; request it first, then the
    # size, so the desired resolution is honoured when the device supports it.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture resolution: requested {CAM_WIDTH}x{CAM_HEIGHT}, got {got_w}x{got_h}")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    log = writer = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        new_file = not Path(args.log).exists()
        log = open(args.log, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(log, fieldnames=LOG_HEADER)
        if new_file:
            writer.writeheader()
        log.flush()

    window = deque(maxlen=WINDOW_SIZE)  # rolling raw probabilities (checked frames)
    verdict = False                     # previous displayed label (NOT SMILING)
    up_count = 0                        # debounce: consecutive predictions >= threshold
    down_count = 0                      # debounce: consecutive predictions < threshold
    frame_idx = 0
    checked = 0                         # frames actually classified (cadence)
    prev_gate_open = False
    last_check_t = float("-inf")        # cadence anchor (first frame always checked)
    last_kps_d = last_bbox_d = last_aligned = last_kps = None

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    print("Press 'q' to quit, 'c' to reset the smoothed verdict.")
    while not stop:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("WARNING: failed to read frame, skipping")
            continue

        # Native capture kept for detection; the display copy is upscaled ONLY
        # for drawing (upscaling before detection degrades SCRFD landmarks).
        # The upscale is a single uniform factor so the display landmark/bbox
        # mapping (kps_d = kps * scale) never distorts.
        orig_h, orig_w = frame.shape[:2]
        scale = min(DISPLAY_WIDTH / orig_w, DISPLAY_HEIGHT / orig_h)
        scale = max(scale, 1.0)  # never downscale the capture itself
        disp_w, disp_h = max(int(round(orig_w * scale)), 1), max(int(round(orig_h * scale)), 1)
        if (disp_w, disp_h) != (orig_w, orig_h):
            display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        else:
            display = frame.copy()

        # Static centered guide geometry in DISPLAY coordinates — fixed, never
        # tied to the face; adapts proportionally to window size.
        side = int(round(guide_size * disp_w))
        x1s, y1s = (disp_w - side) // 2, (disp_h - side) // 2
        x2s, y2s = x1s + side, y1s + side
        guide_area = _guide_area(side, args.guide_oval)

        # Decision cadence: classify at most --fps frames per second (default 5
        # from a ~30 fps stream -> every 6th frame).  Skipped frames reuse the
        # last result for overlays only; they never enter the decision window.
        should_check = (args.fps <= 0) or (time.monotonic() - last_check_t >= 1.0 / args.fps)
        latency_ms = 0.0
        raw = smoothed = None
        if should_check:
            last_check_t = time.monotonic()
            checked += 1
            try:
                t0 = time.perf_counter()
                _, kps, raw, n_faces, det_score, bbox = detect_and_score_detailed(frame)
                latency_ms = (time.perf_counter() - t0) * 1e3
            except Exception as err:
                print(f"WARNING: pipeline failed on frame: {err!r}")
                continue
            last_kps = kps
            last_aligned = align_kps(kps) if kps is not None else None
            last_kps_d = None if kps is None else np.asarray(kps, dtype=float) * scale
            last_bbox_d = None if bbox is None else np.asarray(bbox, dtype=float) * scale
            last_n_faces, last_det = n_faces, det_score
        else:
            # Skipped frame: reuse the last checked geometry for overlays; the
            # decision window is not touched.
            n_faces = last_n_faces if last_kps_d is not None else 0
            det_score = last_det if last_kps_d is not None else None
            raw = smoothed = None
        kps_d = last_kps_d if last_kps_d is not None else None
        bbox_d = last_bbox_d if last_bbox_d is not None else None
        aligned = last_aligned if last_aligned is not None else None

        # Gate: in the guided view the verdict requires a single face centered
        # inside the guide at an acceptable size; in the classic view any single
        # face anywhere qualifies — one code path.
        r = None
        if args.no_square:
            in_square = n_faces == 1
            band_ok = True
        else:
            in_square = _bbox_in_guide(bbox_d, x1s, y1s, x2s, y2s, args.strict)
            bbox_a = _bbox_area(bbox_d)
            r = bbox_a / guide_area if bbox_a is not None else None
            band_ok = r is not None and args.guide_min <= r <= args.guide_max
        gate_open = (n_faces == 1 and band_ok and (args.no_square or in_square))
        hold_active = (not args.raw and gate_open and det_score is not None
                       and det_score < DET_CONF_THRESHOLD)

        # Decision layer — identical regardless of view; only runs on checked
        # frames so the cadence paces the verdicts.
        if should_check:
            if args.raw:
                if gate_open and raw is not None:
                    verdict = bool(raw >= threshold)
                elif gate_open and raw is None:
                    verdict = False
                window.clear()
                smoothed = raw
            else:
                if gate_open and not hold_active and raw is not None:
                    if not prev_gate_open:
                        window.clear()  # fresh history when the gate re-opens
                    window.append(raw)
                smoothed = float(np.mean(window)) if window else None

                if not gate_open:
                    up_count = 0
                    down_count = 0

                if args.debounce > 0 and gate_open and not hold_active and raw is not None:
                    if smoothed is not None and smoothed >= threshold:
                        up_count += 1
                        down_count = 0
                    elif smoothed is not None:
                        down_count += 1
                        up_count = 0
                    if up_count >= args.debounce:
                        verdict = True
                    elif down_count >= args.debounce:
                        verdict = False
                elif gate_open and raw is not None:
                    verdict = _hysteresis_verdict(smoothed, verdict, threshold)
                elif gate_open and raw is None:
                    verdict = False
        prev_gate_open = gate_open

        # On skipped frames keep displaying the live rolling mean (decision
        # window untouched — only the verdict/cadence are gated).
        if smoothed is None and not args.raw:
            smoothed = float(np.mean(window)) if window else None

        if writer is not None:
            writer.writerow({
                "frame_idx": frame_idx, "timestamp_ms": f"{time.time() * 1e3:.1f}",
                "checked": int(should_check), "n_faces": n_faces,
                "det_score": f"{det_score:.4f}" if det_score is not None else "",
                "face_in_square": int(in_square), "gate_open": int(gate_open),
                "bbox_frac": f"{r:.4f}" if r is not None else "",
                "raw": f"{raw:.4f}" if raw is not None else "",
                "smoothed": f"{smoothed:.4f}" if smoothed is not None else "",
                "verdict": int(verdict),
                "latency_ms": f"{latency_ms:.2f}",
                "bbox": "" if bbox_d is None else np.array2string(np.asarray(bbox_d, dtype=float), precision=1),
                "kps": "" if kps_d is None else np.array2string(np.asarray(kps_d, dtype=float), precision=1),
            })
            log.flush()  # interrupted run (Ctrl-C / crash) never loses the tail
            frame_idx += 1

        # Overlays — always landmarks + aligned panel; the box differs by view.
        if not args.no_square:
            _draw_guide(display, x1s, y1s, x2s, y2s, gate_open, args.guide_oval)
        elif bbox_d is not None:
            _draw_bbox(display, bbox_d)
        if kps_d is not None:
            _draw_kps(display, kps_d)
        _draw_aligned(display, aligned)

        # Status text and probability lines. In both views the smile verdict
        # (and its probabilities) only make sense while gate_open.
        if not gate_open:
            if args.no_square:
                reason = ("NO FACE" if n_faces == 0
                          else "MULTIPLE FACES" if n_faces >= 2
                          else "NO FACE")
                _draw_status(display, reason, (128, 128, 128))
                cv2.putText(display, f"Faces detected: {n_faces}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                reason, color = _guide_reason(n_faces, in_square, r,
                                              args.guide_min, args.guide_max)
                _draw_status(display, reason, color)
                cv2.putText(display, f"Faces detected: {n_faces}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, "Position your face in the box", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            label = "SMILING" if verdict else "NOT SMILING"
            color = (0, 255, 0) if verdict else (0, 0, 255)
            if args.no_square and bbox_d is not None:
                _draw_face_label(display, bbox_d, label, color, scale=1.0)
            else:
                _draw_status(display, label, color)
            if raw is not None:
                cv2.putText(display, f"raw: {raw:.2f}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if args.raw:
                    cv2.putText(display, "MODE=RAW (no smoothing)", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(display, f"smoothed: {smoothed:.2f}" if smoothed is not None
                                else "smoothed: --", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display, f"{latency_ms:.1f} ms", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if args.raw:
            mode_tag = "MODE=RAW"
        elif args.debounce > 0:
            mode_tag = f"MODE=DEBOUNCE K={args.debounce}"
        else:
            mode_tag = "MODE=HYST"
        if args.no_square:
            mode_tag += " WHOLE-FRAME"
        rate_tag = "every frame" if args.fps <= 0 else f"{args.fps:g}/s"
        frac_tag = f"guide={r * 100:.0f}%" if r is not None else "guide=--"
        cv2.putText(display, f"guide {side}px [{frac_tag}] | rate {rate_tag} "
                    f"{'strict' if args.strict else 'center'} | {mode_tag}",
                    (20, disp_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, SQUARE_COLOR, 2)

        if hold_active:
            cv2.rectangle(display, (0, 0), (disp_w - 1, disp_h - 1), HOLD_COLOR, 4)
            cv2.putText(display, "HELD (low-confidence detection)", (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, HOLD_COLOR, 2)

        cv2.imshow(WINDOW_TITLE, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            window.clear()
            verdict = False
            print("reset smoothed verdict")

    if log is not None:
        log.close()
    cap.release()
    cv2.destroyAllWindows()
    if args.log:
        print(f"Frame log appended to {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())