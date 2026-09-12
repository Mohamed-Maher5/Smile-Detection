"""
Direct-onnxruntime face detector for the liveness service.

Replaces the heavy `insightface.app.FaceAnalysis` wrapper for the one thing the
liveness checks actually need: 5-point face landmarks (kps) + a detection score.
The buffalo_sc detector is an SCRFD model (`det_500m.onnx`); this module downloads
that model and runs it directly with onnxruntime, re-implementing SCRFD's
pre/post-processing (port of insightface's scrfd.py).
"""

import os
import zipfile
import logging
import urllib.request
import threading
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
# Anchor to the repo root so the cached model is shared no matter the CWD
# (running from notebooks/ must not spawn a second copy under notebooks/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(PROJECT_ROOT, "livenees_models")
BUFFALO_DIR = os.path.join(MODELS_DIR, "buffalo_sc")
DET_MODEL   = os.path.join(BUFFALO_DIR, "det_500m.onnx")
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"

DET_SIZE   = (160, 160)   # matches the previous app.prepare(det_size=(160, 160))
DET_THRESH = 0.5          # FaceAnalysis default detection threshold
NMS_THRESH = 0.4          # SCRFD default NMS threshold
_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


# ── Model download ───────────────────────────────────────────────────────────
def download_buffalo_sc(
    url: str = BUFFALO_URL,
    dest_dir: str = BUFFALO_DIR,
    model_path: str = DET_MODEL,
) -> str:
    """Download buffalo_sc.zip and extract `det_500m.onnx` (idempotent / cached)."""
    if os.path.exists(model_path):
        return model_path

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "buffalo_sc.zip")
    logger.info("Downloading buffalo_sc model from %s", url)
    urllib.request.urlretrieve(url, zip_path)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            member = next(
                (m for m in zf.namelist() if m.endswith("det_500m.onnx")), None
            )
            if member is None:
                raise RuntimeError(f"det_500m.onnx not found inside {url}")
            # Flatten: write to model_path regardless of any parent dir in the zip.
            with zf.open(member) as src, open(model_path, "wb") as dst:
                dst.write(src.read())
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    logger.info("Saved detection model to %s", model_path)
    return model_path


# ── Face container ───────────────────────────────────────────────────────────
@dataclass
class Face:
    """Minimal stand-in for insightface's Face: exposes the fields consumers use."""
    bbox: np.ndarray          # (4,)  x1, y1, x2, y2
    kps: np.ndarray           # (5, 2) left_eye, right_eye, nose, left_mouth, right_mouth
    det_score: float


# ── SCRFD decode helpers (port of insightface.model_zoo.scrfd) ────────────────
def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


# ── Detector ─────────────────────────────────────────────────────────────────
class SCRFDDetector:
    def __init__(
        self,
        model_path: str = DET_MODEL,
        det_size: tuple = DET_SIZE,
        det_thresh: float = DET_THRESH,
        nms_thresh: float = NMS_THRESH,
        num_threads: int | None = None,
    ):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime")

        download_buffalo_sc(model_path=model_path)
        # det_500m bakes in 640-input anchor counts; feeding 160x160 yields fewer
        # anchors and ORT logs benign output-shape mismatch warnings -> silence them.
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # Pin the intra/inter-op thread pools (e.g. num_threads=1 for a 1-CPU
        # container) to avoid thread oversubscription and make timings representative.
        if num_threads is not None:
            so.intra_op_num_threads = num_threads
            so.inter_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.center_cache: dict = {}
        self._init_model_vars()

    def _init_model_vars(self) -> None:
        """Infer SCRFD topology from the number of outputs (as insightface does)."""
        num_outputs = len(self.output_names)
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = False
        if num_outputs == 6:
            pass
        elif num_outputs == 9:
            self.use_kps = True
        elif num_outputs == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif num_outputs == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def _forward(self, det_img: np.ndarray, thresh: float):
        scores_list, bboxes_list, kpss_list = [], [], []
        input_size = tuple(det_img.shape[0:2][::-1])  # (w, h)
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height, input_width = blob.shape[2], blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            kps_preds = net_outs[idx + fmc * 2] * stride if self.use_kps else None

            height, width = input_height // stride, input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def _nms(self, dets: np.ndarray) -> list:
        thresh = self.nms_thresh
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(self, img: np.ndarray) -> list:
        """Detect faces and return a list of `Face`, sorted by det_score desc."""
        input_size = self.det_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]

        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self._forward(det_img, self.det_thresh)
        if not scores_list or sum(len(s) for s in scores_list) == 0:
            return []

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]

        kpss = None
        if self.use_kps:
            kpss = (np.vstack(kpss_list) / det_scale)[order, :, :][keep, :, :]

        faces = []
        for i in range(det.shape[0]):
            kps = kpss[i] if kpss is not None else None
            faces.append(Face(bbox=det[i, 0:4], kps=kps, det_score=float(det[i, 4])))
        return faces


def get_detector() -> SCRFDDetector:
    """Load the SCRFD detector lazily and reuse it across requests."""
    global _DETECTOR
    if _DETECTOR is None:
        with _DETECTOR_LOCK:
            if _DETECTOR is None:
                logger.info("Loading SCRFD face detector …")
                _DETECTOR = SCRFDDetector()
    return _DETECTOR
