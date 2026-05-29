"""Real-time MediaPipe gesture inference with a rolling prediction buffer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "gesture_classification"
    / "checkpoints"
    / "gesture_recognizer.task"
)


@dataclass(frozen=True)
class GestureSample:
    timestamp: float
    scores: dict[str, float]
    top_class: str
    top_score: float


def normalize_class_name(name: str | None) -> str:
    name = (name or "").strip().lower()
    return name if name else "none"


def normalize_score_vector(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in scores.values())
    if total <= 0.0:
        return {"none": 1.0}
    return {key: max(value, 0.0) / total for key, value in scores.items()}


def aggregate_gesture_scores(
    samples: Iterable[GestureSample],
    class_names: Iterable[str],
    threshold: float,
) -> dict[str, float]:
    """Aggregate frame-level predictions into one normalized class vector."""
    totals = {normalize_class_name(name): 0.0 for name in class_names}
    totals.setdefault("none", 0.0)
    frame_count = 0

    for sample in samples:
        frame_count += 1
        top_class = normalize_class_name(sample.top_class)
        if top_class == "none" or sample.top_score < threshold:
            totals["none"] += 1.0
            continue

        for name, score in sample.scores.items():
            class_name = normalize_class_name(name)
            if class_name == "none":
                continue
            totals.setdefault(class_name, 0.0)
            totals[class_name] += max(score, 0.0)

    if frame_count == 0:
        return {"none": 1.0}
    return normalize_score_vector(totals)


class RealtimeGestureRecognizer:
    """Continuously runs gesture inference and stores recent predictions."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        camera_index: int = 0,
        buffer_seconds: float = 20.0,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._model_path = Path(model_path).expanduser().resolve()
        self._camera_index = camera_index
        self._buffer_seconds = buffer_seconds
        self._num_hands = num_hands
        self._det_conf = min_hand_detection_confidence
        self._pres_conf = min_hand_presence_confidence
        self._track_conf = min_tracking_confidence
        self._width = width
        self._height = height

        self._samples: deque[GestureSample] = deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Gesture model not found: {self._model_path}")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gesture-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def samples_between(self, start: float, end: float) -> list[GestureSample]:
        with self._lock:
            return [sample for sample in self._samples if start <= sample.timestamp <= end]

    def aggregate_window(
        self,
        start: float,
        end: float,
        class_names: Iterable[str],
        threshold: float,
    ) -> dict[str, float]:
        return aggregate_gesture_scores(self.samples_between(start, end), class_names, threshold)

    def _append_sample(self, sample: GestureSample) -> None:
        cutoff = sample.timestamp - self._buffer_seconds
        with self._lock:
            self._samples.append(sample)
            while self._samples and self._samples[0].timestamp < cutoff:
                self._samples.popleft()

    def _run(self) -> None:
        try:
            self._run_camera_loop()
        except BaseException as exc:
            self._error = exc
            print(f"[gesture] worker stopped: {exc}")

    def _run_camera_loop(self) -> None:
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizerMP = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOpts = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOpts(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self._num_hands,
            min_hand_detection_confidence=self._det_conf,
            min_hand_presence_confidence=self._pres_conf,
            min_tracking_confidence=self._track_conf,
        )

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self._camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        print("[gesture] realtime worker active")
        try:
            with GestureRecognizerMP.create_from_options(options) as recognizer:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        print("[gesture] camera frame read failed")
                        time.sleep(0.05)
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp = time.monotonic()
                    result = recognizer.recognize_for_video(mp_img, int(timestamp * 1000))
                    self._append_sample(_sample_from_result(result, timestamp))
        finally:
            cap.release()


def _sample_from_result(result, timestamp: float) -> GestureSample:
    if not result.gestures or not result.gestures[0]:
        return GestureSample(timestamp=timestamp, scores={"none": 1.0}, top_class="none", top_score=1.0)

    scores: dict[str, float] = {}
    for category in result.gestures[0]:
        scores[normalize_class_name(category.category_name)] = float(category.score)

    top = result.gestures[0][0]
    top_class = normalize_class_name(top.category_name)
    top_score = float(top.score)
    return GestureSample(timestamp=timestamp, scores=scores, top_class=top_class, top_score=top_score)
