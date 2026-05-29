"""MediaPipe gesture recognizer — importable class + standalone demo.

Importable usage (from main.py):
    from inference.infer_rpi import GestureRecognizer
    rec = GestureRecognizer(model_path=..., camera_index=0, on_gesture=callback)
    rec.start()   # runs in background thread
    rec.stop()

Standalone demo:
    python inference/infer_rpi.py
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import mediapipe as mp


DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "checkpoints" / "gesture_recognizer.task")


def _top_gesture(result) -> tuple[str, float]:
    """Return (category_name, score) for the top gesture, or ('None', 0.0)."""
    if not result.gestures or not result.gestures[0]:
        return "None", 0.0
    top = result.gestures[0][0]
    return top.category_name, top.score


# ---------------------------------------------------------------------------
# Importable class
# ---------------------------------------------------------------------------

class GestureRecognizer:
    """Runs MediaPipe gesture recognition in a background thread.

    Calls on_gesture(name, confidence) once each time the detected gesture
    changes — debounced so it does not fire on every camera frame.
    """

    def __init__(
        self,
        model_path: str,
        on_gesture: Callable[[str, float], None],
        camera_index: int = 0,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._model_path  = Path(model_path).expanduser().resolve()
        self._on_gesture  = on_gesture
        self._camera_index = camera_index
        self._num_hands   = num_hands
        self._det_conf    = min_hand_detection_confidence
        self._pres_conf   = min_hand_presence_confidence
        self._track_conf  = min_tracking_confidence
        self._width       = width
        self._height      = height

        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Gesture model not found: {self._model_path}")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gesture-rec")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        BaseOptions           = mp.tasks.BaseOptions
        GestureRecognizerMP   = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOpts = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

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

        last_gesture = "None"
        print("[gesture] Camera active")

        with GestureRecognizerMP.create_from_options(options) as recognizer:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    print("[gesture] Camera frame read failed; stopping")
                    break

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = int(time.monotonic() * 1000)
                result = recognizer.recognize_for_video(mp_img, ts_ms)
                name, score = _top_gesture(result)

                # Fire callback only when gesture changes to avoid flooding
                if name != last_gesture:
                    last_gesture = name
                    if name != "None":
                        self._on_gesture(name, score)

        cap.release()


# ---------------------------------------------------------------------------
# Standalone demo helpers
# ---------------------------------------------------------------------------

def _format_top_prediction(result) -> str:
    name, score = _top_gesture(result)
    if name == "None":
        return "No gesture"
    handedness = ""
    if result.handedness and result.handedness[0]:
        hand = result.handedness[0][0]
        handedness = f" ({hand.category_name}, {hand.score:.2f})"
    return f"{name}: {score:.2f}{handedness}"


def _draw_prediction(frame, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (20, 20, 20), -1)
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)


def _bounded_confidence(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0.0 and 1.0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live webcam inference with a MediaPipe gesture model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--num-hands", type=_positive_int, default=1)
    parser.add_argument("--min-hand-detection-confidence", type=_bounded_confidence, default=0.5)
    parser.add_argument("--min-hand-presence-confidence",  type=_bounded_confidence, default=0.5)
    parser.add_argument("--min-tracking-confidence",       type=_bounded_confidence, default=0.5)
    parser.add_argument("--width",  type=_positive_int, default=640)
    parser.add_argument("--height", type=_positive_int, default=480)
    parser.add_argument("--print-only", action="store_true",
                        help="Do not open a display window; print predictions to the terminal.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    BaseOptions           = mp.tasks.BaseOptions
    GestureRecognizerMP   = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOpts = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = GestureRecognizerOpts(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    last_printed: Optional[str] = None
    print(f"Loaded model: {model_path}")
    print("Press q or Esc to quit.")

    with GestureRecognizerMP.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed; stopping.")
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int(time.monotonic() * 1000)
            result = recognizer.recognize_for_video(mp_img, ts_ms)
            prediction = _format_top_prediction(result)

            if args.print_only:
                if prediction != last_printed:
                    print(prediction)
                    last_printed = prediction
            else:
                _draw_prediction(frame, prediction)
                cv2.imshow("MediaPipe Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    cap.release()
    if not args.print_only:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
