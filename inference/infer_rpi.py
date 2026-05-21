"""Run webcam inference with a MediaPipe gesture_recognizer.task model.

This script is intended for Raspberry Pi or any machine with a camera that
OpenCV can read. Press q or Esc to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp


DEFAULT_MODEL_PATH = "checkpoints/gesture_recognizer.task"


def bounded_confidence(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0.0 and 1.0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live webcam inference with a MediaPipe gesture model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to gesture_recognizer.task.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--num-hands",
        type=positive_int,
        default=1,
        help="Maximum hands to detect. Keep this at 1 for Raspberry Pi speed.",
    )
    parser.add_argument(
        "--min-hand-detection-confidence",
        type=bounded_confidence,
        default=0.5,
    )
    parser.add_argument(
        "--min-hand-presence-confidence",
        type=bounded_confidence,
        default=0.5,
    )
    parser.add_argument("--min-tracking-confidence", type=bounded_confidence, default=0.5)
    parser.add_argument(
        "--width",
        type=positive_int,
        default=640,
        help="Requested camera frame width.",
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=480,
        help="Requested camera frame height.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Do not open a display window; print predictions to the terminal.",
    )
    return parser


def format_top_prediction(result: mp.tasks.vision.GestureRecognizerResult) -> str:
    if not result.gestures or not result.gestures[0]:
        return "No gesture"

    top_gesture = result.gestures[0][0]
    handedness = ""
    if result.handedness and result.handedness[0]:
        hand = result.handedness[0][0]
        handedness = f" ({hand.category_name}, {hand.score:.2f})"

    return f"{top_gesture.category_name}: {top_gesture.score:.2f}{handedness}"


def draw_prediction(frame, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    last_printed: Optional[str] = None
    print(f"Loaded model: {model_path}")
    print("Press q or Esc to quit.")

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed; stopping.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.monotonic() * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            prediction = format_top_prediction(result)

            if args.print_only:
                if prediction != last_printed:
                    print(prediction)
                    last_printed = prediction
            else:
                draw_prediction(frame, prediction)
                cv2.imshow("MediaPipe Gesture Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    cap.release()
    if not args.print_only:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
