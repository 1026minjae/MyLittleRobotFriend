"""Standalone microphone smoke test for the reusable Vosk recognizer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent / "vosk-model-small-en-us-0.15")


def _load_recognizer_tools():
    try:
        from vosk_stt.recognizer import AudioRecognizer, print_input_devices
    except ModuleNotFoundError as exc:
        if exc.name != "vosk_stt":
            raise
        from recognizer import AudioRecognizer, print_input_devices

    return AudioRecognizer, print_input_devices


def _print_text(text: str) -> None:
    print(f"TEXT: {text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Vosk speech recognition.")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help="Path to the Vosk model directory.")
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index.")
    parser.add_argument("--partials", action="store_true", help="Print partial recognition results.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    AudioRecognizer, print_input_devices = _load_recognizer_tools()

    if args.list_devices:
        print_input_devices()
        return

    recognizer = AudioRecognizer(
        model_dir=args.model,
        device_index=args.device,
        on_text=_print_text,
        emit_partials=args.partials,
    )

    recognizer.start()
    print("Speak now. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        recognizer.stop()


if __name__ == "__main__":
    main()
