"""Event-driven Raspberry Pi orchestrator for MyLittleRobotFriend."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

from core import (
    action_for_class,
    choose_class,
    class_names_from_maps,
    extract_last_word,
    fuse_scores,
    load_command_dict,
    save_command_dict,
    stt_vector,
)
from gesture_inference import DEFAULT_MODEL_PATH, RealtimeGestureRecognizer
from spike_comm import ALLOWED_COMMANDS, SpikeConnection
from stt import DEFAULT_MODEL_DIR, RealtimeSpeechRecognizer, print_input_devices


ROOT = Path(__file__).resolve().parent
COMMAND_DICT_PATH = ROOT / "command_dict.json"


class DryRunSpike:
    """Typed-signal stand-in for SPIKE hardware during local testing."""

    def __init__(self) -> None:
        self._signals: queue.Queue[str] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        print("[dryrun] SPIKE disabled. Type left_button, right_button, center_button, or q.")

    def start_receiver(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._read_stdin, daemon=True, name="dryrun-spike")
        self._thread.start()

    def get_signal(self, timeout: float | None = None) -> str | None:
        try:
            return self._signals.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_command(self, command: str) -> None:
        if command not in ALLOWED_COMMANDS:
            raise ValueError(f"Invalid SPIKE command: {command}")
        print(f"[dryrun] would send: {command}")

    def close(self) -> None:
        self._running = False

    def _read_stdin(self) -> None:
        while self._running:
            raw = input("> ").strip()
            if raw in {"q", "quit", "exit"}:
                raw = "center_button"
            if raw:
                self._signals.put(raw)


class RobotOrchestrator:
    def __init__(
        self,
        gesture_recognizer: RealtimeGestureRecognizer,
        speech_recognizer: RealtimeSpeechRecognizer,
        spike,
        command_dict: dict[str, str],
        capture_seconds: float,
        audio_weight: float,
        gesture_weight: float,
        gesture_threshold: float,
    ) -> None:
        self.gesture_recognizer = gesture_recognizer
        self.speech_recognizer = speech_recognizer
        self.spike = spike
        self.command_dict = command_dict
        self.capture_seconds = capture_seconds
        self.audio_weight = audio_weight
        self.gesture_weight = gesture_weight
        self.gesture_threshold = gesture_threshold
        self._busy = False

    def handle_signal(self, signal: str) -> bool:
        signal = signal.strip()
        if signal == "center_button":
            print("[main  ] center button received; stopping")
            return False
        if signal not in {"left_button", "right_button"}:
            print(f"[main  ] ignoring unknown signal: {signal}")
            return True
        if self._busy:
            print(f"[main  ] busy; ignoring {signal}")
            return True

        self._busy = True
        try:
            if signal == "left_button":
                self._run_interaction_window()
            else:
                self._run_learning_window()
        finally:
            self._busy = False
        return True

    def _collect_window(self) -> tuple[float, float, str, dict[str, float]]:
        start = time.monotonic()
        print(f"[main  ] running realtime inference window for {self.capture_seconds:.1f}s")
        time.sleep(self.capture_seconds)
        end = time.monotonic()

        class_names = class_names_from_maps(self.command_dict)
        last_word = self.speech_recognizer.last_word_between(start, end)
        gesture_scores = self.gesture_recognizer.aggregate_window(
            start=start,
            end=end,
            class_names=class_names,
            threshold=self.gesture_threshold,
        )
        return start, end, last_word, gesture_scores

    def _run_interaction_window(self) -> None:
        _, _, last_word, gesture_scores = self._collect_window()
        class_names = class_names_from_maps(self.command_dict)
        audio_scores = stt_vector(last_word, self.command_dict, class_names)
        fused = fuse_scores(audio_scores, gesture_scores, self.audio_weight, self.gesture_weight)
        gesture_class = choose_class(fused)
        action = action_for_class(gesture_class)

        print(f"[main  ] stt last word: {last_word}")
        print(f"[main  ] gesture scores: {_format_scores(gesture_scores)}")
        print(f"[main  ] fused scores: {_format_scores(fused)}")
        print(f"[main  ] selected: {gesture_class} -> {action}")
        self.spike.send_command(action)

    def _run_learning_window(self) -> None:
        _, _, last_word, gesture_scores = self._collect_window()
        learned_key = extract_last_word(last_word)
        gesture_class = choose_class(gesture_scores)

        if learned_key == "none":
            self.command_dict["none"] = "none"
            save_command_dict(self.command_dict, COMMAND_DICT_PATH)
            print("[learn ] no usable STT word heard; kept reserved 'none' -> 'none'")
            return

        self.command_dict[learned_key] = gesture_class
        save_command_dict(self.command_dict, COMMAND_DICT_PATH)

        print(f"[learn ] saved '{learned_key}' -> '{gesture_class}' in {COMMAND_DICT_PATH.name}")


def _format_scores(scores: dict[str, float]) -> str:
    nonzero = {key: value for key, value in scores.items() if value > 0.0}
    return ", ".join(f"{key}={value:.2f}" for key, value in sorted(nonzero.items())) or "none=1.00"


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _bounded_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0.0 and 1.0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Raspberry Pi integration for MyLittleRobotFriend.")
    parser.add_argument("--gesture-model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--vosk-model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mic-device", type=int, default=None)
    parser.add_argument("--spike-mac", default=None, help="Optional SPIKE Hub MAC address; otherwise choose interactively.")
    parser.add_argument("--capture-seconds", type=_positive_float, default=5.0)
    parser.add_argument("--audio-weight", type=_bounded_float, default=0.5)
    parser.add_argument("--gesture-weight", type=_bounded_float, default=0.5)
    parser.add_argument("--gesture-threshold", type=_bounded_float, default=0.7)
    parser.add_argument("--dry-run", action="store_true", help="Use typed button signals and print outgoing commands.")
    parser.add_argument("--list-mic-devices", action="store_true")
    parser.add_argument("--partials", action="store_true", help="Print Vosk partial results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_mic_devices:
        print_input_devices()
        return 0

    command_dict = load_command_dict(COMMAND_DICT_PATH)
    buffer_seconds = max(20.0, args.capture_seconds + 2.0)
    gesture = RealtimeGestureRecognizer(
        model_path=args.gesture_model,
        camera_index=args.camera,
        buffer_seconds=buffer_seconds,
    )
    speech = RealtimeSpeechRecognizer(
        model_dir=args.vosk_model,
        device_index=args.mic_device,
        buffer_seconds=buffer_seconds,
        emit_partials=args.partials,
    )
    spike = DryRunSpike() if args.dry_run else SpikeConnection()

    orchestrator = RobotOrchestrator(
        gesture_recognizer=gesture,
        speech_recognizer=speech,
        spike=spike,
        command_dict=command_dict,
        capture_seconds=args.capture_seconds,
        audio_weight=args.audio_weight,
        gesture_weight=args.gesture_weight,
        gesture_threshold=args.gesture_threshold,
    )

    try:
        gesture.start()
        speech.start()
        spike.connect() if args.dry_run else spike.connect(args.spike_mac)
        spike.start_receiver()

        print("[main  ] ready: left=interact, right=learn, center=stop")
        running = True
        while running:
            signal = spike.get_signal(timeout=0.25)
            if gesture.error:
                raise RuntimeError(f"gesture worker failed: {gesture.error}") from gesture.error
            if speech.error:
                raise RuntimeError(f"speech worker failed: {speech.error}") from speech.error
            if signal is not None:
                running = orchestrator.handle_signal(signal)
    except KeyboardInterrupt:
        print("\n[main  ] stopped by user")
    finally:
        spike.close()
        speech.stop()
        gesture.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
