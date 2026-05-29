"""
Raspberry Pi orchestrator for MyLittleRobotFriend.

Two modes:
  Normal mode   — a gesture or audio command triggers a robot action.
  Learning mode — an unrecognized audio word paired with a recent valid
                  gesture is added to the audio map and persisted to disk.
                  New gestures are never added (gesture labels are fixed by
                  the trained model).

Usage:
  python main.py
  python main.py --learning-mode
  python main.py --dry-run          # skip Bluetooth, print commands only
  python main.py --list-mic-devices
  python main.py --help
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Command vocabulary
# ---------------------------------------------------------------------------

AUDIO_MAP_PATH = Path("audio_to_action.json")

VALID_ACTIONS = {
    "walk_forward",
    "walk_backward",
    "left_turn",
    "right_turn",
    "fist_bump",
    "powerful_dance",
    "moderate_dance",
    "handshake",
    "raise_right_arm",
    "shake_head",
    "see_arround",
    "idle",
}

ACTION_ALIASES = {
    "walk": "walk_forward",
}

# Seed audio→action pairs.  Expanded at runtime in learning mode.
# Keys are the words a user would say aloud.
DEFAULT_AUDIO_MAP: dict[str, str] = {
    "walk":      "walk_forward",
    "forward":   "walk_forward",
    "back":      "walk_backward",
    "backward":  "walk_backward",
    "left":      "left_turn",
    "right":     "right_turn",
    "bump":      "fist_bump",
    "dance":     "powerful_dance",
    "groove":    "moderate_dance",
    "handshake": "handshake",
    "wave":      "raise_right_arm",
    "shake":     "shake_head",
    "look":      "see_arround",
}

# Gesture label → action.
# Keys must match the category_name values your trained model outputs.
# The current checked-in model contains: none, dance, fist_bump, turn_around.
DEFAULT_GESTURE_MAP: dict[str, str] = {
    "dance":      "powerful_dance",
    "fist_bump":  "fist_bump",
    "turn_around": "left_turn",

    # Backward-compatible aliases for older/future training labels.
    "walk_gesture":  "walk_forward",
    "walk_forward":  "walk_forward",
    "walk_backward": "walk_backward",
    "left_gesture":  "left_turn",
    "right_gesture": "right_turn",
    "fist_gesture":  "fist_bump",
    "dance_gesture": "powerful_dance",
    "wave_gesture":  "raise_right_arm",
    "shake_gesture": "shake_head",
    "look_gesture":  "see_arround",
}


def normalize_action(action: str) -> str:
    return ACTION_ALIASES.get(action, action)


def load_audio_map() -> dict[str, str]:
    audio_map = dict(DEFAULT_AUDIO_MAP)
    if AUDIO_MAP_PATH.exists():
        loaded = json.loads(AUDIO_MAP_PATH.read_text())
        audio_map.update(loaded)

    clean_map = {}
    for word, action in audio_map.items():
        normalized = normalize_action(action)
        if normalized in VALID_ACTIONS:
            clean_map[word] = normalized
    return clean_map


def save_audio_map(audio_map: dict[str, str]) -> None:
    AUDIO_MAP_PATH.write_text(json.dumps(audio_map, indent=2))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

GESTURE_CONFIDENCE = 0.70  # minimum score to accept a gesture prediction
ACTION_COOLDOWN    = 1.5   # minimum seconds between successive robot actions
LEARN_WINDOW       = 4.0   # max seconds between gesture and audio to pair them


class RobotOrchestrator:
    """Wires gesture and audio recognition to SPIKE Hub commands.

    Thread-safe: on_gesture and on_audio are called from background threads.
    """

    def __init__(self, learning_mode: bool, dry_run: bool) -> None:
        self.gesture_map   = dict(DEFAULT_GESTURE_MAP)
        self.audio_map     = load_audio_map()
        self.learning_mode = learning_mode
        self.dry_run       = dry_run

        # In learning mode, set when a gesture fires so an incoming audio
        # word can be correlated with it.
        self._pending_gesture: tuple[str, float] | None = None  # (action, timestamp)
        self._last_action_time: float = 0.0
        self._lock = threading.Lock()

        self.sender: Any | None = None
        if not dry_run:
            from spike_sender_skeleton import SpikeCommandSender

            self.sender = SpikeCommandSender()

    # ------------------------------------------------------------------
    # Callbacks — called from background threads
    # ------------------------------------------------------------------

    def on_gesture(self, gesture: str, confidence: float) -> None:
        if gesture in ("None", "none", "") or confidence < GESTURE_CONFIDENCE:
            return

        action = self.gesture_map.get(gesture)
        if action is None:
            print(f"[gesture] {gesture} ({confidence:.2f}) — not in gesture map")
            return
        action = normalize_action(action)

        with self._lock:
            if self.learning_mode:
                # Store so an incoming audio word can be paired with this action.
                self._pending_gesture = (action, time.monotonic())
                print(f"[gesture] {gesture} ({confidence:.2f}) → {action}  [learning: waiting for audio]")
            else:
                print(f"[gesture] {gesture} ({confidence:.2f}) → {action}")
            self._fire(action)

    def on_audio(self, text: str) -> None:
        text = text.strip().lower()
        if not text:
            return

        with self._lock:
            action = self._lookup_audio(text)
            if action:
                print(f"[audio ] '{text}' → {action}")
                self._fire(action)
            elif self.learning_mode:
                # Use only the first word so the learned key stays consistent
                # with how a user would say it in future sessions.
                keyword = text.split()[0]
                self._try_learn(keyword)
            else:
                print(f"[audio ] '{text}' — unknown  (use --learning-mode to teach me)")

    # ------------------------------------------------------------------
    # Internal helpers — called under self._lock
    # ------------------------------------------------------------------

    def _lookup_audio(self, text: str) -> str | None:
        """Return the action for text, checking full phrase then each word."""
        if text in self.audio_map:
            return self.audio_map[text]
        for word in text.split():
            if word in self.audio_map:
                return self.audio_map[word]
        return None

    def _fire(self, action: str) -> None:
        action = normalize_action(action)
        if action not in VALID_ACTIONS:
            print(f"[action] {action} — invalid robot command")
            return

        now = time.monotonic()
        if now - self._last_action_time < ACTION_COOLDOWN:
            return
        self._last_action_time = now
        print(f"[action] {action}")
        if self.sender:
            self.sender.send_command(action)

    def _try_learn(self, word: str) -> None:
        if self._pending_gesture is None:
            print(f"[learn ] '{word}' heard but no recent gesture to pair — show a gesture first")
            return

        action, ts = self._pending_gesture
        if time.monotonic() - ts > LEARN_WINDOW:
            self._pending_gesture = None
            print(f"[learn ] '{word}' heard but the gesture was too long ago — try again")
            return

        self.audio_map[word] = action
        save_audio_map(self.audio_map)
        self._pending_gesture = None
        print(f"[learn ] Learned: '{word}' → {action}  (saved to {AUDIO_MAP_PATH})")

    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.sender:
            self.sender.connect()

    def close(self) -> None:
        if self.sender:
            self.sender.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MyLittleRobotFriend — Pi orchestrator")
    p.add_argument(
        "--gesture-model",
        default="inference/checkpoints/gesture_recognizer.task",
        help="Path to gesture_recognizer.task (default: %(default)s)",
    )
    p.add_argument(
        "--vosk-model",
        default="vosk_stt/vosk-model-small-en-us-0.15",
        help="Path to Vosk model directory (default: %(default)s)",
    )
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="sounddevice input device index for mic (default: system default)",
    )
    p.add_argument(
        "--list-mic-devices",
        action="store_true",
        help="List available sounddevice audio devices and exit",
    )
    p.add_argument(
        "--learning-mode",
        action="store_true",
        help="Enable learning mode: pair unrecognized audio words with gestures",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Bluetooth connection; print commands only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_mic_devices:
        from vosk_stt.recognizer import print_input_devices

        print_input_devices()
        return

    from gesture_classification.infer_rpi import GestureRecognizer
    from vosk_stt.recognizer import AudioRecognizer

    print(
        f"Mode: {'LEARNING' if args.learning_mode else 'NORMAL'}  |  "
        f"Bluetooth: {'off (dry-run)' if args.dry_run else 'on'}"
    )

    bot = RobotOrchestrator(learning_mode=args.learning_mode, dry_run=args.dry_run)
    gesture_rec = None
    audio_rec = None
    try:
        if not args.dry_run:
            bot.connect()

        gesture_rec = GestureRecognizer(
            model_path=args.gesture_model,
            camera_index=args.camera,
            on_gesture=bot.on_gesture,
        )
        audio_rec = AudioRecognizer(
            model_dir=args.vosk_model,
            device_index=args.mic_device,
            on_text=bot.on_audio,
        )

        gesture_rec.start()
        audio_rec.start()

        print("Running — Ctrl-C to stop")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if gesture_rec:
            gesture_rec.stop()
        if audio_rec:
            audio_rec.stop()
        bot.close()


if __name__ == "__main__":
    main()
