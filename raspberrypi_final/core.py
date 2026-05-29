"""Pure decision logic for the Raspberry Pi orchestrator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


PRIMARY_GESTURE_CLASSES = ["dance", "fist_bump", "turn_around", "none"]

GESTURE_TO_ACTION = {
    "dance": "powerful_dance",
    "fist_bump": "fist_bump",
    "turn_around": "left_turn",
    "none": "shake_head",
    "walk_gesture": "walk_forward",
    "walk_forward": "walk_forward",
    "walk_backward": "walk_backward",
    "left_gesture": "left_turn",
    "right_gesture": "right_turn",
    "fist_gesture": "fist_bump",
    "dance_gesture": "powerful_dance",
    "wave_gesture": "raise_right_arm",
    "shake_gesture": "shake_head",
    "look_gesture": "see_arround",
}

DEFAULT_COMMAND_DICT = {
    "dance": "dance",
    "fist_bump": "fist_bump",
    "turn_around": "turn_around",
    "none": "none",
    "bump": "fist_bump",
    "fist": "fist_bump",
    "turn": "turn_around",
    "around": "turn_around",
    "walk": "walk_gesture",
    "forward": "walk_forward",
    "back": "walk_backward",
    "backward": "walk_backward",
    "left": "left_gesture",
    "right": "right_gesture",
    "wave": "wave_gesture",
    "shake": "shake_gesture",
    "look": "look_gesture",
}


def normalize_class_name(name: str | None) -> str:
    name = (name or "").strip().lower()
    return name if name else "none"


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_last_word(text: str | None) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "none"
    words = re.findall(r"[a-z0-9_']+", normalized)
    return words[-1] if words else "none"


def load_command_dict(path: Path) -> dict[str, str]:
    command_dict = dict(DEFAULT_COMMAND_DICT)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a JSON object")
        command_dict.update({str(key).lower(): normalize_class_name(str(value)) for key, value in loaded.items()})
    command_dict.setdefault("none", "none")
    return command_dict


def save_command_dict(command_dict: dict[str, str], path: Path) -> None:
    path.write_text(json.dumps(command_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def class_names_from_maps(command_dict: dict[str, str]) -> list[str]:
    names = set(PRIMARY_GESTURE_CLASSES)
    names.update(normalize_class_name(name) for name in command_dict.values())
    names.update(GESTURE_TO_ACTION)
    return sorted(names)


def stt_vector(last_word: str, command_dict: dict[str, str], class_names: Iterable[str]) -> dict[str, float]:
    scores = {normalize_class_name(name): 0.0 for name in class_names}
    gesture_class = command_dict.get((last_word or "none").lower(), "none")
    scores[normalize_class_name(gesture_class)] = 1.0
    return scores


def fuse_scores(
    audio_scores: dict[str, float],
    gesture_scores: dict[str, float],
    audio_weight: float,
    gesture_weight: float,
) -> dict[str, float]:
    classes = set(audio_scores) | set(gesture_scores) | {"none"}
    return {
        name: audio_weight * audio_scores.get(name, 0.0) + gesture_weight * gesture_scores.get(name, 0.0)
        for name in classes
    }


def choose_class(scores: dict[str, float]) -> str:
    if not scores:
        return "none"
    return max(sorted(scores), key=lambda name: scores[name])


def action_for_class(gesture_class: str) -> str:
    return GESTURE_TO_ACTION.get(normalize_class_name(gesture_class), "shake_head")
