from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    DEFAULT_COMMAND_DICT,
    action_for_class,
    choose_class,
    class_names_from_maps,
    extract_last_word,
    fuse_scores,
    stt_vector,
)


def test_extract_last_word_uses_command_word():
    assert extract_last_word("Hey robot, dance") == "dance"
    assert extract_last_word("") == "none"


def test_unknown_stt_maps_to_none_vector():
    classes = class_names_from_maps(DEFAULT_COMMAND_DICT)
    scores = stt_vector("unknownword", DEFAULT_COMMAND_DICT, classes)
    assert scores["none"] == 1.0


def test_weighted_fusion_can_choose_gesture():
    fused = fuse_scores(
        audio_scores={"dance": 1.0, "fist_bump": 0.0},
        gesture_scores={"dance": 0.0, "fist_bump": 1.0},
        audio_weight=0.25,
        gesture_weight=0.75,
    )
    assert choose_class(fused) == "fist_bump"


def test_none_action_is_shake_head():
    assert action_for_class("none") == "shake_head"


def test_default_dict_keeps_primary_classes():
    assert DEFAULT_COMMAND_DICT["dance"] == "dance"
    assert DEFAULT_COMMAND_DICT["fist_bump"] == "fist_bump"
    assert DEFAULT_COMMAND_DICT["turn_around"] == "turn_around"
    assert DEFAULT_COMMAND_DICT["none"] == "none"
