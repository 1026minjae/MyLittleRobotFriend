"""Reusable offline speech recognizer using Vosk."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Callable

import sounddevice as sd
import vosk

DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent / "vosk-model-small-en-us-0.15")
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000


class AudioRecognizer:
    """Listen on the microphone and call on_text(text) for each utterance."""

    def __init__(
        self,
        model_dir: str,
        on_text: Callable[[str], None],
        device_index: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        emit_partials: bool = False,
    ) -> None:
        model_path = Path(model_dir).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'. "
                "Download vosk-model-small-en-us-0.15 and pass its path with --vosk-model."
            )

        vosk.SetLogLevel(-1)
        self._model = vosk.Model(str(model_path))
        self._on_text = on_text
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._emit_partials = emit_partials

        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-rec")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio ] {status}")
        self._audio_queue.put(bytes(indata))

    def _run(self) -> None:
        recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)

        try:
            with sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                dtype="int16",
                channels=1,
                device=self._device_index,
                callback=self._callback,
            ):
                print("[audio ] Microphone active")
                while not self._stop_event.is_set():
                    try:
                        data = self._audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            self._on_text(text)
                    elif self._emit_partials:
                        partial = json.loads(recognizer.PartialResult())
                        text = partial.get("partial", "").strip()
                        if text:
                            print(f"[audio ] partial: {text}", end="\r")
        except Exception as exc:
            print(f"[audio ] recognizer stopped: {exc}")


def print_input_devices() -> None:
    """Print sounddevice input devices for choosing --mic-device."""
    print(sd.query_devices())
