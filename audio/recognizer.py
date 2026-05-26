"""Offline speech recognizer using Vosk.

Download a model from https://alphacephei.com/vosk/models and unzip it to
audio/vosk-model/ (or pass --vosk-model <path> when running main.py).

Recommended model for Raspberry Pi: vosk-model-small-en-us-0.15 (~40 MB).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable

import pyaudio
import vosk

RATE = 16000
CHUNK = 4000   # ~250 ms of audio per read; keeps latency low on Pi


class AudioRecognizer:
    """Listens on the microphone and calls on_text(text) for each utterance.

    Runs in a background daemon thread.  Call start() / stop() to control it.
    """

    def __init__(
        self,
        model_dir: str,
        on_text: Callable[[str], None],
        device_index: int | None = None,
    ) -> None:
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'.\n"
                "Download vosk-model-small-en-us-0.15 from "
                "https://alphacephei.com/vosk/models and unzip it there."
            )
        # Vosk prints a lot of init noise; silence it
        vosk.SetLogLevel(-1)
        self._model = vosk.Model(str(model_path))
        self._on_text = on_text
        self._device_index = device_index
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

    def _run(self) -> None:
        rec = vosk.KaldiRecognizer(self._model, RATE)
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=CHUNK,
        )
        stream.start_stream()
        print("[audio ] Microphone active")

        try:
            while not self._stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        self._on_text(text)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
