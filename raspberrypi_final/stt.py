"""Real-time Vosk speech recognition with a rolling text buffer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import queue
import re
import threading
import time
from pathlib import Path

import sounddevice as sd
import vosk


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "vosk_stt" / "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000


@dataclass(frozen=True)
class SpeechSample:
    timestamp: float
    text: str
    last_word: str


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_last_word(text: str | None) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "none"
    words = re.findall(r"[a-z0-9_']+", normalized)
    return words[-1] if words else "none"


class RealtimeSpeechRecognizer:
    """Continuously runs Vosk STT and stores final recognized phrases."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device_index: int | None = None,
        buffer_seconds: float = 20.0,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        emit_partials: bool = False,
    ) -> None:
        model_path = Path(model_dir).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'. "
                "Download vosk-model-small-en-us-0.15 and place it in vosk_stt/, "
                "or pass --vosk-model."
            )

        vosk.SetLogLevel(-1)
        self._model = vosk.Model(str(model_path))
        self._device_index = device_index
        self._buffer_seconds = buffer_seconds
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._emit_partials = emit_partials

        self._samples: deque[SpeechSample] = deque()
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="speech-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def samples_between(self, start: float, end: float) -> list[SpeechSample]:
        with self._lock:
            return [sample for sample in self._samples if start <= sample.timestamp <= end]

    def last_word_between(self, start: float, end: float) -> str:
        samples = self.samples_between(start, end)
        if not samples:
            return "none"
        return samples[-1].last_word or "none"

    def _append_sample(self, text: str, timestamp: float) -> None:
        normalized = normalize_text(text)
        if not normalized:
            return
        sample = SpeechSample(timestamp=timestamp, text=normalized, last_word=extract_last_word(normalized))
        cutoff = timestamp - self._buffer_seconds
        with self._lock:
            self._samples.append(sample)
            while self._samples and self._samples[0].timestamp < cutoff:
                self._samples.popleft()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio ] {status}")
        self._audio_queue.put(bytes(indata))

    def _run(self) -> None:
        try:
            recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
            with sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                dtype="int16",
                channels=1,
                device=self._device_index,
                callback=self._callback,
            ):
                print("[audio ] realtime worker active")
                while not self._stop_event.is_set():
                    try:
                        data = self._audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        self._append_sample(result.get("text", ""), time.monotonic())
                    elif self._emit_partials:
                        partial = json.loads(recognizer.PartialResult())
                        text = normalize_text(partial.get("partial", ""))
                        if text:
                            print(f"[audio ] partial: {text}", end="\r")
        except BaseException as exc:
            self._error = exc
            print(f"[audio ] worker stopped: {exc}")


def print_input_devices() -> None:
    print(sd.query_devices())
