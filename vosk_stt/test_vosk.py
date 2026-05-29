import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

MODEL_PATH = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

print("Speak now. Press Ctrl+C to stop.")

try:
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback
    ):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                print("TEXT:", result.get("text", ""))
            else:
                partial = json.loads(recognizer.PartialResult())
                print("PARTIAL:", partial.get("partial", ""), end="\r")

except KeyboardInterrupt:
    print("\nStopped.")