# Vosk Speech-to-Text

Reusable STT code lives in `recognizer.py`. Use `test_vosk.py` to verify the
microphone and Vosk model before running the full robot orchestrator.

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv unzip wget portaudio19-dev
pip install -r vosk_stt/requirements.txt

wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O vosk_stt/vosk-model-small-en-us-0.15.zip
unzip vosk_stt/vosk-model-small-en-us-0.15.zip -d vosk_stt/
python3 vosk_stt/test_vosk.py
```

List microphone devices with:

```bash
python3 vosk_stt/test_vosk.py --list-devices
```
