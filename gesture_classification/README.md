# Raspberry Pi Inference Folder

Copy this folder to the Raspberry Pi after training. Then copy the trained model
from Colab into:

```text
inference/checkpoints/gesture_recognizer.task
```

Expected structure on the Raspberry Pi:

```text
inference/
  infer_rpi.py
  requirements-rpi.txt
  checkpoints/
    gesture_recognizer.task
```

Install dependencies and run:

```bash
pip install -r requirements-rpi.txt
python infer_rpi.py
```

Use terminal-only output with:

```bash
python infer_rpi.py --print-only
```
