# Raspberry Pi Inference Folder

Copy this folder to the Raspberry Pi after training. Then copy the trained model
from Colab into:

```text
inference/exported_model/gesture_recognizer.task
```

Expected structure on the Raspberry Pi:

```text
inference/
  infer_rpi.py
  requirements-rpi.txt
  exported_model/
    gesture_recognizer.task
```

Install dependencies and run:

```bash
pip install -r requirements-rpi.txt
python infer_rpi.py --model exported_model/gesture_recognizer.task
```

Use terminal-only output with:

```bash
python infer_rpi.py --model exported_model/gesture_recognizer.task --print-only
```
