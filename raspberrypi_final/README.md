# Raspberry Pi Final Integration

This folder is the final Raspberry Pi-side runtime for MyLittleRobotFriend. It glues together:

- LEGO SPIKE Hub Bluetooth signals.
- Real-time MediaPipe gesture inference.
- Real-time Vosk speech-to-text.
- Weighted STT plus gesture decision fusion.
- Command learning through a persisted dictionary.

The rest of the repository is left untouched. This folder is intended to be copied to the Raspberry Pi as one self-contained runtime package.

## Folder Structure

```text
raspberrypi_final/
  main.py
  core.py
  gesture_inference.py
  stt.py
  spike_comm.py
  command_dict.json
  requirements.txt
  README.md
  gesture_classification/
    checkpoints/
      gesture_recognizer.task
  vosk_stt/
    vosk-model-small-en-us-0.15/   # downloaded separately
  tests/
    test_core.py
```

`main.py` is the orchestrator. It starts every runtime worker, waits for SPIKE button signals, opens 5-second decision windows, and either sends an action or learns a command.

`core.py` contains pure decision logic: default dictionaries, last-word extraction, vector fusion, class selection, and gesture-class-to-robot-action mapping.

`gesture_inference.py` runs MediaPipe continuously. It does not wait for a button press to start inference. It stores timestamped frame predictions in a rolling buffer.

`stt.py` runs Vosk continuously. It stores timestamped final recognized phrases in a rolling buffer and exposes the last recognized word inside a time window.

`spike_comm.py` handles newline-delimited Bluetooth communication with the SPIKE Hub.

`command_dict.json` maps spoken last words to gesture classes.

## End-to-End Flow

At startup:

1. The Pi loads `command_dict.json`.
2. The camera worker starts real-time gesture inference.
3. The microphone worker starts real-time Vosk STT.
4. The Pi connects to the SPIKE Hub over Bluetooth.
5. The Pi waits for newline-delimited SPIKE button messages.

The SPIKE Hub should send:

```text
left_button
right_button
center_button
```

`left_button` means interaction mode.

`right_button` means command-adding mode.

`center_button` stops the Pi program.

## Interaction Mode

When SPIKE sends `left_button`:

1. The Pi opens a 5-second inference window.
2. Gesture and STT are already running in real time, so they keep producing predictions during that window.
3. The Pi reads gesture frame predictions from the window and aggregates them into a vector over gesture classes.
4. The Pi reads the last STT word from the window.
5. The last word is mapped through `command_dict.json` into a one-hot gesture-class vector.
6. The STT vector and gesture vector are fused.
7. The winning gesture class is mapped to a robot action.
8. The Pi sends that action back to SPIKE with a trailing newline.

Example:

```text
SPIKE -> Pi: left_button
STT hears: "hey robot dance"
last word: dance
command_dict["dance"] -> dance
gesture window votes mostly dance
winner: dance
action: powerful_dance
Pi -> SPIKE: powerful_dance
```

## Command-Adding Mode

When SPIKE sends `right_button`:

1. The Pi opens the same 5-second real-time inference window.
2. The Pi extracts the last STT word.
3. The Pi chooses the top gesture class from the gesture window.
4. The Pi saves:

```json
"last_word": "gesture_class"
```

Example:

```text
STT hears: "hey robot groove"
last word: groove
gesture winner: dance
saved mapping: "groove": "dance"
```

No action is sent to SPIKE in command-adding mode.

If a word already exists, the new mapping overwrites the old one.

## Real-Time Inference Window

The button press does not mean "record for 5 seconds, then run inference."

Instead, inference is always active:

- The gesture worker classifies camera frames continuously.
- The STT worker transcribes speech continuously.
- Each worker stores timestamped outputs in a rolling buffer.
- A button press marks the start of a decision window.
- After 5 seconds, `main.py` asks both buffers what happened during that time range.

This keeps the behavior closer to the live robot interaction: the Pi is always listening and watching, but decisions are only made after SPIKE asks for one.

The default window is:

```bash
--capture-seconds 5.0
```

## Gesture Aggregation

Each camera frame produces gesture scores from MediaPipe. The app aggregates all frame-level scores inside the 5-second window.

Predictions are treated as `none` when:

- MediaPipe returns no gesture.
- The class is empty.
- The top confidence is below `--gesture-threshold`.

Default:

```bash
--gesture-threshold 0.7
```

If there are no usable gesture frames in the window, the gesture vector becomes:

```text
none = 1.0
```

## STT Handling

Vosk may produce phrases like:

```text
hey robot dance
```

The app uses only the last word:

```text
dance
```

This makes command phrases flexible. Users can say "robot please dance" or "hey robot dance"; both become `dance`.

Missing, empty, or unknown STT becomes `none`.

## Weighted Vector Fusion

Both STT and gesture are converted into the same class space: raw gesture classes.

The default primary classes are:

```text
dance
fist_bump
turn_around
none
```

STT becomes a one-hot vector. If the last word is `dance`, then:

```text
dance = 1.0
everything else = 0.0
```

Gesture becomes a probability-like vector from frame votes. Example:

```text
dance = 0.80
fist_bump = 0.10
turn_around = 0.00
none = 0.10
```

Fusion uses:

```text
score[class] = audio_weight * stt_score[class] + gesture_weight * gesture_score[class]
```

Defaults:

```bash
--audio-weight 0.5
--gesture-weight 0.5
```

The app takes the argmax class and converts it to a SPIKE action.

## Class and Action Maps

The command dictionary maps spoken last words to gesture classes, not directly to robot actions.

Default command entries include the same classes as the gesture map:

```json
{
  "dance": "dance",
  "fist_bump": "fist_bump",
  "turn_around": "turn_around",
  "none": "none"
}
```

There are also convenient aliases:

```json
{
  "bump": "fist_bump",
  "turn": "turn_around",
  "around": "turn_around"
}
```

The gesture-to-action map in `core.py` contains:

```text
dance       -> powerful_dance
fist_bump   -> fist_bump
turn_around -> left_turn
none        -> shake_head
```

So if both STT and gesture are missing or unusable, the selected class is `none`, and the robot action is:

```text
shake_head
```

Older labels such as `walk_gesture`, `left_gesture`, and `dance_gesture` are kept for backward compatibility.

## Setup on Raspberry Pi

Install system dependencies first:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev libbluetooth-dev
```

Install Python dependencies:

```bash
cd raspberrypi_final
pip install -r requirements.txt
```

Download the Vosk model:

```bash
cd raspberrypi_final
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d vosk_stt/
```

Expected model path:

```text
raspberrypi_final/vosk_stt/vosk-model-small-en-us-0.15/
```

The gesture model should already be at:

```text
raspberrypi_final/gesture_classification/checkpoints/gesture_recognizer.task
```

## Running

List microphone devices:

```bash
python3 main.py --list-mic-devices
```

Run with interactive Bluetooth device selection:

```bash
python3 main.py
```

Run with a known SPIKE MAC address:

```bash
python3 main.py --spike-mac AA:BB:CC:DD:EE:FF
```

Run with a specific mic device:

```bash
python3 main.py --mic-device 1
```

Tune the fusion weights:

```bash
python3 main.py --audio-weight 0.6 --gesture-weight 0.4
```

Local dry run:

```bash
python3 main.py --dry-run
```

In dry run, type:

```text
left_button
right_button
center_button
```

## SPIKE Protocol

SPIKE to Pi:

```text
left_button\n
right_button\n
center_button\n
```

Pi to SPIKE:

```text
powerful_dance\n
fist_bump\n
left_turn\n
shake_head\n
```

The newline matters because the SPIKE receiver code usually uses `readline()`.

## Troubleshooting

If Vosk model is missing:

```text
Vosk model not found
```

Download and unzip `vosk-model-small-en-us-0.15` into `vosk_stt/`.

If the mic does not work:

```bash
python3 main.py --list-mic-devices
python3 main.py --mic-device <device_number>
```

If the camera fails:

```bash
python3 main.py --camera 0
python3 main.py --camera 1
```

If Bluetooth fails:

- Pair the Raspberry Pi and SPIKE Hub first.
- Confirm the hub is powered on.
- Try `--spike-mac` to skip interactive selection.
- Confirm `pybluez` installed correctly.

If the robot shakes its head often:

- STT may be empty or unknown.
- Gesture confidence may be below threshold.
- Lower `--gesture-threshold`.
- Add the spoken command in command-adding mode with `right_button`.

If unexpected SPIKE signals appear:

- Check that the SPIKE code sends exactly `left_button`, `right_button`, or `center_button`.
- Each signal should end with `\n`.
