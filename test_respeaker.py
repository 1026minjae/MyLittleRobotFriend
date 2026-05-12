import subprocess
from pathlib import Path


DURATION = 10
OUTPUT_FILE = Path("respeaker_test.wav")

# arecord -l 결과에 따라 바꿔야 할 수 있음
# 예: card 1, device 0이면 "plughw:1,0"
AUDIO_DEVICE = "plughw:1,0"


def main():
    print("[INFO] ReSpeaker recording test started.")
    print(f"[INFO] Duration: {DURATION} seconds")
    print(f"[INFO] Output file: {OUTPUT_FILE}")
    print(f"[INFO] Audio device: {AUDIO_DEVICE}")

    cmd = [
        "arecord",
        "-D", AUDIO_DEVICE,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "2",
        "-d", str(DURATION),
        str(OUTPUT_FILE),
    ]

    print("[INFO] Running command:")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode == 0 and OUTPUT_FILE.exists():
        print("[OK] Recording finished successfully.")
        print(f"[OK] Saved as: {OUTPUT_FILE}")
    else:
        print("[ERROR] Recording failed.")
        print("Check your ReSpeaker device with:")
        print("  arecord -l")


if __name__ == "__main__":
    main()