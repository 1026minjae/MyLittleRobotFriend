"""Bluetooth communication with the LEGO SPIKE Hub."""

from __future__ import annotations

import queue
import threading
from typing import Callable


ALLOWED_COMMANDS = [
    "walk_forward",
    "walk_backward",
    "left_turn",
    "right_turn",
    "fist_bump",
    "powerful_dance",
    "moderate_dance",
    "handshake",
    "raise_right_arm",
    "shake_head",
    "see_arround",
    "idle",
]


class SpikeConnection:
    """Newline-delimited SPIKE Bluetooth sender and signal receiver."""

    def __init__(self, on_signal: Callable[[str], None] | None = None) -> None:
        self._on_signal = on_signal
        self._sock = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._signals: queue.Queue[str] = queue.Queue()

    def connect(self, mac_address: str | None = None) -> None:
        bluetooth = _load_bluetooth()
        if mac_address is None:
            mac_address = self._choose_device(bluetooth)

        print(f"[spike ] connecting to {mac_address}")
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((mac_address, 1))
        self._sock = sock
        print("[spike ] connected")

    def start_receiver(self) -> None:
        if self._sock is None:
            raise RuntimeError("SPIKE socket is not connected")
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True, name="spike-receiver")
        self._thread.start()
        print("[spike ] waiting for button signals")

    def get_signal(self, timeout: float | None = None) -> str | None:
        try:
            return self._signals.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_command(self, command: str) -> None:
        command = command.strip()
        if command not in ALLOWED_COMMANDS:
            raise ValueError(f"Invalid SPIKE command: {command}")
        if self._sock is None:
            raise RuntimeError("SPIKE socket is not connected")

        self._sock.send((command + "\n").encode("utf-8"))
        print(f"[spike ] sent: {command}")

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread:
            self._thread.join(timeout=1.0)

    def _receive_loop(self) -> None:
        buffer = ""
        while self._running:
            try:
                data = self._sock.recv(1024)
                if not data:
                    continue
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    signal = message.strip()
                    if signal:
                        print(f"[spike ] received: {signal}")
                        self._signals.put(signal)
                        if self._on_signal:
                            self._on_signal(signal)
            except OSError:
                break
            except Exception as exc:
                if self._running:
                    print(f"[spike ] receive error: {exc}")
                break

    def _choose_device(self, bluetooth) -> str:
        print("[spike ] searching for paired Bluetooth devices")
        devices = bluetooth.discover_devices(lookup_names=True)
        if not devices:
            raise RuntimeError("No Bluetooth devices found. Pair the Raspberry Pi with the SPIKE Hub first.")

        for idx, (addr, name) in enumerate(devices):
            print(f"{idx}: {name} ({addr})")

        while True:
            raw = input("Select SPIKE device number: ").strip()
            try:
                idx = int(raw)
                return devices[idx][0]
            except (ValueError, IndexError):
                print("Enter a valid device number.")


def _load_bluetooth():
    try:
        import bluetooth
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyBluez is required for SPIKE Bluetooth. Install the 'pybluez' package.") from exc
    return bluetooth
