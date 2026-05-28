from bt import BT
import sys
import time
import threading


class SpikeCommandSender:
    def __init__(self):
        self.bt = BT()
        self.bt_devices = []
        self.sock = None
        self.running = False

        self.allowed_commands = [
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
            "idle"
        ]

    def find_devices(self):
        print("Searching for Bluetooth devices...\n")

        devices = self.bt.find_devices()

        if len(devices) == 0:
            print("No paired Bluetooth devices found.")
            print("Pair Raspberry Pi with LEGO SPIKE Hub first.")
            sys.exit(1)

        for i, device in enumerate(devices):
            name = device["name"]
            addr = device["addr"]

            self.bt_devices.append([name, addr])

            print(f"NUM: {i} | DEVICE NAME: {name} | MAC ADDRESS: {addr}")

        print()

    def select_device(self):
        print("Enter the number of the SPIKE Hub device to connect.")

        while True:
            try:
                device_id = int(input("NUM: "))
                target_device = self.bt_devices[device_id]
                return target_device

            except ValueError:
                print("Invalid input. Please enter a number.")

            except IndexError:
                print("There is no device with that number.")

    def connect(self):
        self.find_devices()
        target_device = self.select_device()

        name = target_device[0]
        mac_address = target_device[1]

        print(f"\nConnecting to {name} : {mac_address}\n")

        self.sock = self.bt.connect(mac_address)

        if self.sock is None:
            print("Bluetooth connection failed.")
            print("Check SPIKE Hub power, Bluetooth pairing, and receiver code.")
            sys.exit(1)

        print("Connected successfully.\n")

    def send_command(self, command):
        command = command.strip()

        if command not in self.allowed_commands:
            print(f"Invalid command: {command}")
            print("Available commands:")
            for cmd in self.allowed_commands:
                print(f" - {cmd}")
            return

        # SPIKE Hub uses readline(), so '\n' is important.
        message = command + "\n"

        try:
            self.sock.send(message.encode("utf-8"))
            print(f"Sent: {command}")

        except Exception as e:
            print("Failed to send command:", e)

    def start_receiver_thread(self):
        self.running = True

        receiver_thread = threading.Thread(target=self.receive_loop)
        receiver_thread.daemon = True
        receiver_thread.start()

        print("Receiver thread started.")
        print("Waiting for SPIKE Hub button signals...\n")

    def receive_loop(self):
        buffer = ""

        while self.running:
            try:
                data = self.sock.recv(1024)

                if not data:
                    continue

                buffer += data.decode("utf-8")

                # SPIKE Hub sends signals with '\n'
                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    signal = message.strip()

                    if signal:
                        self.handle_spike_signal(signal)

            except OSError:
                # Socket was closed
                break

            except Exception as e:
                if self.running:
                    print("Receive error:", e)
                break

    def handle_spike_signal(self, signal):
        print(f"\nReceived from SPIKE Hub: {signal}")

        if signal == "left_button":
            print("Action: SPIKE Hub left button was pressed.")

        elif signal == "right_button":
            print("Action: SPIKE Hub right button was pressed.")

        elif signal == "both_buttons":
            print("Action: Both SPIKE Hub buttons were pressed.")
            print("SPIKE Hub program may stop.")

        else:
            print(f"Unknown signal from SPIKE Hub: {signal}")

        print("Command: ", end="", flush=True)

    def run_interactive_mode(self):
        print("Interactive command mode started.")
        print("Type one of the following commands:")
        for cmd in self.allowed_commands:
            print(f" - {cmd}")

        print("\nType 'q' or 'quit' to exit.\n")

        while True:
            command = input("Command: ").strip()

            if command in ["q", "quit", "exit"]:
                print("Exit command sender.")
                break

            self.send_command(command)

            # Small delay to avoid sending commands too aggressively
            time.sleep(0.1)

    def close(self):
        self.running = False

        if self.sock is not None:
            try:
                self.sock.close()
                print("\nBluetooth socket closed.")
            except Exception:
                pass


if __name__ == "__main__":
    sender = SpikeCommandSender()

    try:
        sender.connect()

        # Start receiving button signals from SPIKE Hub
        sender.start_receiver_thread()

        # Continue sending commands to SPIKE Hub
        sender.run_interactive_mode()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        sender.close()