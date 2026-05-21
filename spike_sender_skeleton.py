from bt import BT
import sys
import time


class SpikeCommandSender:
    def __init__(self):
        self.bt = BT()
        self.bt_devices = []
        self.sock = None

        self.allowed_commands = [
            "walk",
            "left_turn",
            "right_turn",
            "fist_bump",
            "powerful_dance",
            "moderate_dance",
            "handshake",
            "raise_right_arm",
            "shake_head",
            "see_arround"
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
        if self.sock is not None:
            try:
                self.sock.close()
                print("Bluetooth socket closed.")
            except Exception:
                pass


if __name__ == "__main__":
    sender = SpikeCommandSender()

    try:
        sender.connect()
        sender.run_interactive_mode()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        sender.close()