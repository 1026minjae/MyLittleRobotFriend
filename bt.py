import bluetooth

class BT():
    def __init__(self):
        pass

    def find_devices(self):
        devices = []
        nearby_devices = bluetooth.discover_devices(lookup_names=True)
        for addr, name in nearby_devices:
            devices.append({'name' : name, 'addr' : addr})

        return devices
    
    def connect(self, addr):
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        try:
            sock.connect((addr, 1))
        except OSError:
            return None
        else:
            return sock