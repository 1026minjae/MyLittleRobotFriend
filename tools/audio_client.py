#!/usr/bin/env python3
"""
Audio Streaming Client for Raspberry Pi
Captures audio from microphone and streams to server over network
"""

import socket
import pyaudio
import wave
import time
import sys
import argparse

# Audio Configuration
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 2  # Stereo (adjust to 1 for mono)
RATE = 44100  # Sample rate (Hz)

# Network Configuration
SERVER_HOST = '192.168.1.100'  # Change to your laptop's IP
SERVER_PORT = 9999

def get_audio_device_index(p, device_name_keyword="seeed"):
    """
    Find the audio device index by searching for keyword in device name
    """
    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']} - Channels: {info['maxInputChannels']}")
        
        if device_name_keyword.lower() in info['name'].lower():
            if info['maxInputChannels'] > 0:
                print(f"\n✓ Found matching device: {info['name']} (index {i})")
                return i
    
    print(f"\n⚠ No device found with keyword '{device_name_keyword}'")
    return None

def stream_audio(server_host, server_port, device_index=None, duration=None):
    """
    Stream audio from microphone to server
    
    Args:
        server_host: Server IP address
        server_port: Server port number
        device_index: Audio device index (None for default)
        duration: Recording duration in seconds (None for continuous)
    """
    p = pyaudio.PyAudio()
    
    # If no device specified, try to find Seeed device
    if device_index is None:
        device_index = get_audio_device_index(p)
        if device_index is None:
            print("\nUsing default audio device")
    
    # Connect to server
    print(f"\nConnecting to server at {server_host}:{server_port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client_socket.connect((server_host, server_port))
        print("✓ Connected to server")
        
        # Send audio configuration to server
        config = f"{CHANNELS},{RATE},{FORMAT}".encode()
        client_socket.sendall(config + b'\n')
        
        # Open audio stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print(f"\n🎤 Recording and streaming audio...")
        print(f"   Sample rate: {RATE} Hz")
        print(f"   Channels: {CHANNELS}")
        print(f"   Format: 16-bit PCM")
        if duration:
            print(f"   Duration: {duration} seconds")
        else:
            print(f"   Duration: Continuous (Press Ctrl+C to stop)")
        
        # Calculate number of chunks if duration specified
        chunks_to_record = None
        if duration:
            chunks_to_record = int(RATE / CHUNK * duration)
        
        chunk_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Read audio chunk
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Send chunk size followed by data
                chunk_size = len(data).to_bytes(4, byteorder='big')
                client_socket.sendall(chunk_size + data)
                
                chunk_count += 1
                
                # Print progress every second
                if chunk_count % (RATE // CHUNK) == 0:
                    elapsed = time.time() - start_time
                    print(f"   Streaming... {elapsed:.1f}s elapsed")
                
                # Stop if duration reached
                if chunks_to_record and chunk_count >= chunks_to_record:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⏹ Stopping stream...")
        
        # Send end signal (chunk size of 0)
        client_socket.sendall((0).to_bytes(4, byteorder='big'))
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Streaming completed")
        print(f"   Total time: {elapsed:.1f} seconds")
        print(f"   Total chunks: {chunk_count}")
        
    except ConnectionRefusedError:
        print(f"✗ Error: Could not connect to server at {server_host}:{server_port}")
        print("  Make sure the server is running and the IP address is correct")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        client_socket.close()
        p.terminate()

def list_audio_devices():
    """List all available audio input devices"""
    p = pyaudio.PyAudio()
    print("\n=== Available Audio Input Devices ===\n")
    
    found_devices = False
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            found_devices = True
            print(f"Device {i}:")
            print(f"  Name: {info['name']}")
            print(f"  Channels: {info['maxInputChannels']}")
            print(f"  Sample Rate: {int(info['defaultSampleRate'])} Hz")
            print()
    
    if not found_devices:
        print("No input devices found!")
    
    p.terminate()

def main():
    parser = argparse.ArgumentParser(
        description='Stream audio from Raspberry Pi to server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream to server (auto-detect Seeed device)
  python3 audio_client.py --host 192.168.1.100
  
  # Stream for 30 seconds
  python3 audio_client.py --host 192.168.1.100 --duration 30
  
  # Use specific device
  python3 audio_client.py --host 192.168.1.100 --device 1
  
  # List available devices
  python3 audio_client.py --list
        """
    )
    
    parser.add_argument('--host', type=str, default=SERVER_HOST,
                        help=f'Server IP address (default: {SERVER_HOST})')
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                        help=f'Server port (default: {SERVER_PORT})')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index (default: auto-detect Seeed)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Recording duration in seconds (default: continuous)')
    parser.add_argument('--list', action='store_true',
                        help='List available audio devices and exit')
    
    args = parser.parse_args()
    
    if args.list:
        list_audio_devices()
        return
    
    print("=" * 50)
    print("Audio Streaming Client - Raspberry Pi")
    print("=" * 50)
    
    stream_audio(args.host, args.port, args.device, args.duration)

if __name__ == "__main__":
    main()
