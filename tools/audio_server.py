#!/usr/bin/env python3
"""
Audio Streaming Server for Laptop
Receives audio stream from Raspberry Pi and saves to WAV file
"""

import socket
import wave
import time
import sys
import argparse
from datetime import datetime
import os

# Network Configuration
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 9999

def receive_audio(port, output_dir='recordings'):
    """
    Receive audio stream from client and save to WAV file
    
    Args:
        port: Port number to listen on
        output_dir: Directory to save recordings
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((SERVER_HOST, port))
        server_socket.listen(1)
        
        print("=" * 50)
        print("Audio Streaming Server")
        print("=" * 50)
        print(f"\n🎧 Server listening on port {port}...")
        print(f"   Recordings will be saved to: {os.path.abspath(output_dir)}/")
        print(f"   Waiting for client connection...")
        
        while True:
            # Accept client connection
            client_socket, client_address = server_socket.accept()
            print(f"\n✓ Client connected from {client_address[0]}:{client_address[1]}")
            
            try:
                # Receive audio configuration
                config_data = b''
                while b'\n' not in config_data:
                    config_data += client_socket.recv(1024)
                
                config = config_data.decode().strip().split(',')
                channels = int(config[0])
                rate = int(config[1])
                sample_width = 2  # 16-bit = 2 bytes
                
                print(f"\n📊 Audio Configuration:")
                print(f"   Channels: {channels}")
                print(f"   Sample rate: {rate} Hz")
                print(f"   Sample width: {sample_width} bytes (16-bit)")
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.wav"
                filepath = os.path.join(output_dir, filename)
                
                print(f"\n💾 Saving to: {filename}")
                
                # Open WAV file
                wf = wave.open(filepath, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                
                print(f"\n🎤 Recording audio stream...")
                
                chunk_count = 0
                total_bytes = 0
                start_time = time.time()
                
                try:
                    while True:
                        # Receive chunk size (4 bytes, big-endian)
                        size_data = b''
                        while len(size_data) < 4:
                            packet = client_socket.recv(4 - len(size_data))
                            if not packet:
                                break
                            size_data += packet
                        
                        if len(size_data) < 4:
                            break
                        
                        chunk_size = int.from_bytes(size_data, byteorder='big')
                        
                        # Check for end signal
                        if chunk_size == 0:
                            print("\n   Received end signal from client")
                            break
                        
                        # Receive audio data
                        audio_data = b''
                        while len(audio_data) < chunk_size:
                            packet = client_socket.recv(chunk_size - len(audio_data))
                            if not packet:
                                break
                            audio_data += packet
                        
                        if len(audio_data) < chunk_size:
                            break
                        
                        # Write to WAV file
                        wf.writeframes(audio_data)
                        
                        chunk_count += 1
                        total_bytes += len(audio_data)
                        
                        # Print progress every second
                        if chunk_count % 43 == 0:  # Approximately 1 second at 44100 Hz
                            elapsed = time.time() - start_time
                            size_mb = total_bytes / (1024 * 1024)
                            print(f"   Recording... {elapsed:.1f}s | {size_mb:.2f} MB")
                
                except KeyboardInterrupt:
                    print("\n\n⏹ Stopping recording...")
                
                # Close WAV file
                wf.close()
                
                elapsed = time.time() - start_time
                duration_seconds = chunk_count * 1024 / rate  # Approximate duration
                size_mb = total_bytes / (1024 * 1024)
                
                print(f"\n✓ Recording saved successfully!")
                print(f"\n📈 Recording Statistics:")
                print(f"   File: {filepath}")
                print(f"   Duration: {duration_seconds:.1f} seconds")
                print(f"   Size: {size_mb:.2f} MB")
                print(f"   Chunks received: {chunk_count}")
                print(f"   Transfer time: {elapsed:.1f} seconds")
                
                if elapsed > 0:
                    bitrate = (total_bytes * 8) / elapsed / 1000000  # Mbps
                    print(f"   Average bitrate: {bitrate:.2f} Mbps")
                
            except Exception as e:
                print(f"\n✗ Error during recording: {e}")
            finally:
                client_socket.close()
                print(f"\n⏹ Client disconnected")
                print(f"\n🎧 Waiting for next connection...")
    
    except KeyboardInterrupt:
        print("\n\n⏹ Server shutting down...")
    except Exception as e:
        print(f"\n✗ Server error: {e}")
    finally:
        server_socket.close()
        print("\n✓ Server closed")

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Create a dummy socket to find local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "Unable to determine"

def main():
    parser = argparse.ArgumentParser(
        description='Receive audio stream from Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port
  python3 audio_server.py
  
  # Use custom port
  python3 audio_server.py --port 8888
  
  # Save to custom directory
  python3 audio_server.py --output my_recordings
        """
    )
    
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                        help=f'Port to listen on (default: {SERVER_PORT})')
    parser.add_argument('--output', type=str, default='recordings',
                        help='Output directory for recordings (default: recordings)')
    
    args = parser.parse_args()
    
    # Display local IP for client configuration
    local_ip = get_local_ip()
    print(f"\n💡 Your laptop's IP address: {local_ip}")
    print(f"   Use this IP in the client script (--host {local_ip})")
    
    receive_audio(args.port, args.output)

if __name__ == "__main__":
    main()
