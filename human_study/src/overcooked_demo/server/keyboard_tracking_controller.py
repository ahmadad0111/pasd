import socket
import time

class TrackingController:
    def __init__(self, host="localhost", keyboard_port=5002):
        self.host = host
        self.keyboard_port = keyboard_port

    def send_command(self, port, message):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.host, port))
                s.sendall(message.encode())
            except ConnectionRefusedError:
                print(f"[ERROR] Could not connect to {self.host}:{port}")

    def start_tracking(self, session_id):
        print("Starting keyboard tracking...")
        self.send_command(self.keyboard_port, f"START {session_id}\n")

    def stop_tracking(self):
        print("Stopping keyboard tracking...")
        self.send_command(self.keyboard_port, "STOP\n")
