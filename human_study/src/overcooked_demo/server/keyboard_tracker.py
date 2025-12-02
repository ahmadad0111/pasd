import socket
import psycopg2
import keyboard
from datetime import datetime
from threading import Thread
import json
from time import time

class KeyboardTracker:
    def __init__(self,
                 config_file="db.json",
                 server_host="0.0.0.0",
                 server_port=5002,
                 is_bulk=False):
        # Load database credentials from JSON file
        with open(config_file, "r") as file:
            self.config = json.load(file)

        self.server_host = server_host
        self.server_port = server_port
        self.current_hash_key = None
        self.running = False
        self.is_bulk = is_bulk
        self.bulk_events = []
        self.insert_query = """
                            INSERT INTO keyboard_events (hash_key, timestamp, key, event_type)
                            VALUES (%s, %s, %s, %s)
                            """
        self.insert_query_bulk = """
                                INSERT INTO keyboard_events (hash_key, timestamp, key, event_type)
                                VALUES {}
                                """

        # Establish database connection
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"Error connecting to the database: {e}")
        print("Keyboard tracker initialized. Is bulk inserting keyboard events into database? ",self.is_bulk)

    # Connect to DB and insert keyboard event
    def insert_keyboard_event(self,
                              key,
                              event_type):
        if not self.is_bulk:
            if not self.current_hash_key:
                return
            try:
                self.cursor.execute(self.insert_query, (self.current_hash_key, time(), key, event_type),
                )
                self.conn.commit()
            except Exception as e:
                print(f"Database Error: {e}")
        else:
            self.bulk_events.append(tuple([self.current_hash_key, time(), key, event_type]))

    # Keyboard listener callback
    def on_key(self, event):
        if self.running:
            self.insert_keyboard_event(event.name, event.event_type)

    def start_listener(self):
        keyboard.hook(self.on_key)
        keyboard.wait()

    # Socket listener for receiving commands
    def socket_listener(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.server_host, self.server_port))
        server.listen(1)
        print(f"Keyboard Tracker listening on {self.server_host}:{self.server_port}")

        while True:
            conn, addr = server.accept()
            with conn:
                print(f"Connection from {addr}")
                while True:
                    data = conn.recv(1024).decode().strip()
                    if not data:
                        break

                    if data.startswith("START"):
                        self.current_hash_key = data.split()[1]
                        self.running = True
                        try:
                            self.cursor.execute("INSERT INTO session (hash_key) VALUES (%s) ON CONFLICT DO NOTHING", (self.current_hash_key,))
                            self.conn.commit()
                        except Exception as e:
                            print(f"DB Error: {e}")
                        print(f"Recording started with key: {self.current_hash_key}")

                    elif data == "STOP":
                        self.running = False
                        self.current_hash_key = None
                        if self.is_bulk and len(self.bulk_events)>0:
                            try:
                                records_list_template = ','.join(['%s'] * len(self.bulk_events))
                                self.cursor.execute(self.insert_query_bulk.format(records_list_template),
                                                    self.bulk_events)
                                self.conn.commit()
                            except:
                                print("!!! Keyboard events failed to be inserted into db !!!")
                                print("Recorded keyboard events: ",self.bulk_events)
                            print("Keyboard events recorded and finished insertion into db.")
                        print("Recording stopped.")

    # Start both the socket listener and keyboard listener
    def start_tracking(self):
        # Start socket listener in a separate thread
        Thread(target=self.socket_listener, daemon=True).start()
        # Start keyboard listener
        self.start_listener()


if __name__ == "__main__":
    kb_tracker_server = KeyboardTracker(is_bulk=True)
    kb_tracker_server.start_tracking()
