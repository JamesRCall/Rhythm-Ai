# OsuDataCollector.py
import json
from websocket import create_connection
import mss
import numpy as np
from collections import deque
import cv2
import gymnasium as gym
import threading
import keyboard

class OsuDataCollector():
    def __init__(self):
        self.should_terminate = False
        self.game_state = None
        self.accuracy = 1
        self.score = 0
        self.data_thread = threading.Thread(target=self.collect_data)
        self.data_thread.start()
        self.frame_stack_size = 4  # number of frames to stack
        self.frame_stack = deque([], maxlen=self.frame_stack_size)
        # Initialize with zero frames
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(np.zeros((42, 126, 1), dtype=np.uint8))
    
    def on_exit_key(self):
        self.should_terminate = True
        print("Exit key pressed, preparing to exit...")

    keyboard.on_press_key('esc', on_exit_key)
    
    def collect_data(self):
        uri = "ws://127.0.0.1:24050/ws"
        ws = create_connection(uri)
        print("Connected to WebSocket")
        while not self.should_terminate:
            try:
                data = ws.recv()
                json_data = json.loads(data)
                gameplay_data = json_data.get('gameplay', {})  # Local variable to hold gameplay data
                self.accuracy = gameplay_data.get('accuracy', 100)  # Storing as a class variable
                self.score = gameplay_data.get('score', 0)  # Storing as a class variable
                self.game_state = json_data.get('menu', {}).get('state', None)
            except Exception as e:
                print(f"An error occurred: {e}")
        print("Terminating thread.")

    def get_current_score(self):
        return self.score, self.accuracy
    
    def capture_observation(self):
        with mss.mss() as sct:
            top = 1080 - 200
            left = (1920 - 600) // 2
            bbox = {'top': top, 'left': left, 'width': 600, 'height': 200}
            screenshot = sct.grab(bbox)
            image_np = np.array(screenshot)
            grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_BGRA2GRAY)
            
            # Resize to 126x42
            resized_image = cv2.resize(grayscale_image, (126, 42))
            
            # Add channel dimension
            frame = np.expand_dims(resized_image, axis=-1)
            
            # Append frame to deque
            self.frame_stack.append(frame)
            
            # Stack frames along the last dimension to make it (126, 42, frame_stack_size)
            stacked_frames = np.concatenate(list(self.frame_stack), axis=-1) if len(self.frame_stack) == self.frame_stack_size else None

            # Reshape the observation with frame_stack_size at the front
            if stacked_frames is not None:
                stacked_frames = stacked_frames.transpose(2, 0, 1)

                # Save the observation as an image
            
            return stacked_frames
