import mss
import mss.tools
import numpy as np
import cv2
import gymnasium as gym
import pyautogui
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from pyautogui import press
from stable_baselines3.common.env_checker import check_env
import json
import time
import keyboard
from websocket import create_connection
import threading
import cProfile
import ctypes

should_terminate = False
rewards_data = []

def on_exit_key(self):
    model.save("rythm_ai_model")
    global should_terminate  # Declare it global so that we can modify it
    should_terminate = True
    print("Exit key pressed, preparing to exit...")

keyboard.on_press_key('q', on_exit_key)

class OsuEnv(gym.Env):
    VK_CODE = {'a': 0x41, 'd': 0x44, 'f': 0x46, 'j': 0x4A, 'k': 0x4B}    

    key_states = {'d': 'up', 'f': 'up', 'j': 'up', 'k': 'up'}

    def __init__(self):
        super(OsuEnv, self).__init__()

        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2, 2])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.done = False
        self.current_score = 0
        self.current_accuracy = 100
        self.width = 600
        self.height = 810
        self.reward = 0
        self.action_state = 0
        # Initialize game state
        self.game_state = None  # None means the state is unknown
        self.data_thread = threading.Thread(target=self.collect_data)
        self.data_thread.start()

    

    def press_key(self, key):
        key = OsuEnv.VK_CODE[key]
        ctypes.windll.user32.keybd_event(key, 0, 0, 0)

    def release_key(self, key):
        key = OsuEnv.VK_CODE[key]
        ctypes.windll.user32.keybd_event(key, 0, 2, 0)

    def capture_observation(self):
        with mss.mss() as sct:
            top = 1080 - 600
            left = (1920 - 600) // 2
            bbox = {'top': top, 'left': left, 'width': 600, 'height': 600}
            screenshot = sct.grab(bbox)
            image_np = np.array(screenshot)
            grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_BGRA2GRAY)
            
            # Resize to 84x84
            resized_image = cv2.resize(grayscale_image, (84, 84))
            
            # Add channel dimension
            observations = resized_image

        return observations


    def collect_data(self):
        global should_terminate  # Declare it global so that we can read it
        uri = "ws://127.0.0.1:24050/ws"
        ws = create_connection(uri)
        print("Connected to WebSocket")
        while not should_terminate:  # Added a termination condition here
            try:
                data = ws.recv()
                json_data = json.loads(data)
                
                gameplay_data = json_data.get('gameplay', {})
                # hits_data = gameplay_data.get('hits', {})
                self.new_data_event = threading.Event()
                # Update game state
                self.game_state = json_data.get('menu', {}).get('state', None)

                self.gameplay_data = json_data.get('menu', {})
                
                # with open("output.txt", "w") as f:
                #     f.write(f"{json_data}\n")
                # self.h0 = hits_data.get('0', 0)
                # self.h50 = hits_data.get('50', 0)
                # self.h100 = hits_data.get('100', 0)
                # self.h200 = hits_data.get('katu', 0)
                # self.h300 = hits_data.get('300', 0)
                # self.h320 = hits_data.get('geki', 0)

                self.current_score = gameplay_data.get('score', 0)
                self.current_accuracy = gameplay_data.get('accuracy', 100)  # Assuming accuracy is in percentage
                

            except Exception as e:
                print(f"An error occurred: {e}")
        print("Terminating  thread.")

    # def step(self, action, profile=False):
    def step(self, action):
        # Action is an array of length 4, where each value is either 0, 1, or 2.
        # These correspond to "release", "click", or "hold" for each of the four lanes.
        while not should_terminate:
            
            if self.game_state != 2 and self.action_state == 1:
                print("Starting Next Song...")
                self.release_all_keys()
                self.update_rewards_data()
                time.sleep(10)
                pyautogui.click(100, 1000)
                time.sleep(5)
                pyautogui.click(600, 1000)
                time.sleep(5)
                pyautogui.click(1400, 500)
                pyautogui.click(1400, 500)
                dummy_observation = np.zeros((84, 84, 1), dtype=np.uint8)
                dummy_reward = 0
                self.done = True  # Or False, depending on how you want to handle this
                info = {}
                return dummy_observation, dummy_reward, self.done, False, info
            else:
                lane_keys = ['d', 'f', 'j', 'k']

                for i, lane_key in enumerate(lane_keys):
                    state = action[i]  # No need to round, as action[i] is already an integer (0, 1, or 2)

                    if state == 0:  # Release if holding
                        if self.key_states[lane_key] == 'down':
                            self.release_key(lane_key)
                            self.key_states[lane_key] = 'up'
                    elif state == 1:  # Hold down if not already
                        if self.key_states[lane_key] == 'up':
                            self.press_key(lane_key)
                            self.key_states[lane_key] = 'down'
                    
                observation = self.capture_observation()
                observation = observation.reshape((84, 84, 1))
                

                # TODO: Define your reward function here, for now using a placeholder
                normalized_score = self.current_score / 500000
                normalized_diff_from_perfect_accuracy = (100 - self.current_accuracy) / 100  

                weight_for_diff_from_perfect_accuracy = 0.1 * normalized_score * normalized_diff_from_perfect_accuracy  

                self.reward = normalized_score - weight_for_diff_from_perfect_accuracy

                # TODO: Set done condition here, for now always False
                self.done = (self.game_state != 2)
                # if profile:
                #     pr.disable()
                #     pr.dump_stats("step_output.txt")

                

                return observation, self.reward, self.done, False, {}
    
    def release_all_keys(self):
        lane_keys = ['d', 'f', 'j', 'k']

        for lane_key in lane_keys:
            if self.key_states[lane_key] == 'down':
                self.release_key(lane_key)
                self.key_states[lane_key] = 'up'

    def reset(self, seed=None, options=None):
        observation = self.capture_observation() # Add channel dimension
        observation = observation.reshape((84, 84, 1))
        return observation, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def update_rewards_data(self):
        global rewards_data
        # Append the reward to the list
        rewards_data.append(self.reward)

        # Plot and save the rewards graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(rewards_data)), rewards_data, marker='o')
        plt.xlabel('Episode or Song Number')
        plt.ylabel('Reward (0-1)')
        plt.title('AI Rewards Over Time')
        plt.grid(True)

        # Save the graph as an image (you can choose the format)
        plt.savefig('ai_rewards_graph.png')

def main():
    global should_terminate
    global model
    env = OsuEnv()
    check_env(env)
    

    try:
        model = PPO.load("rythm_ai_model")
    except:
        model = PPO("CnnPolicy", env, batch_size=128, n_steps=512, n_epochs=8, verbose=1, gamma=0.995)
    
    model.learning_rate=0.0001

    while not should_terminate and env.action_state != 2:
        if env.game_state == 2 or env.action_state == 1:
            env.action_state = 1
            model.learn(total_timesteps=300000)
            
        else:
            print("Game is not in the playing state, waiting...")
            time.sleep(1)
            
                
            

if __name__ == '__main__':
    main()
