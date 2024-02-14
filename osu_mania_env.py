import numpy as np
import gymnasium as gym
import pyautogui
import matplotlib.pyplot as plt
import time
import ctypes
from osu_data import OsuDataCollector

should_terminate = False


class OsuEnv(gym.Env):
    def __init__(self):
        super(OsuEnv, self).__init__()
        self.key_states = {'d': 'up', 'f': 'up', 'j': 'up', 'k': 'up'}
        self.VK_CODE = {'a': 0x41, 'd': 0x44, 'f': 0x46, 'j': 0x4A, 'k': 0x4B}    
        self.rewards_data = []
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2, 2])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 126, 42), dtype=np.uint8)
        self.done = False
        self.reward = 0
        self.osu_data = OsuDataCollector()
        self.enabled = False
        # Initialize game state

    def press_key(self, key):
        key = self.VK_CODE[key]
        ctypes.windll.user32.keybd_event(key, 0, 0, 0)

    def release_key(self, key):
        key = self.VK_CODE[key]
        ctypes.windll.user32.keybd_event(key, 0, 2, 0)

    def step(self, action):
        # Action is an array of length 4, where each value is either 0, 1, or 2.
        # These correspond to "release", "click", or "hold" for each of the four lanes.
        while not self.osu_data.should_terminate:
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
                
            observation = self.render()
            observation = observation.reshape((4, 126, 42))
            
            current_score, current_accuracy = self.osu_data.get_current_score()
            normalized_score = current_score / 250000
            normalized_diff_from_perfect_accuracy = (100 - current_accuracy) / 100  

            weight_for_diff_from_perfect_accuracy = 0.1 * normalized_score * normalized_diff_from_perfect_accuracy  

            self.reward = normalized_score - weight_for_diff_from_perfect_accuracy

            self.done = (self.osu_data.game_state != 2)

            return observation, self.reward, self.done, False, {}
    
    def release_all_keys(self):
        lane_keys = ['d', 'f', 'j', 'k']

        for lane_key in lane_keys:
            if self.key_states[lane_key] == 'down':
                self.release_key(lane_key)
                self.key_states[lane_key] = 'up'

    def reset(self, seed=None, options=None):
        while not should_terminate and self.osu_data.game_state != 2:
            if self.enabled == True:
                self.seed = seed
                
                self.release_all_keys()
                self.update_rewards_data()

                self.reward = 0
                self.osu_data.score = 0
                self.osu_data.accuracy = 0

                print("Starting Next Song...")

                time.sleep(10)
                pyautogui.click(100, 1000)
                time.sleep(5)
                pyautogui.click(600, 1000)
                time.sleep(5)
                pyautogui.click(1400, 500)
                pyautogui.click(1400, 500)
                time.sleep(1)
    
                observation = np.zeros((4, 126, 42), dtype=np.uint8) # Add channel dimension
                observation = observation.reshape((4, 126, 42))

                return observation, {}
            else:
                print("Game is not in the playing state, waiting...")
                time.sleep(1)
                if self.osu_data.game_state == 2:
                    self.enabled = True
        # To handle the case when `self.enabled` is not True
        return np.zeros((4, 126, 42), dtype=np.uint8), {}
        

    def render(self):
        return self.osu_data.capture_observation()

    def close(self):
        pass

    def update_rewards_data(self):
        # Append the reward to the list
        self.rewards_data.append(self.reward)

        # Plot and save the rewards graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.rewards_data)), self.rewards_data, marker='o')
        plt.xlabel('Episode or Song Number')
        plt.ylabel('Reward (0-1)')
        plt.title('AI Rewards Over Time')
        plt.grid(True)

        # Save the graph as an image (you can choose the format)
        plt.savefig('ai_rewards_graph.png')