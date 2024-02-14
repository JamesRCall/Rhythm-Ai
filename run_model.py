import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from osu_mania_env import OsuEnv

def main():
    env = OsuEnv()

    check_env(env)

    # Wrap it with a monitor
    env = Monitor(env)

    # Dummy vectorized environment
    env = DummyVecEnv([lambda: env])

    # Frame stacking
    env = VecFrameStack(env, n_stack=4)

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./models/', name_prefix='mania_ai_model')
    learn_steps = 30
    
    try:
        model = PPO.load("rythm_ai_model")
    except:
        model = PPO("CnnPolicy", env, batch_size=516, n_steps=2024, n_epochs=10, verbose=1, gamma=0.99995, learning_rate=0.0001)
    for i in range(learn_steps):
        model.learn(total_timesteps=300000, callback=checkpoint_callback)
            
if __name__ == '__main__':
    main()