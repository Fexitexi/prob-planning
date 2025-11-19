import gymnasium as gym
import time
import random

gym.envs.registration.register(
    id="GardenerEnv-v0",
    entry_point="env.GardenerEnv:GardenerEnv",
)

if __name__ == "__main__":
    env = gym.make("GardenerEnv-v0")
    obs, info = env.reset(seed=42)
    done = False

    while not done:
        mask = obs["action_mask"]
        valid_actions = [i for i in range(len(mask)) if mask[i] == 1]
        action = random.choice(valid_actions)
        gar = env.unwrapped
        gar.sample(3, 500)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1)
        done = terminated or truncated
        print(f"Obs: {obs}, Reward: {reward}")

    env.close()