import gymnasium as gym
import time
import random

from env.GardenerQAgent import GardenerQAgent
from env.state import ObservationState

gym.envs.registration.register(
    id="GardenerEnv-v0",
    entry_point="env.GardenerEnv:GardenerEnv",
)

if __name__ == "__main__":
    env = gym.make("GardenerEnv-v0")
    obs, info = env.reset(seed=42)
    done = False

    # RL agent test
    q_agent = GardenerQAgent()

    while not done:
        mask = obs["action_mask"]
        valid_actions = [i for i in range(len(mask)) if mask[i] == 1]
        action = random.choice(valid_actions)
        gar = env.unwrapped
        gar.sample(3, 500)
        #features = gar.get_features(action)
        #print(features)
        obs, reward, terminated, truncated, info = env.step(action)
        state = ObservationState.from_obs(obs)

        env.render()
        time.sleep(1)
        done = terminated or truncated

    env.close()