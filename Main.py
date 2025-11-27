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
    numTraining = 500
    numTesting = 100
    q_agent = GardenerQAgent()

    while numTraining > 0 or numTesting > 0:
        if numTraining > 0:
            numTraining -= 1
        else:
            numTesting -= 1
        obs, info = env.reset()
        done = False

        # RL agent test
        state = ObservationState.from_obs(obs)
        q_agent.registerInitialState(state)

        print("Starting episode {}\n".format(numTraining))

        while not done:
            action = q_agent.getAction(state)
            #mask = obs["action_mask"]
            #valid_actions = [i for i in range(len(mask)) if mask[i] == 1]
            #action = random.choice(valid_actions)
            #gar = env.unwrapped
            #gar.sample(3, 500)
            #features = gar.get_features(action)
            #print(features)
            obs, reward, terminated, truncated, info = env.step(action)
            state = ObservationState.from_obs(obs)
            q_agent.observeTransition(action, state, reward)
            env.render()
            if numTraining == 0:
                time.sleep(0.5)
            done = terminated or truncated

    env.close()