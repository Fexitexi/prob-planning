import math

import gymnasium as gym
import time
import random

from env.ASPTransformer import ASPTransformer
from env.GardenerQAgent import GardenerQAgent
from env.state import ObservationState

gym.envs.registration.register(
    id="GardenerEnv-v0",
    entry_point="env.GardenerEnv:GardenerEnv",
)

if __name__ == "__main__":
    env = gym.make("GardenerEnv-v0")
    gar = env.unwrapped

    # parameter
    horizon = 3
    n_rot = 60
    n_asp = 1700


    # load the pre-trained weights
    q_agent = GardenerQAgent()
    asp_transformer = ASPTransformer(q_agent)
    q_agent.stopLearning()
    q_agent.load_weights("weights.pkl")

    # test the new loop
    obs, info = env.reset()
    done = False

    state = ObservationState.from_obs(obs)
    static = asp_transformer.build_static(state, horizon)

    while not done:
        start_time = time.time()
        # check rule of three
        rot = True
        for i in range(n_rot):
            rot = gar.simulate_samples(horizon, q_agent)
            if not rot: break

        if rot:
            # rule of three is fulfilled, execute RL policy
            action = q_agent.getAction(state)
        else:
            # rule of three is not fulfilled, create emergency fix
            dynamic = asp_transformer.build_dynamic_worlds(state, n_asp, horizon)
            actions = asp_transformer.call_clingo_new(static, dynamic, horizon)
            action = actions[0]

        obs, reward, terminated, truncated, info = env.step(action)
        state = ObservationState.from_obs(obs)
        elapsed = time.time() - start_time
        sleep = max(0, 0.5 - elapsed)
        time.sleep(sleep)
        env.render()
        done = terminated or truncated



    env.close()