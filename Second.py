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
    actions = []
    while not done:
        start_time = time.time()
        # check rule of three
        rot = True
        for i in range(n_rot):
            rot = gar.simulate_samples(horizon, q_agent, actions)
            if not rot: break
        #print(f"Sampling (rot) took {time.time() - start_time:.6f} seconds.")

        if rot:
            # rule of three is fulfilled, execute RL policy
            if len(actions) > 0:
                action = actions.pop(0)
            else:
                action = q_agent.getAction(state)
        else:
            # rule of three is not fulfilled, create emergency fix
            start_time_asp = time.time()
            dynamic = asp_transformer.build_dynamic_worlds(state, n_asp, horizon)
            dynamic_time = time.time()
            actions = asp_transformer.call_clingo_new(static, dynamic, horizon)
            solve_time = time.time()
            action = actions.pop(0)
            elapsed_asp = time.time() - start_time_asp
            elapsed_dynamic = dynamic_time - start_time_asp
            elapsed_solve = solve_time - dynamic_time
            print(f"clingo took {elapsed_asp:.6f} seconds of which {elapsed_dynamic:.6f} seconds were spent on dynamic worlds and {elapsed_solve:.6f} seconds on solving.")

        obs, reward, terminated, truncated, info = env.step(action)
        state = ObservationState.from_obs(obs)
        elapsed = time.time() - start_time
        sleep = max(0, 0.5 - elapsed)
        time.sleep(sleep)
        env.render()
        done = terminated or truncated



    env.close()