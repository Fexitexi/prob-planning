import math

import gymnasium as gym
import time
import random

from numpy.ma.core import count

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
#
    # load the pre-trained weights
    q_agent = GardenerQAgent()
    asp_transformer = ASPTransformer(q_agent)
    q_agent.stopLearning()
    q_agent.load_weights("weights.pkl")

    # test the new loop
    seed = random.randint(0, 1000000)
    print(f"Seed: {seed}")
    obs, info = env.reset(seed=318687)
    done = False

    state = ObservationState.from_obs(obs)
    static = asp_transformer.build_static(state, horizon)
    actions = []
    step = 0
    while not done:
        step += 1
        #print(f"Step: {step}")
        start_time = time.time()
        # check rule of three
        rot = True
        for i in range(n_rot):
            rot, executed_actions = gar.simulate_samples(horizon, q_agent, actions)
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
            asp_transformer.reset()
            # here we should start the loop
            convergence = asp_transformer.add_constraint(executed_actions)
            asp_transformer.build_dynamic_worlds(state, n_asp, horizon, append_lines=False)
            count = 0
            violations = []
            generate_time_sum = 0
            check_time_sum = 0
            while not convergence:
                count += 1
                generate_time = time.time()
                policy_fix = asp_transformer.call_clingo_generate(state, violations)
                generate_time_sum += time.time() - generate_time
                check_time = time.time()

                new_violations = asp_transformer.call_clingo_check(state, policy_fix)
                print(f"Number of violations: {len(new_violations)} with actions: {policy_fix}")
                check_time_sum += time.time() - check_time
                for violation in new_violations:
                    if violation not in violations: violations.append(violation)
                if len(new_violations) > 0:
                    convergence = asp_transformer.add_constraint(policy_fix)
                    if convergence:
                        actions = policy_fix
                        action = actions.pop(0)
                    #policy_fix = asp_transformer.call_clingo_generate()
                    #exit(1)
                else:
                    convergence = True
                    actions = policy_fix
                    action = actions.pop(0)
            print(f"Time for generating worlds: {generate_time_sum:.6f} seconds, Time for checking worlds: {check_time_sum:.6f} seconds, Number of iterations: {count}.")

        obs, reward, terminated, truncated, info = env.step(action)
        state = ObservationState.from_obs(obs)
        elapsed = time.time() - start_time
        sleep = max(0, 0.1 - elapsed)
        time.sleep(sleep)
        env.render()
        done = terminated or truncated



    env.close()