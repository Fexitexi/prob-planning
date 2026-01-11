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
    actions = 5
    horizon = 5
    n_rot = 60
    epsilon = 0.05
    delta = 0.05
    n_asp = math.ceil((1/(2 * math.pow(epsilon, 2))) * math.log((2*math.pow(actions, horizon))/delta))
    print(f"Number of asp: {n_asp}")
#
    # load the pre-trained weights
    q_agent = GardenerQAgent()
    asp_transformer = ASPTransformer(q_agent)
    q_agent.stopLearning()
    q_agent.load_weights("weights.pkl")

    # test the new loop
    seed = random.randint(0, 1000000)
    # saved seeds: 788618, 784741, 692529
    print(f"Seed: {692529}")
    obs, info = env.reset(seed=692529)
    done = False

    state = ObservationState.from_obs(obs)
    static = asp_transformer.build_static(state, horizon)
    lake_full = state.lakes_full
    actions = []
    step = 0
    full_gen_time = 0
    full_check_time = 0
    full_fixing_time = 0
    gen_count = 0
    check_count = 0
    fixing_count = 0
    sips = {}
    while not done:
        step += 1
        rot_count = 1
        if sips:
            print(f"Sips: {sips}")
        #print(f"Step: {step}")
        # check rule of three
        # disable for now, testing rot in ASP
        #rot = True
        #for i in range(n_rot):
        #    rot, executed_actions = gar.simulate_samples(horizon, q_agent, actions)
        #    if not rot: break
        #print(f"Sampling (rot) took {time.time() - start_time:.6f} seconds.")
        asp_transformer.reset()
        asp_transformer.build_dynamic_worlds(state, n_asp, horizon, sips)
        # todo this only works as long as the policy is deterministic, otherwise I need to use the ASP program
        _, executed_actions = gar.simulate_samples(horizon, q_agent, actions)
        start_time = time.time()
        new_violations, rot = asp_transformer.call_clingo_check(state, executed_actions, [], n_rot, rot_count, sips)
        check_count += 1
        full_check_time += time.time() - start_time
        if rot:
            # rule of three is fulfilled, execute RL policy
            if len(actions) > 0:
                action = actions.pop(0)
            else:
                action = executed_actions[0]
        else:
            # rule of three is not fulfilled, create emergency fix

            # here we should start the loop
            tested_policies = [executed_actions]
            violations = new_violations

            fixing_start_time = start_time
            fixing_count+=1
            while True:
                print(rot_count)
                start_time = time.time()
                policy_fix = asp_transformer.call_clingo_generate(state, violations)
                gen_count += 1
                full_gen_time += time.time() - start_time
                if rot_count != -1:
                    if policy_fix not in tested_policies:
                        rot_count += 1
                        tested_policies.append(policy_fix)
                    else:
                        rot_count = -1
                else:
                    if policy_fix in tested_policies:
                        actions = policy_fix
                        action = actions.pop(0)
                        break
                start_time = time.time()
                new_violations, rot = asp_transformer.call_clingo_check(state,
                                                                        policy_fix,
                                                                        violations,
                                                                        n_rot,
                                                                        rot_count, sips)
                check_count += 1
                if rot:
                    actions = policy_fix
                    action = actions.pop(0)
                    break
                else:
                    for v in new_violations:
                        if v not in violations:
                            violations.append(v)
                full_check_time += time.time() - start_time
            fixing_time = time.time() - fixing_start_time
            full_fixing_time += fixing_time

        obs, reward, terminated, truncated, info = env.step(action)
        state = ObservationState.from_obs(obs)

        # test
        remove = []
        for s in sips:
            sips[s][1] -= 1
            if sips[s][1] == 0:
                remove.append(s)
            elif state.agent[0] == state.frogs[sips[s][0]][0] and state.agent[1] == state.frogs[sips[s][0]][1]:
                remove.append(s)
        for r in remove:
            sips.pop(r)
        new_lake_full = state.lakes_full
        for lake in range(len(lake_full)):
            if lake_full[lake] and not new_lake_full[lake]:
                for i,(c,r) in enumerate(state.frogs):
                    if abs(state.lakes[lake][0] - c) + abs(state.lakes[lake][1] - r) == 1:
                        print(f"Frog {i} is at lake {lake}")
                        sips[lake] = [i, 4]
        lake_full = new_lake_full.copy()



        elapsed = time.time() - start_time
        sleep = max(0, 0.0 - elapsed)
        time.sleep(sleep)
        env.render()
        done = terminated or truncated
    print(f"Full generation time: {full_gen_time:.6f} seconds, Full checking time: {full_check_time:.6f} seconds.")
    print(f"Full generation time: {full_gen_time:.6f} seconds, Full checking time: {full_check_time:.6f} seconds, Full fixing time: {full_fixing_time:.6f} seconds.")
    print(f"Average gen time: {full_gen_time/gen_count:.6f} seconds, Average check time: {full_check_time/gen_count:.6f} seconds, Average fixing time: {full_fixing_time/fixing_count:.6f} seconds.")



    env.close()