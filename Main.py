import math
import argparse
import gymnasium as gym
import time
import random

from numpy.ma.core import count, ceil

from env.ASPTransformer import ASPTransformer
from env.GardenerQAgent import GardenerQAgent
from env.old.ClingoHelper import ClingoHelperOld
from env.state import ObservationState

gym.envs.registration.register(
    id="GardenerEnv-v0",
    entry_point="env.GardenerEnv:GardenerEnv",
)


def run(method=0, seed=None, ctd=False, horizon=3, size=15, epsilon=0.05, delta=0.05, render=False):
    global env
    env = gym.make("GardenerEnv-v0", size=size)
    gar = env.unwrapped

    # parameter
    n_actions = 5
    n_rot = math.ceil(math.log(delta)/math.log(1-epsilon))
    n_asp = math.ceil((1 / (2 * math.pow(epsilon, 2))) * math.log(
        (2 * math.pow(n_actions, horizon)) / delta))
    #print(f"Number of asp: {n_asp}")

    # load the pre-trained weights
    q_agent = GardenerQAgent()
    asp_transformer = ASPTransformer(q_agent)
    q_agent.stopLearning()
    q_agent.load_weights("weights.pkl")

    # test the new loop
    if seed is None:
        seed = random.randint(0, 1000000)
    # saved seeds: 788618, 784741, 692529, 723724, 155116, 352561,540491,468544
    #seed = 978930
    #print(f"Seed: {seed}")
    obs, info = env.reset(seed=seed, options={"save_screenshot": False})
    done = False

    state = ObservationState.from_obs(obs)
    asp_transformer.build_static(state, horizon)
    lake_full = state.lakes_full
    actions = []
    step = 0
    check_times = []
    fix_times = []
    gen_count = 0
    fixing_count = 0
    ctd_success = 0
    ctd_failure = 0
    frogs_killed = 0
    rot_counts = []
    sips = {}
    intervention_count = 0

    # old
    clingoHelperOld = ClingoHelperOld(state, q_agent, horizon + 1, 10)
    clingoHelperOld.setup()

    while not done:
        step += 1
        rot_count = 1
        if method < 2:
            start_time_check = time.time()
            asp_transformer.reset(ctd)
            asp_transformer.build_dynamic_worlds(state, n_asp, horizon, sips)
            _, executed_actions = gar.simulate_samples(horizon, q_agent,
                                                       actions)
            new_violations, rot = asp_transformer.call_clingo_check(state,
                                                                    executed_actions,
                                                                    [], n_rot,
                                                                    rot_count,
                                                                    sips)
            end_time_check = time.time()
            check_times.append(end_time_check - start_time_check)
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

                fixing_count += 1
                while True:
                    # print(rot_count)
                    policy_fix = asp_transformer.call_clingo_generate(state,
                                                                      violations)
                    gen_count += 1
                    if rot_count != -1:
                        if policy_fix not in tested_policies:
                            rot_count += 1
                            tested_policies.append(policy_fix)
                        else:
                            rot_count = -1
                    else:
                        if policy_fix in tested_policies:
                            # cache
                            if method == 1:
                                actions = policy_fix
                                action = actions.pop(0)
                            else:
                                action = policy_fix.pop(0)
                            break
                    new_violations, rot = asp_transformer.call_clingo_check(
                        state,
                        policy_fix,
                        violations,
                        n_rot,
                        rot_count, sips)
                    if rot:
                        #cache
                        if method == 1:
                            actions = policy_fix
                            action = actions.pop(0)
                        else:
                            action = policy_fix.pop(0)
                        break
                    else:
                        for v in new_violations:
                            if v not in violations:
                                violations.append(v)
            end_time_gen = time.time()
            fix_times.append(end_time_gen - start_time_check)
            best_actions = q_agent.getBestActions(state)
            if action not in best_actions:
                intervention_count += 1
            rot_counts.append(rot_count)
        elif method == 2:
            # OLD METHOD EXECUTION
            start_time_gen = time.time()
            best_actions = q_agent.getBestActions(state)
            action = clingoHelperOld.get_action(state)
            if action not in best_actions:
                intervention_count += 1
            fix_times.append(time.time() - start_time_gen)
        elif method == 3:
            action = q_agent.getAction(state)

        obs, reward, terminated, truncated, info = env.step(action)
        state = ObservationState.from_obs(obs)

        # add this to the environment
        remove = []
        for s in sips:
            if state.agent[0] == state.frogs[sips[s][0]][0] and state.agent[
                1] == state.frogs[sips[s][0]][1]:
                remove.append(s)
                ctd_success += 1
                msg = "CDT SUCCESS!"
                gar._state.capt_frogs[sips[s][0]] = True
                #print(f"\033[31m{msg}\033[0m")
            elif sips[s][1] == 0:
                remove.append(s)
                ctd_failure += 1
            sips[s][1] -= 1
        for r in remove:
            sips.pop(r)
        new_lake_full = state.lakes_full
        for lake in range(len(lake_full)):
            if lake_full[lake] and not new_lake_full[lake]:
                #print(f"Lake {lake} is now empty at step {step}.")
                for f, (c, r) in enumerate(state.frogs):
                    prox = False
                    if not state.dead_frogs[f]:
                        if abs(state.lakes[lake][0] - c) + abs(
                            state.lakes[lake][1] - r) == 1:
                            prox = True
                        elif abs(state.lakes[lake][0] - c) + abs(
                            state.lakes[lake][1] - r) == 2 and abs(state.lakes[lake][0] - c) == 1:
                            prox = True
                        elif abs(state.lakes[lake][0] - c) + abs(
                            state.lakes[lake][1] - r) == 2 and abs(state.lakes[lake][1] - r) == 1:
                            prox = True
                    if prox:
                        #print(
                        #    f"Frog {f} is at lake {lake}, which is at coordinates {c}, {r}.")
                        sips[lake] = [f, 4, c, r]
        lake_full = new_lake_full.copy()

        # for visualization purposes
        #sleep = max(0, 0.1)
        #time.sleep(sleep)
        if render:
            env.render()
        done = terminated or truncated

    ctd_triggered = ctd_success + ctd_failure
    if ctd_triggered > 0:
        ctd_success = ctd_success / ctd_triggered
    else:
        ctd_success = 1

    frogs_killed = state.dead_frogs.sum()
    if ctd:
        frogs_killed -= (ctd_success * frogs_killed)

    env.close()
    return step, intervention_count, rot_counts, check_times, fix_times, frogs_killed, ctd_success, ctd_triggered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=int, default=0, help="0 - new framework / 1 - new framework w/ cache / 2 - old framework / 3 - RL")
    parser.add_argument("--horizon", type=int, default=3, help="Horizon")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--ctd", type=int, default=0, help="0 - no ctd / 1 - ctd")
    parser.add_argument("--render", type=int, default=0, help="0 - no render / 1 - render")
    parser.add_argument("--size", type=int, default=15, help="grid size")
    parser.add_argument("--epsilon", type=float, default=0.05, help="error tolerance")
    parser.add_argument("--delta", type=float, default=0.05, help="confidence delta (0.05 = 95% confidence)")
    args = parser.parse_args()

    random.seed(args.seed)
    seeds = []
    for i in range(args.rounds):
        seeds.append(random.randint(0, 1000000))

    # 0 - new framework / 1 - new framework w/ cache / 2 - old framework / 3 - RL
    method = args.method
    rounds = args.rounds
    ctd = args.ctd
    horizon = args.horizon
    render = args.render
    size = args.size
    epsilon = args.epsilon
    delta = args.delta

    all_step = 0
    all_intervention_count = 0
    all_rot_counts = []
    all_check_times = []
    all_fix_times = []
    all_frogs_killed = 0
    all_ctd_success = 0
    all_ctd_triggered = 0
    for i in range(rounds):
        step, intervention_count, rot_counts, check_times, fix_times, frogs_killed, ctd_success, ctd_triggered = run(method, seeds[i], ctd, horizon, size, epsilon, delta, render)
        all_step += step
        all_intervention_count += intervention_count
        all_rot_counts.extend(rot_counts)
        all_check_times.extend(check_times)
        all_fix_times.extend(fix_times)
        all_frogs_killed += frogs_killed
        all_ctd_success += ctd_success
        all_ctd_triggered += ctd_triggered
    if method < 2:
        if all_rot_counts:
            sum_rot = 0
            count_neg = 0
            for r in all_rot_counts:
                if r != -1:
                    sum_rot += r
                else:
                    count_neg += 1
            avg_rot = sum_rot / (len(all_rot_counts) - count_neg)
            max_rot = max(all_rot_counts)
            #print(f"Average rot_checks: {avg_rot:.2f}, Max rot_checks: {max_rot:.2f}, Neg rot_checks: {count_neg}")
        if all_check_times:
            avg_check = sum(all_check_times) / len(all_check_times)
            max_check = max(all_check_times)
            #print(f"Average checking time: {avg_check:.4f}, Max checking time: {max_check:.4f}")
    if all_fix_times and method < 3:
        avg_fix = sum(all_fix_times) / len(all_fix_times)
        max_fix = max(all_fix_times)
        #print(f"Average fixing time: {avg_fix:.4f}, Max fixing time: {max_fix:.4f}")
    #print(f"Steps: {all_step / rounds}, Interventions: {all_intervention_count / rounds}")
    #print(f"Frogs killed: {all_frogs_killed / rounds}")
    if method == 3:
        print(f"{all_step / rounds:.2f}, 0.00, 0.00, {all_frogs_killed / rounds:.2f}, {all_ctd_triggered / rounds:.2f}, {(1 - (all_ctd_success / rounds)) * (all_ctd_triggered / rounds):.2f}")
    else:
        print(f"{all_step / rounds:.2f}, {all_intervention_count / rounds:.2f}, {avg_fix * 1000:.2f}, {all_frogs_killed / rounds:.2f}, {all_ctd_triggered / rounds:.2f}, {(1 - (all_ctd_success / rounds)) * (all_ctd_triggered / rounds):.2f}")
    #if ctd:
    #    print(f"ctd_success: {all_ctd_success / rounds}, ctd_triggered: {all_ctd_triggered / rounds}")
