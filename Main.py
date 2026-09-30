import argparse
import math
import random
import time

import gymnasium as gym

from config import Config, SamplingMode
from env.ASPTransformer import ASPTransformer
from env.GardenerEnv import GardenerEnv
from env.GardenerQAgent import GardenerQAgent
from env.MCTSNode import MCTSNode
from env.Sequential import SeqentialCheck
from env.simulation_state import SimulationState
from env.state import ObservationState
from env.Statistics import StatisticsCounter

gym.envs.registration.register(
    id="GardenerEnv-v0",
    entry_point="env.GardenerEnv:GardenerEnv",
)


def run(config: Config, statsCounter: StatisticsCounter, seed: int | None = None):
    global env
    env = gym.make("GardenerEnv-v0", size=config.size, frogs=config.frogs)
    gar: GardenerEnv = env.unwrapped

    # load the pre-trained weights
    q_agent = GardenerQAgent()
    asp_transformer = ASPTransformer(q_agent, config.sampling.mode)
    q_agent.stopLearning()
    q_agent.load_weights("weights.pkl")

    # test the new loop
    if seed is None:
        seed = random.randint(0, 1000000)
    # saved seeds: 788618, 784741, 692529, 723724, 155116, 352561,540491,468544
    # seed = 978930
    # print(f"Seed: {seed}")
    obs, info = env.reset(seed=seed, options={"save_screenshot": False})
    done = False

    state: ObservationState = ObservationState.from_obs(obs)
    asp_transformer.build_static(state, config.horizon)
    actions = []
    step = 0
    gen_count = 0
    rejected_count = 0
    samples_arr = []
    check_times = []
    gen_times = []
    full_times = []
    mtn_1 = 0
    mtn_2 = 0
    ctd = 0
    checks = 0
    typeIerror = 0
    typeIIerror = 0

    # cache frequently accessed config values for the hot loop
    sampling_mode = config.sampling.mode
    horizon = config.horizon
    n_rot = math.ceil(
        math.log(config.sampling.delta) / math.log(1 - config.sampling.epsilon)
    )
    n_asp = math.ceil(
        (1 / (2 * math.pow(config.sampling.epsilon, 2)))
        * math.log((2 * math.pow(5, horizon)) / config.sampling.delta)
    )

    while not done:
        check_time = 0
        gen_time = 0
        sampled_trajectories = 0

        start_full_time = time.time()
        step += 1
        rot_count = 1
        asp_transformer.reset(config.ctd)
        asp_transformer.build_dynamic_worlds(state, n_asp, horizon)
        _, executed_actions = gar.simulate_samples(horizon, q_agent, actions)
        start_time_check = time.time()
        match sampling_mode:
            case SamplingMode.RANDOM:
                new_violations, rot, samples = asp_transformer.call_clingo_check(
                    state, executed_actions, [], n_rot, rot_count, False
                )
            case SamplingMode.SEQUENTIAL:
                check = SeqentialCheck(
                    gar._state,
                    config.sampling.epsilon,
                    config.sampling.delta,
                    horizon,
                )
                new_violations, rot, samples = check.check(
                    executed_actions, n_asp, False
                )

            case SamplingMode.MCTS:
                sim_state = SimulationState.from_state(gar._state, config.ctd)
                node = MCTSNode(sim_state, executed_actions[0], horizon)
                new_violations, rot, samples = node.check_MCTS(
                    executed_actions,
                    config.sampling.epsilon,
                    config.sampling.delta,
                    config.sampling.indifference,
                )
            case _:
                raise NotImplementedError(
                    f"Sampling mode {sampling_mode!r} is not wired in Main.py"
                )
        end_time_check = time.time()
        check_time += end_time_check - start_time_check
        sampled_trajectories += samples

        if config.logLevel > 2:
            sim_state: SimulationState = SimulationState.from_state(
                gar._state, config.ctd
            )
            real_probability = sim_state.calculate_violation_probability_brute_force(
                executed_actions, horizon
            )
            checks += 1
            if rot and real_probability > config.sampling.epsilon:
                typeIIerror += 1
            if not rot and real_probability <= config.sampling.epsilon:
                typeIerror += 1

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
            acceptPolicy = sampled_trajectories >= n_asp

            while True:
                # TODO: this is inaccurate for MCTS
                acceptPolicy = sampled_trajectories >= n_asp or acceptPolicy
                gen_count += 1
                fix_gen_start = time.time()
                policy_fix = asp_transformer.call_clingo_generate(state, violations)
                fix_gen_end = time.time()
                gen_time += fix_gen_end - fix_gen_start

                if not acceptPolicy:
                    if policy_fix not in tested_policies:
                        rot_count += 1
                        tested_policies.append(policy_fix)
                    else:
                        acceptPolicy = True
                else:
                    actions = policy_fix
                    action = actions.pop(0)
                    break

                start_time_check = time.time()
                match sampling_mode:
                    case SamplingMode.RANDOM:
                        new_violations, rot, samples = (
                            asp_transformer.call_clingo_check(
                                state,
                                policy_fix,
                                violations,
                                n_rot,
                                rot_count,
                                acceptPolicy,
                            )
                        )
                    case SamplingMode.SEQUENTIAL:
                        new_violations, rot, samples = check.check(
                            policy_fix, n_asp - sampled_trajectories, acceptPolicy
                        )
                    case SamplingMode.MCTS:
                        node = MCTSNode(sim_state, policy_fix[0], horizon)
                        new_violations, rot, samples = node.check_MCTS(
                            policy_fix,
                            config.sampling.epsilon,
                            config.sampling.delta,
                            config.sampling.indifference,
                        )
                    case _:
                        raise NotImplementedError(
                            f"Sampling mode {sampling_mode!r} is not wired in Main.py"
                        )
                end_time_check = time.time()
                check_time += end_time_check - start_time_check
                sampled_trajectories += samples
                if config.logLevel > 2:
                    checks += 1
                    # calculate real_probability via brute force
                    sim_state: SimulationState = SimulationState.from_state(
                        gar._state, config.ctd
                    )
                    real_probability = (
                        sim_state.calculate_violation_probability_brute_force(
                            policy_fix, horizon
                        )
                    )

                    if rot and real_probability > config.sampling.epsilon:
                        typeIIerror += 1
                    if not rot and real_probability <= config.sampling.epsilon:
                        typeIerror += 1

                if rot:
                    # cache
                    actions = policy_fix
                    action = actions.pop(0)
                    break
                else:
                    rejected_count += 1
                    for v in new_violations:
                        if v not in violations:
                            violations.append(v)

        obs, reward, terminated, truncated, info = env.step(action)
        state: ObservationState = ObservationState.from_obs(obs)

        # for visualization purposes
        # sleep = max(0, 0.1)
        # time.sleep(sleep)
        if config.render:
            env.render()
        done = terminated or truncated
        end_full_time = time.time()
        full_times.append(end_full_time - start_full_time)
        check_times.append(check_time)
        gen_times.append(gen_time)
        samples_arr.append(sampled_trajectories)

    mtn_1 = state.dead_frogs.sum()
    mtn_2 = state.stun_counter
    ctd = state.stun_counter - state.capt_frogs.sum()

    if gen_count == 0:
        rej_percent = 0
    else:
        rej_percent = rejected_count / gen_count

    statsCounter.record_Round(
        step,
        rej_percent,
        samples_arr,
        check_times,
        gen_times,
        full_times,
        mtn_1,
        mtn_2,
        ctd,
    )
    if config.logLevel > 2:
        print(f"type I: {typeIerror / checks}")
        print(f"type II: {typeIIerror / checks}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indifference",
        type=float,
        default=0.1,
        help="specifies the radius of the indifference interval as percent of epsilon",
    )
    parser.add_argument(
        "--sampling", type=int, default=1, help="0 - random / 1 - sequential / 2 - MCTS"
    )
    parser.add_argument("--horizon", type=int, default=5, help="Horizon")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--ctd", type=int, default=1, choices=[0, 1], help="0 - no ctd / 1 - ctd"
    )
    parser.add_argument(
        "--logLevel",
        type=int,
        default=0,
        help="determines the level of statistics printed",
    )
    parser.add_argument(
        "--render",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 - no render / 1 - render",
    )
    parser.add_argument("--size", type=int, default=15, help="grid size")
    parser.add_argument(
        "--frogs", type=float, default=0.01, help="percent of spaces filled with frogs"
    )
    parser.add_argument("--epsilon", type=float, default=0.05, help="error tolerance")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="confidence delta (0.05 = 95%% confidence)",
    )
    args = parser.parse_args()

    config = Config.from_args(args)

    random.seed(config.seed)
    seeds = [random.randint(0, 1000000) for _ in range(config.rounds)]

    statsCounter = StatisticsCounter(config.rounds)
    for i in range(config.rounds):
        run(config, statsCounter, seeds[i])

    statsCounter.print_statistics(config.logLevel)
