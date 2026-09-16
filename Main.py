import argparse
import random
import time

import gymnasium as gym

from config import Config, Method, SamplingMode
from env.ASPTransformer import ASPTransformer
from env.GardenerEnv import GardenerEnv
from env.GardenerQAgent import GardenerQAgent
from env.MCTSNode import MCTSNode
from env.Sequential import SeqentialCheck
from env.simulation_state import SimulationState
from env.state import ObservationState
from env.StateLibrary import StateLibrary, custom_edges

gym.envs.registration.register(
    id="GardenerEnv-v0",
    entry_point="env.GardenerEnv:GardenerEnv",
)


def run(config: Config, seed: int | None = None):
    global env
    env = gym.make("GardenerEnv-v0", size=config.size)
    gar: GardenerEnv = env.unwrapped

    # load the pre-trained weights
    q_agent = GardenerQAgent()
    asp_transformer = ASPTransformer(q_agent, config.sampling)
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
    sampled_trajectories = 0
    full_times = []
    check_times = []
    fix_times = []
    gen_times = []
    gen_count = 0
    fixing_count = 0
    frogs_killed = 0
    rot_counts = []
    checkCount = 0
    gen_count = 0
    rejected_count = 0
    intervention_count = 0

    if config.logLevel > 2:
        lib = StateLibrary(
            "sim_states",
            edges=custom_edges(
                0,
                0.005,
                0.01,
                0.15,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
            ),
            max_per_bucket=1000,
        )

    # cache frequently accessed config values for the hot loop
    method = config.method
    sampling_mode = config.sampling.mode
    horizon = config.horizon

    while not done:
        start_full_time = time.time()
        step += 1
        rot_count = 1
        checkCount += 1
        start_time_check = time.time()
        asp_transformer.reset(config.ctd)
        asp_transformer.build_dynamic_worlds(state)
        _, executed_actions = gar.simulate_samples(horizon, q_agent, actions)
        match sampling_mode:
            case SamplingMode.RANDOM:
                new_violations, rot, samples = asp_transformer.call_clingo_check(
                    state, executed_actions, [], rot_count
                )
            case SamplingMode.STRATIFIED:
                check = SeqentialCheck(
                    gar._state,
                    config.sampling.epsilon,
                    config.sampling.delta,
                    config.horizon,
                )
                new_violations, rot, samples = check.check(executed_actions)

            case SamplingMode.MCTS:
                sim_state = SimulationState.from_state(gar._state, config.ctd)
                node = MCTSNode(
                    sim_state,
                    None,
                    sim_state.get_possible_actions_with_probabilities(
                        executed_actions[0]
                    ),
                )
                new_violations, rot, samples = node.check_MCTS(
                    executed_actions,
                    config.horizon,
                    config.sampling.epsilon,
                    config.sampling.max_visits,
                )
            case _:
                raise NotImplementedError(
                    f"Sampling mode {sampling_mode!r} is not wired in Main.py"
                )
        end_time_check = time.time()
        check_times.append(end_time_check - start_time_check)
        sampled_trajectories += samples

        if config.logLevel > 2:
            sim_state: SimulationState = SimulationState.from_state(
                gar._state, config.ctd
            )
            real_probability = sim_state.calculate_violation_probability_brute_force(
                executed_actions
            )
            lib.try_save(gar._state, real_probability, executed_actions)

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
                gen_count += 1
                fix_time_start = time.time()
                policy_fix = asp_transformer.call_clingo_generate(state, violations)
                fix_time_end = time.time()
                gen_times.append(fix_time_end - fix_time_start)

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
                        if method == Method.NEW_CACHE:
                            actions = policy_fix
                            action = actions.pop(0)
                        else:
                            action = policy_fix.pop(0)
                        break
                checkCount += 1
                start_time_check = time.time()
                match sampling_mode:
                    case SamplingMode.RANDOM:
                        new_violations, rot, samples = (
                            asp_transformer.call_clingo_check(
                                state, policy_fix, violations, rot_count
                            )
                        )
                    case SamplingMode.STRATIFIED:
                        new_violations, rot, samples = check.check(policy_fix)
                    case SamplingMode.MCTS:
                        sim_state = SimulationState.from_state(gar._state, config.ctd)
                        node = MCTSNode(
                            sim_state,
                            None,
                            sim_state.get_possible_actions_with_probabilities(
                                policy_fix[0]
                            ),
                        )
                        new_violations, rot, samples = node.check_MCTS(
                            policy_fix,
                            config.horizon,
                            config.sampling.epsilon,
                            config.sampling.max_visits,
                        )
                    case _:
                        raise NotImplementedError(
                            f"Sampling mode {sampling_mode!r} is not wired in Main.py"
                        )
                end_time_check = time.time()
                check_times.append(end_time_check - start_time_check)
                sampled_trajectories += samples
                if config.logLevel > 2:
                    # calculate real_probability via brute force
                    sim_state: SimulationState = SimulationState.from_state(
                        gar._state, config.ctd
                    )
                    real_probability = (
                        sim_state.calculate_violation_probability_brute_force(
                            executed_actions
                        )
                    )
                    lib.try_save(gar._state, real_probability, executed_actions)

                if rot:
                    # cache
                    if method == Method.NEW_CACHE:
                        actions = policy_fix
                        action = actions.pop(0)
                    else:
                        action = policy_fix.pop(0)
                    break
                else:
                    rejected_count += 1
                    for v in new_violations:
                        if v not in violations:
                            violations.append(v)
        end_time_gen = time.time()
        fix_times.append(end_time_gen - start_time_check)
        best_actions = q_agent.getBestActions(state)
        if action not in best_actions:
            intervention_count += 1
        rot_counts.append(rot_count)

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

    frogs_killed = state.dead_frogs.sum()
    ctd_triggered = state.stun_counter
    ctd_success = state.capt_frogs.sum()

    env.close()
    return (
        step,
        intervention_count,
        rot_counts,
        check_times,
        fix_times,
        full_times,
        gen_times,
        frogs_killed,
        ctd_success,
        ctd_triggered,
        checkCount,
        sampled_trajectories,
        gen_count,
        rejected_count,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indifference",
        type=float,
        default=0.1,
        help="specifies the radius of the indifference interval as percent of epsilon",
    )
    parser.add_argument(
        "--strata",
        type=int,
        default=32,
        help="specifies the amount of trajectories within one batch for sampling=1",
    )
    parser.add_argument(
        "--sampling", type=int, default=2, help="0 - random / 1 - stratified / 2 - MCTS"
    )
    parser.add_argument(
        "--method",
        type=int,
        default=1,
        help="0 - new framework / 1 - new framework w/ cache / 2 - old framework / 3 - RL",
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
        default=1,
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
    parser.add_argument("--epsilon", type=float, default=0.05, help="error tolerance")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="confidence delta (0.05 = 95% confidence)",
    )
    args = parser.parse_args()

    config = Config.from_args(args)

    random.seed(config.seed)
    seeds = [random.randint(0, 1000000) for _ in range(config.rounds)]

    all_step = 0
    all_samples = 0
    all_intervention_count = 0
    all_rot_counts = []
    all_check_times = []
    all_fix_times = []
    all_full_times = []
    all_gen_times = []
    all_frogs_killed = 0
    all_ctd_success = 0
    all_ctd_triggered = 0
    all_check_counts = 0
    all_gen_counts = 0
    all_rejected_counts = 0
    for i in range(config.rounds):
        (
            step,
            intervention_count,
            rot_counts,
            check_times,
            fix_times,
            full_times,
            gen_times,
            frogs_killed,
            ctd_success,
            ctd_triggered,
            checkCount,
            samples,
            gen_counts,
            rejected_counts,
        ) = run(config, seeds[i])
        all_step += step
        all_samples += samples
        all_intervention_count += intervention_count
        all_rot_counts.extend(rot_counts)
        all_check_times.extend(check_times)
        all_fix_times.extend(fix_times)
        all_full_times.extend(full_times)
        all_gen_times.extend(gen_times)
        all_frogs_killed += frogs_killed
        all_ctd_success += ctd_success
        all_ctd_triggered += ctd_triggered
        all_check_counts += checkCount
        all_gen_counts += gen_counts
        all_rejected_counts += rejected_counts

    if all_fix_times and config.method != Method.RL:
        avg_fix = sum(all_fix_times) / len(all_fix_times)
        max_fix = max(all_fix_times)

    if config.logLevel > 1:
        lib = StateLibrary(
            "sim_states",
            edges=custom_edges(
                0,
                0.005,
                0.01,
                0.15,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
            ),
            max_per_bucket=1000,
        )
        match config.sampling.mode:
            case SamplingMode.RANDOM:
                q_agent = GardenerQAgent()
                asp_transformer = ASPTransformer(q_agent, config.sampling)
                q_agent.stopLearning()
                q_agent.load_weights("weights.pkl")

        typeIerror = 0
        typeIIerror = 0

        entries = lib.load_range(0.0, 1.0)

        for entry in entries:
            match config.sampling.mode:
                case SamplingMode.RANDOM:
                    asp_transformer.build_static(entry.state, config.horizon)
                    asp_transformer.reset(config.ctd)
                    asp_transformer.build_dynamic_worlds(entry.state, [])
                    new_violations, rot, _ = asp_transformer.call_clingo_check(
                        entry.state, entry.actions, [], 0, []
                    )
                case SamplingMode.STRATIFIED:
                    check = SeqentialCheck(
                        entry.state,
                        config.sampling.epsilon,
                        config.sampling.indifference,
                        config.sampling.confidence,
                        config.horizon,
                        [],
                    )
                    new_violations, rot, _ = check.check(entry.actions)
                case SamplingMode.MCTS:
                    sim_state = entry.state
                    node = MCTSNode(
                        sim_state,
                        None,
                        sim_state.get_possible_actions_with_probabilities(
                            entry.actions[0]
                        ),
                    )
                    new_violations, rot, _ = node.check_MCTS(
                        entry.actions,
                        config.horizon,
                        config.sampling.epsilon,
                        config.sampling.max_visits,
                    )
                case _:
                    raise NotImplementedError(
                        f"Sampling mode {config.sampling.mode!r} is not wired in Main.py"
                    )
            if rot and entry.value > config.sampling.epsilon:
                typeIerror += 1
            if not rot and entry.value <= config.sampling.epsilon:
                typeIIerror += 1

        print()
        print(f"Average type I error: {typeIerror / len(entries):.4f}")
        print(f"Average type II error: {typeIIerror / len(entries):.4f}")

    if config.logLevel > 0:
        print()
        print(f"Average Steps: {all_step / config.rounds}")
        print(
            f"Percent of Solutions rejected: {(all_rejected_counts / all_gen_counts) * 100:.2f}\n"
        )
        print(
            f"Average Samples: {(all_samples / all_check_counts) / config.rounds:.2f}"
        )
        if config.method in (Method.NEW, Method.NEW_CACHE):
            if all_check_times:
                avg_check = sum(all_check_times) / len(all_check_times)
                max_check = max(all_check_times)
                print(
                    f"Average checking time: {avg_check:.4f}, Max checking time: {
                        max_check:.4f}"
                )
            if all_gen_times:
                avg_gen = sum(all_gen_times) / len(all_gen_times)
                max_gen = max(all_gen_times)
                print(
                    f"Average solution generation time: {
                        avg_gen:.4f}, Max solution generation time: {max_gen:.4f}"
                )
        if all_full_times and config.method != Method.RL:
            avg_full = sum(all_full_times) / len(all_full_times)
            max_full = max(all_full_times)
            print(
                f"Average time to compute step: {
                    avg_full:.4f}, Max time to compute step: {max_full:.4f}\n"
            )
        print(f"MTN1: {all_frogs_killed / config.rounds}")
        print(f"MTN2: {all_ctd_triggered / config.rounds}")
        print(f"CTD1: {(all_ctd_triggered - all_ctd_success) / config.rounds}")

    if config.logLevel < 1:
        if config.method == Method.RL:
            print(
                f"{all_step / config.rounds:.2f}, 0.00, 0.00, {
                    all_frogs_killed / config.rounds:.2f}, {
                    all_ctd_triggered / config.rounds:.2f}, {
                    (1 - (all_ctd_success / config.rounds))
                    * (all_ctd_triggered / config.rounds):.2f}"
            )
        else:
            print(
                f"{all_step / config.rounds:.2f}, {
                    (all_rejected_counts / all_gen_counts) * 100:.2f}, {
                    (all_samples / all_check_counts) / config.rounds:.2f}, {
                    sum(all_check_times) / len(all_check_times):.4f}, {
                    sum(all_gen_times) / len(all_gen_times):.4f}, {
                    sum(all_full_times) / len(all_full_times):.4f}, {
                    all_frogs_killed / config.rounds:.2f}, {
                    all_ctd_triggered / config.rounds:.2f}, {
                    (all_ctd_triggered - all_ctd_success) / config.rounds:.2f}"
            )
    # if config.ctd:
    #    print(f"ctd_success: {all_ctd_success / config.rounds}, ctd_triggered: {all_ctd_triggered / config.rounds}")
