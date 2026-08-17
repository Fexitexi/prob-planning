import argparse
import random
import time

import gymnasium as gym

from config import Config, Method, SamplingMode
from env.ASPTransformer import ASPTransformer
from env.GardenerQAgent import GardenerQAgent
from env.MCTSNode import MCTSNode
from env.old.ClingoHelper import ClingoHelperOld
from env.simulation_state import SimulationState
from env.state import ObservationState

gym.envs.registration.register(
    id="GardenerEnv-v0",
    entry_point="env.GardenerEnv:GardenerEnv",
)


def run(config: Config, seed: int | None = None):
    global env
    env = gym.make("GardenerEnv-v0", size=config.size)
    gar = env.unwrapped

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

    state = ObservationState.from_obs(obs)
    asp_transformer.build_static(state, config.horizon)
    lake_full = state.lakes_full
    actions = []
    step = 0
    full_times = []
    check_times = []
    fix_times = []
    gen_count = 0
    fixing_count = 0
    ctd_success = 0
    ctd_failure = 0
    frogs_killed = 0
    rot_counts = []
    checkCount = 0
    typeI_errors = 0
    typeII_errors = 0
    sips = {}
    intervention_count = 0

    # old
    clingoHelperOld = ClingoHelperOld(state, q_agent, config.horizon + 1, 10)
    clingoHelperOld.setup()

    # cache frequently accessed config values for the hot loop
    method = config.method
    sampling_mode = config.sampling.mode
    horizon = config.horizon
    use_new_framework = method in (Method.NEW, Method.NEW_CACHE)

    while not done:
        start_full_time = time.time()
        step += 1
        rot_count = 1
        if use_new_framework:
            start_time_check = time.time()
            asp_transformer.reset(config.ctd)
            asp_transformer.build_dynamic_worlds(state, sips)
            _, executed_actions = gar.simulate_samples(horizon, q_agent, actions)
            match sampling_mode:
                case SamplingMode.RANDOM:
                    new_violations, rot = asp_transformer.call_clingo_check(
                        state, executed_actions, [], rot_count, sips
                    )
                case SamplingMode.STRATIFIED:
                    new_violations, rot = asp_transformer.call_stratified_check(
                        state, executed_actions, sips
                    )
                case SamplingMode.MCTS:
                    sim_state = SimulationState.from_state(gar._state, config.ctd)
                    node = MCTSNode(
                        sim_state,
                        None,
                        sim_state.get_possible_actions_with_probabilities(
                            executed_actions[0]
                        ),
                    )
                    new_violations, rot = node.check_MCTS(
                        executed_actions,
                        config.horizon,
                        config.sampling.confidence,
                        config.sampling.indifference,
                        config.sampling.max_visits,
                    )
                    rot = rot <= config.sampling.delta
                case _:
                    raise NotImplementedError(
                        f"Sampling mode {sampling_mode!r} is not wired in Main.py"
                    )
            end_time_check = time.time()
            check_times.append(end_time_check - start_time_check)

            if config.logLevel > 2:
                checkCount += 1
                # calculate typeI and II errors via brute_force
                sim_state: SimulationState = SimulationState.from_state(
                    gar._state, config.ctd
                )
                real_probability = (
                    sim_state.calculate_violation_probability_brute_force(
                        executed_actions
                    )
                )
                above_risk_threshold = real_probability > config.sampling.epsilon

                if rot and above_risk_threshold:
                    # type I error has been made, accepting null hypothesis though incorrect
                    typeI_errors += 1

                if not rot and not above_risk_threshold:
                    # type II error has been made, accepting alternative hypothesis though incorrect
                    typeII_errors += 1

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
                    policy_fix = asp_transformer.call_clingo_generate(state, violations)
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
                    match sampling_mode:
                        case SamplingMode.RANDOM:
                            new_violations, rot = asp_transformer.call_clingo_check(
                                state, policy_fix, violations, rot_count, sips
                            )
                        case SamplingMode.STRATIFIED:
                            new_violations, rot = asp_transformer.call_stratified_check(
                                state, policy_fix, sips
                            )
                        case SamplingMode.MCTS:
                            sim_state = SimulationState.from_state(
                                gar._state, config.ctd
                            )
                            node = MCTSNode(
                                sim_state,
                                None,
                                sim_state.get_possible_actions_with_probabilities(
                                    policy_fix[0]
                                ),
                            )
                            new_violations, rot = node.check_MCTS(
                                policy_fix,
                                config.horizon,
                                config.sampling.confidence,
                                config.sampling.indifference,
                                config.sampling.max_visits,
                            )
                            rot = rot <= config.sampling.delta
                        case _:
                            raise NotImplementedError(
                                f"Sampling mode {sampling_mode!r} is not wired in Main.py"
                            )
                    if config.logLevel > 2:
                        checkCount += 1
                        # calculate typeI and II errors via brute_force
                        sim_state: SimulationState = SimulationState.from_state(
                            gar._state, config.ctd
                        )
                        real_probability = (
                            sim_state.calculate_violation_probability_brute_force(
                                policy_fix
                            )
                        )
                        above_risk_threshold = (
                            real_probability > config.sampling.epsilon
                        )

                        if rot and above_risk_threshold:
                            # type I error has been made, accepting null hypothesis though incorrect
                            typeI_errors += 1

                        if not rot and not above_risk_threshold:
                            # type II error has been made, accepting alternative hypothesis though incorrect
                            typeII_errors += 1

                    if rot:
                        # cache
                        if method == Method.NEW_CACHE:
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
        elif method == Method.OLD:
            # OLD METHOD EXECUTION
            start_time_gen = time.time()
            best_actions = q_agent.getBestActions(state)
            action = clingoHelperOld.get_action(state)
            if action not in best_actions:
                intervention_count += 1
            fix_times.append(time.time() - start_time_gen)
        elif method == Method.RL:
            action = q_agent.getAction(state)

        obs, reward, terminated, truncated, info = env.step(action)
        state = ObservationState.from_obs(obs)

        # add this to the environment
        remove = []
        for s in sips:
            if (
                state.agent[0] == state.frogs[sips[s][0]][0]
                and state.agent[1] == state.frogs[sips[s][0]][1]
            ):
                remove.append(s)
                ctd_success += 1
                msg = "CDT SUCCESS!"
                gar._state.capt_frogs[sips[s][0]] = True
                # print(f"\033[31m{msg}\033[0m")
            elif sips[s][1] == 0:
                remove.append(s)
                ctd_failure += 1
            sips[s][1] -= 1
        for r in remove:
            sips.pop(r)
        new_lake_full = state.lakes_full
        for lake in range(len(lake_full)):
            if lake_full[lake] and not new_lake_full[lake]:
                # print(f"Lake {lake} is now empty at step {step}.")
                for f, (c, r) in enumerate(state.frogs):
                    prox = False
                    if not state.dead_frogs[f]:
                        if (
                            abs(state.lakes[lake][0] - c)
                            + abs(state.lakes[lake][1] - r)
                            == 1
                        ):
                            prox = True
                        elif (
                            abs(state.lakes[lake][0] - c)
                            + abs(state.lakes[lake][1] - r)
                            == 2
                            and abs(state.lakes[lake][0] - c) == 1
                        ):
                            prox = True
                        elif (
                            abs(state.lakes[lake][0] - c)
                            + abs(state.lakes[lake][1] - r)
                            == 2
                            and abs(state.lakes[lake][1] - r) == 1
                        ):
                            prox = True
                    if prox:
                        # print(
                        #    f"Frog {f} is at lake {lake}, which is at coordinates {c}, {r}.")
                        sips[lake] = [f, 4, c, r]
        lake_full = new_lake_full.copy()

        # for visualization purposes
        # sleep = max(0, 0.1)
        # time.sleep(sleep)
        if config.render:
            env.render()
        done = terminated or truncated
        end_full_time = time.time()
        full_times.append(end_full_time - start_full_time)

    ctd_triggered = ctd_success + ctd_failure
    if ctd_triggered > 0:
        ctd_success = ctd_success / ctd_triggered
    else:
        ctd_success = 1

    frogs_killed = state.dead_frogs.sum()
    if config.ctd:
        frogs_killed -= ctd_success * frogs_killed

    env.close()
    return (
        step,
        intervention_count,
        rot_counts,
        check_times,
        fix_times,
        full_times,
        frogs_killed,
        ctd_success,
        ctd_triggered,
        checkCount,
        typeI_errors,
        typeII_errors,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indifference",
        type=float,
        default=0.005,
        help="specifies the radius of the indifference interval",
    )
    parser.add_argument(
        "--strata",
        type=int,
        default=1,
        help="specifies the amount of trajectories within one batch for sampling=1",
    )
    parser.add_argument(
        "--sampling", type=int, default=0, help="0 - random / 1 - stratified / 2 - MCTS"
    )
    parser.add_argument(
        "--method",
        type=int,
        default=0,
        help="0 - new framework / 1 - new framework w/ cache / 2 - old framework / 3 - RL",
    )
    parser.add_argument("--horizon", type=int, default=3, help="Horizon")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--ctd", type=int, default=0, choices=[0, 1], help="0 - no ctd / 1 - ctd"
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
    all_intervention_count = 0
    all_rot_counts = []
    all_check_times = []
    all_fix_times = []
    all_full_times = []
    all_frogs_killed = 0
    all_ctd_success = 0
    all_ctd_triggered = 0
    all_check_counts = 0
    all_typeI_errors = 0
    all_typeII_errors = 0
    for i in range(config.rounds):
        (
            step,
            intervention_count,
            rot_counts,
            check_times,
            fix_times,
            full_times,
            frogs_killed,
            ctd_success,
            ctd_triggered,
            checkCount,
            typeI_errors,
            typeII_errors,
        ) = run(config, seeds[i])
        all_step += step
        all_intervention_count += intervention_count
        all_rot_counts.extend(rot_counts)
        all_check_times.extend(check_times)
        all_fix_times.extend(fix_times)
        all_full_times.extend(full_times)
        all_frogs_killed += frogs_killed
        all_ctd_success += ctd_success
        all_ctd_triggered += ctd_triggered
        all_check_counts += checkCount
        all_typeI_errors += typeI_errors
        all_typeII_errors += typeII_errors

    if config.logLevel > 2:
        if config.method in (Method.NEW, Method.NEW_CACHE):
            avg_typeI_error = all_typeI_errors / all_check_counts
            avg_typeII_error = all_typeII_errors / all_check_counts
            print(
                f"Average Type I error: {avg_typeI_error:.2f}, Average Type II error: {avg_typeII_error:.2f}"
            )
    if config.logLevel > 0:
        if config.method in (Method.NEW, Method.NEW_CACHE):
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
                print(
                    f"Average rot_checks: {avg_rot:.2f}, Max rot_checks: {
                        max_rot:.2f}, Neg rot_checks: {count_neg}"
                )
            if all_check_times:
                avg_check = sum(all_check_times) / len(all_check_times)
                max_check = max(all_check_times)
                print(
                    f"Average checking time: {avg_check:.4f}, Max checking time: {
                        max_check:.4f}"
                )
        if all_fix_times and config.method != Method.RL:
            avg_fix = sum(all_fix_times) / len(all_fix_times)
            max_fix = max(all_fix_times)
            print(f"Average fixing time: {avg_fix:.4f}, Max fixing time: {max_fix:.4f}")
        if all_full_times and config.method != Method.RL:
            avg_full = sum(all_full_times) / len(all_full_times)
            max_full = max(all_full_times)
            print(
                f"Average time to compute step: {
                    avg_full:.4f}, Max time to compute step: {max_full:.4f}"
            )
        print(
            f"Steps: {all_step / config.rounds}, Interventions: {
                all_intervention_count / config.rounds
            }"
        )
        print(f"Frogs killed: {all_frogs_killed / config.rounds}")

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
                all_intervention_count / config.rounds:.2f}, {avg_fix * 1000:.2f}, {
                all_frogs_killed / config.rounds:.2f}, {
                all_ctd_triggered / config.rounds:.2f}, {
                (1 - (all_ctd_success / config.rounds))
                * (all_ctd_triggered / config.rounds):.2f}"
        )
    # if config.ctd:
    #    print(f"ctd_success: {all_ctd_success / config.rounds}, ctd_triggered: {all_ctd_triggered / config.rounds}")
