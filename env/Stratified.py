import random

import numpy as np

from env.ASPSolver import ASPSolver
from env.simulation_state import SimulationState


class StratifiedCheck:
    __slots__ = (
        "altmean",
        "horizon",
        "lambdaAccepting",
        "lambdaRejecting",
        "nullmean",
        "rejectionBoundary",
        "solver",
        "state",
        "strata",
        "wealthAccepting",
        "wealthRejecting",
    )

    def __init__(self, state, epsilon, indifference, confidence, strata, horizon):
        self.state = state
        self.solver: ASPSolver = ASPSolver(True, horizon)
        self.strata = strata
        self.horizon = horizon

        self.nullmean = epsilon - (epsilon * indifference)
        self.altmean = epsilon + (epsilon * indifference)
        self.rejectionBoundary = 1 / (1 - confidence)

        self.wealthAccepting = 1
        self.wealthRejecting = 1
        self.lambdaAccepting = (self.altmean / self.nullmean) * (
            (1 - self.altmean) / (1 - self.nullmean)
        )
        self.lambdaRejecting = (self.nullmean / self.altmean) * (
            (1 - self.nullmean) / (1 - self.altmean)
        )
        self.solver.instantiate_level(self.state)
        self.solver.instantiate_state(self.state, self.strata)
        self.solver.prepare_agent_movement()
        self.solver.prepare_frog_movement(self.state, self.strata)

    def check(self, actions):
        violations: list = []
        loop = 0
        self.solver.clear_agent_movement()
        self.solver.instantiate_agent_movement(actions)
        while loop < 1000:
            loop += 1
            loopViolations, outcome = self.sample_ASP()
            violations.extend(loopViolations)
            self.update_martingale(outcome)

            if self.wealthRejecting >= self.rejectionBoundary:
                self.reset()
                return violations, False, loop * self.strata
            if self.wealthAccepting >= self.rejectionBoundary:
                self.reset()
                return violations, True, loop * self.strata

        self.reset()
        return violations, False, loop * self.strata

    def reset(self):
        self.wealthAccepting = 1
        self.wealthRejecting = 1

    def sample_ASP(self):
        self.solver.clear_frog_movement()
        self.solver.instantiate_frog_movement(
            SimulationState.from_state(self.state), self.strata
        )
        violations = self.solver.check()
        return violations, len(violations)

    def sample(self):
        states: list[SimulationState] = [self.state for i in range(self.strata)]
        trajectories: list = [[None] * self.horizon for _ in range(self.strata)]
        worlds = {k: False for k in range(self.strata)}
        cells = list(range(self.strata))

        for t in range(self.horizon):
            random.shuffle(cells)
            for index, world in enumerate(cells):
                possible_actions = states[
                    world
                ].get_possible_actions_with_probabilities(self.actions[t])

                selected_action = select_action(
                    possible_actions, (index + random.random()) / self.strata
                )
                trajectories[world][t] = selected_action
                states[world] = SimulationState.apply_action(
                    states[world], selected_action
                )
                if states[world].violation:
                    worlds[world] = True

        result = [tuple(trajectories[i]) for i in range(self.strata) if worlds[i]]
        return result, sum(worlds.values())

    def update_martingale(self, outcome):
        # wealthAccepting tracks the wealth of a gambler betting that the true mean is below the mean of the null
        # if he makes money, his intuition is correct and we can accept the null
        self.wealthAccepting *= 1 - (
            ((outcome / self.strata) - self.nullmean) * self.lambdaAccepting
        )
        # wealthRejecting tracks the wealth of a gambler betting that the true mean is above the mean of the alternative
        # if he makes money, his intuition is correct and we can reject the null
        self.wealthRejecting *= 1 + (
            ((outcome / self.strata) - self.altmean) * self.lambdaRejecting
        )


def select_action(possible_actions, probability):
    s = 0.0
    for a, p in possible_actions.items():
        s += p
        if s > probability:
            return a
    raise ValueError(
        "select_action could not select an action, the sum of probabilities did not add up to one"
    )
