import math
import random

import numpy as np

from env.ASPSolver import ASPSolver
from env.simulation_state import SimulationState


class SeqentialCheck:
    __slots__ = (
        "epsilon",
        "horizon",
        "outcomeSum",
        "rejectionBoundary",
        "solver",
        "state",
        "wealthAccepting",
        "wealthRejecting",
    )

    def __init__(self, state, epsilon, horizon):
        self.state = state
        self.solver: ASPSolver = ASPSolver(True, horizon)
        self.horizon = horizon
        self.epsilon = epsilon

        self.rejectionBoundary = 1 / epsilon

        self.wealthAccepting = 1
        self.wealthRejecting = 1
        self.outcomeSum = 1
        self.solver.instantiate_level(self.state)
        self.solver.instantiate_state(self.state)
        self.solver.prepare_agent_movement()
        self.solver.prepare_frog_action(self.state)

    def check(self, actions, maxSamples):
        violations: list = []
        loop = 0
        lambdaAccepting = 1
        lambdaRejecting = 1
        acceptingSumSquared = 1
        rejectingSumSquared = 1
        self.solver.clear_agent_movement()
        self.solver.instantiate_agent_movement(actions)
        while loop < maxSamples:
            loop += 1
            loopViolations, outcome = self.sample_ASP()
            violations.extend(loopViolations)

            difference = outcome - self.epsilon
            acceptingReward = 1 - (lambdaAccepting * difference)
            rejectingReward = 1 + (lambdaRejecting * difference)

            self.wealthAccepting *= acceptingReward
            self.wealthRejecting *= rejectingReward

            if self.wealthRejecting >= self.rejectionBoundary:
                self.reset()
                return violations, False, loop
            if self.wealthAccepting >= self.rejectionBoundary:
                self.reset()
                return violations, True, loop

            # calculate lambdas for next iteration via ONS from Waudby-Smith
            acceptingSumSquared += acceptingReward**2
            rejectingSumSquared += rejectingReward**2

            lambdaAccepting -= (2 * difference / acceptingReward) / (
                (2 - math.log(3)) * acceptingSumSquared
            )
            lambdaRejecting -= (2 * difference / rejectingReward) / (
                (2 - math.log(3)) * rejectingSumSquared
            )

            lambdaAccepting = max(
                min(lambdaAccepting, 0.99 / (1 - self.epsilon)),
                -1 / (1 - self.epsilon),
            )
            lambdaRejecting = min(
                max(lambdaRejecting, -0.99 / (1 - self.epsilon)),
                1 / self.epsilon,
            )

        self.reset()
        return violations, False, loop

    def reset(self):
        self.wealthAccepting = 1
        self.wealthRejecting = 1
        self.outcomeSum = 1

    def sample_ASP(self):
        self.solver.clear_frog_movement()
        self.solver.instantiate_frog_actions(SimulationState.from_state(self.state))
        violations, outcome = self.solver.check()
        return violations, outcome


def select_action(possible_actions, probability):
    s = 0.0
    for a, p in possible_actions.items():
        s += p
        if s > probability:
            return a
    raise ValueError(
        "select_action could not select an action, the sum of probabilities did not add up to one"
    )
