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

    def __init__(self, state, epsilon, confidence, horizon):
        self.state = state
        self.solver: ASPSolver = ASPSolver(True, horizon)
        self.horizon = horizon
        self.epsilon = epsilon

        self.rejectionBoundary = 1 / confidence

        self.wealthAccepting = 1
        self.wealthRejecting = 1
        self.outcomeSum = 1
        self.solver.instantiate_level(self.state)
        self.solver.instantiate_state(self.state)
        self.solver.prepare_agent_movement()
        self.solver.prepare_frog_action(self.state)

    def check(self, actions):
        violations: list = []
        loop = 0
        rejectingConstant = (2 / self.epsilon) / ((1 / self.epsilon) + 1)
        acceptingConstant = 2 / self.epsilon
        self.solver.clear_agent_movement()
        self.solver.instantiate_agent_movement(actions)
        while loop < 1000:
            loop += 1
            loopViolations, outcome = self.sample_ASP()
            violations.extend(loopViolations)

            if outcome > 0:
                acceptingBet = (self.outcomeSum / acceptingConstant) / self.epsilon
                rejectingBet = (self.outcomeSum / rejectingConstant) / self.epsilon
            else:
                acceptingBet = (1 - (self.outcomeSum / acceptingConstant)) / (
                    1 - self.epsilon
                )
                rejectingBet = (1 - (self.outcomeSum / rejectingConstant)) / (
                    1 - self.epsilon
                )

            self.wealthAccepting *= acceptingBet
            self.wealthRejecting *= rejectingBet

            self.outcomeSum += outcome
            acceptingConstant += 1
            rejectingConstant += 1

            if self.wealthRejecting >= self.rejectionBoundary:
                # print(f"rejected at {loop}")
                self.reset()
                return violations, False, loop
            if self.wealthAccepting >= self.rejectionBoundary:
                # print(f"accepted at {loop}")
                self.reset()
                return violations, True, loop

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
