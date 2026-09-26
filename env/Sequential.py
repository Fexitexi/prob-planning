from env.ASPSolver import ASPSolver
from env.simulation_state import SimulationState


class SeqentialCheck:
    __slots__ = (
        "epsilon",
        "horizon",
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
        self.solver.instantiate_level(self.state)
        self.solver.instantiate_state(self.state)
        self.solver.prepare_agent_movement()
        self.solver.prepare_frog_action(self.state)

    def check(self, actions, maxSamples):
        violations: list = []
        loop = 0
        empMean = 0.0
        empSum2 = 0.0
        aGRAPAlambda = 0.0
        lambdaAccepting = 0.0
        lambdaRejecting = 0.0
        self.solver.clear_agent_movement()
        self.solver.instantiate_agent_movement(actions)
        while loop < maxSamples:
            loop += 1
            loopViolations, outcome = self.sample_ASP()
            if outcome == 1:
                violations.append(loopViolations)

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
            # update mean and variance according to Welfords online algorithm
            oldmean = empMean
            empMean += (outcome - empMean) / loop
            empSum2 += (outcome - oldmean) * (outcome - empMean)

            aGRAPAlambda = (empMean - self.epsilon) / (
                (empSum2 / loop) + ((empMean - self.epsilon) ** 2)
            )

            lambdaAccepting = max(0, min(-aGRAPAlambda, 0.75 / (1 - self.epsilon)))
            lambdaRejecting = max(0, min(aGRAPAlambda, 0.75 / self.epsilon))

        self.reset()
        return violations, False, loop

    def reset(self):
        self.wealthAccepting = 1
        self.wealthRejecting = 1

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
