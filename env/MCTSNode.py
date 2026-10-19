import math
import random

from env.simulation_state import SimulationState


class MCTSNode:
    C = 1
    __slots__ = (
        "children",
        "horizon",
        "maxWeight",
        "state",
        "transitions",
        "value",
        "visitCount",
    )

    def __init__(self, state: SimulationState, horizon):
        self.state: SimulationState = state
        self.children = {}  # {action: MCTSNode}
        self.transitions: dict = None
        self.horizon = horizon
        self.value = 0
        self.visitCount = 1
        self.maxWeight = 1

    def check_MCTS(self, agentActions, epsilon, delta, indifference):
        violations = []
        empMean = 0.0
        empSum2 = 0.0
        aGRAPAlambda = 0.0
        lambdaAccepting = 0.0
        lambdaRejecting = 0.0
        wealthAccepting = 1
        wealthRejecting = 1
        loop = 0
        maxmax = 0
        while True:
            loop += 1
            outcome, transitionSequence = self.sample(
                self.horizon, 1, [], agentActions, False
            )
            maxmax = max(maxmax, self.maxWeight)
            if len(transitionSequence) != self.horizon:
                print(f"ERROR: depth: {self.horizon}, length:{len(transitionSequence)}")

            if outcome > 0:
                violations.append(transitionSequence)

            # update martingales
            acceptingDifference = outcome - (epsilon * (1 + indifference))
            rejectingDifference = outcome - (epsilon * (1 - indifference))

            acceptingReward = 1 - (
                lambdaAccepting * acceptingDifference
            )  # makes money if outcome == 0
            rejectingReward = 1 + (
                lambdaRejecting * rejectingDifference
            )  # makes money if outcome  == 1
            if acceptingReward <= 0:
                print(
                    f"ERROR {loop}: acceptingReward:{acceptingReward}, aGRAPAlambda: {aGRAPAlambda}, intermediate:{min(-aGRAPAlambda, 0.75 / (self.maxWeight - epsilon))}, lambdaAccepting:{lambdaAccepting}, outcome:{outcome}, highestPossibleOutcome:{self.maxWeight}"
                )
            if rejectingReward <= 0:
                print(
                    f"ERROR rejectingReward:{rejectingReward}, lambdaRejecting:{lambdaRejecting}, outcome:{outcome}, highestPossibleOutcome:{self.maxWeight}"
                )
            if acceptingReward > 1 and rejectingReward > 1:
                print(
                    f"outcome:{outcome}, lambdaRejecting:{lambdaRejecting}, lambdaAccepting:{lambdaAccepting}"
                )

            wealthAccepting *= acceptingReward
            wealthRejecting *= rejectingReward

            if wealthRejecting > 1 / delta:
                return violations, False, loop
            if wealthAccepting > 1 / delta:
                return violations, True, loop

            # update mean and variance according to Welfords online algorithm
            oldmean = empMean
            empMean += (outcome - empMean) / loop
            empSum2 += (outcome - oldmean) * (outcome - empMean)

            aGRAPAlambda = (empMean - epsilon) / (
                (empSum2 / loop) + ((empMean - epsilon) ** 2)
            )

            lambdaAccepting = max(
                0,
                min(
                    -aGRAPAlambda,
                    0.75 / (self.maxWeight - epsilon * (1 + indifference)),
                ),
            )
            lambdaRejecting = max(
                0, min(aGRAPAlambda, 0.75 / epsilon * (1 - indifference))
            )
        # print(f"wealthAccepting:{wealthAccepting}, wealthRejecting:{wealthRejecting}")
        return violations, False, loop

    def sample(self, depth, weight, transitionSequence, agentActions, violation):
        # basic return condition
        if depth == 0:
            if self.state.arr[-1] == 1 or violation:
                return weight, transitionSequence
            else:
                return 0, transitionSequence
        else:
            depth -= 1

        if self.transitions is None:
            self.transitions = self.state.get_possible_actions_with_probabilities(
                self.state.arr, agentActions[0], self.horizon
            )
        # calculating everything
        probabilities = self.calculate_probabilities_normalized(agentActions[0])
        # print(f"probabilities in iteration {iteration}: {probabilities}")
        selectedTransition, proposed_probability = self.select_action(probabilities)
        transitionSequence.append(selectedTransition)

        # update the weight for importance sampling
        weight *= self.transitions[selectedTransition] / proposed_probability
        violation = violation or self.state.arr[-1] == 1

        if selectedTransition in self.children:
            # set new node
            outcome, transitionSequence = self.children.get(selectedTransition).sample(
                depth, weight, transitionSequence, agentActions[1:], violation
            )
        else:
            # expand
            newArr = self.state.apply_action(self.state.arr, selectedTransition)
            newState = SimulationState(newArr, self.state.cache, self.state.env)
            newNode = MCTSNode(newState, self.horizon)
            self.children[selectedTransition] = newNode
            # simulate
            outcome, mc_transitions = newNode.simulateMC(agentActions[1:], depth)
            transitionSequence.extend(mc_transitions)
            if outcome > 0 or violation:
                outcome = weight

        if outcome > 0:
            self.value += 1
        self.visitCount += 1

        self.updateMaxWeight(self.calculate_probabilities_normalized(agentActions[0]))

        return outcome, transitionSequence

    def get_child(self, state: SimulationState):
        for c in self.children.values():
            if state.__eq__(c.state):
                return c
        return MCTSNode(state, self.horizon)

    def select_action(self, probabilities):
        s = 0.0
        rnd = random.random()
        for a, p in probabilities.items():
            s += p
            if s > rnd:
                return a, p
        raise ValueError(
            "select_action could not select an action, the sum of probabilities did not add up to one"
        )

    def calculate_probabilities_normalized(self, agentAction):
        values = {}
        for a in self.transitions:
            if a in self.children:
                childNode = self.children[a]
                exploit = childNode.value / childNode.visitCount
                explore = MCTSNode.C * math.sqrt(
                    math.log(self.visitCount) / childNode.visitCount
                )
                values[a] = (0.2 * self.transitions[a]) + (
                    0.8 * self.transitions[a] * (exploit + explore)
                )
            else:
                values[a] = (
                    self.transitions[a]
                    * MCTSNode.C
                    * math.sqrt(math.log(self.visitCount))
                )

        total = sum(values.values())
        if total == 0:
            return {
                action: self.transitions[action] for action, value in values.items()
            }
        return {action: value / total for action, value in values.items()}

    def simulateMC(self, agentActions, depth):
        result = self.state.arr[-1]

        actions = []
        arr = self.state.arr
        for i in range(depth):
            possible_actions = self.state.get_possible_actions_with_probabilities(
                arr, agentActions[i], self.horizon
            )
            a, _ = self.select_action(possible_actions)
            actions.append(a)
            arr = self.state.apply_action(arr, a)
            result += arr[-1]

        if result > 0:
            self.value += 1
        return result, actions

    def updateMaxWeight(self, probabilities):
        maxWeight = 0

        for a in self.transitions:
            if a in self.children:
                downstreamWeight = self.children[a].maxWeight
            else:
                downstreamWeight = 1
            branchWeight = (self.transitions[a] / probabilities[a]) * downstreamWeight
            maxWeight = max(maxWeight, branchWeight)

        self.maxWeight = maxWeight
