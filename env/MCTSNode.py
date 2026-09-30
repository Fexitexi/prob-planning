import math
import random

from env.simulation_state import SimulationState


class MCTSNode:
    C = 1
    __slots__ = (
        "actions",
        "children",
        "horizon",
        "maxWeight",
        "state",
        "transitions",
        "value",
        "visitCount",
    )

    def __init__(self, state: SimulationState, action, horizon):
        self.state: SimulationState = state
        self.actions = [action]  # [agentAction]
        self.children = {}  # {action: MCTSNode}
        self.transitions: dict = state.get_possible_actions_with_probabilities(
            action, horizon
        )  # {action: probability}
        self.horizon = horizon
        self.value = 0
        self.visitCount = 1
        self.maxWeight = 1

    def check_MCTS(self, agentActions, epsilon, delta, indifference):
        violations = []
        quickAcceptWealth = 1
        quickWinPossible = True
        empMean = 0.0
        empSum2 = 0.0
        aGRAPAlambda = 0.0
        lambdaAccepting = 0.0
        lambdaRejecting = 0.0
        wealthAccepting = 1
        wealthRejecting = 1
        loop = 0
        while loop < 10000:
            loop += 1
            outcome, transitionSequence = self.sample(
                self.horizon, 1, [], agentActions, False
            )
            if len(transitionSequence) != self.horizon:
                print(f"ERROR: depth: {self.horizon}, length:{len(transitionSequence)}")

            if outcome > 0:
                violations.append(transitionSequence)
                quickWinPossible = False

            if quickWinPossible:
                quickAcceptWealth *= 1 - (outcome - epsilon)
                if quickAcceptWealth > 1 / delta:
                    return [], True, loop
            # update martingales
            acceptingDifference = outcome - epsilon
            rejectingDifference = outcome - epsilon

            acceptingReward = 1 - (
                lambdaAccepting * acceptingDifference
            )  # makes money if outcome == 0
            rejectingReward = 1 + (
                lambdaRejecting * rejectingDifference
            )  # makes money if outcome  == 1
            if acceptingReward <= 0:
                print(
                    f"ERROR {loop}: acceptingReward:{acceptingReward}, lambdaAccepting:{lambdaAccepting}, outcome:{outcome}, highestPossibleOutcome:{self.maxWeight}"
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
                # print(f"rejected after {loop} iterations, empMean: {empMean}")
                return violations, False, loop
            if wealthAccepting > 1 / delta:
                # print(f"accepted after {loop} iterations, empMean: {empMean}")
                return violations, True, loop

            # update mean and variance according to Welfords online algorithm
            oldmean = empMean
            empMean += (outcome - empMean) / loop
            empSum2 += (outcome - oldmean) * (outcome - empMean)

            aGRAPAlambda = (empMean - epsilon) / (
                (empSum2 / loop) + ((empMean - epsilon) ** 2)
            )

            lambdaAccepting = max(
                0, min(-aGRAPAlambda, 0.75 / (self.maxWeight - epsilon))
            )
            lambdaRejecting = max(0, min(aGRAPAlambda, 0.75 / epsilon))
            if loop % 10000 == 0:
                print(f"empMean: {empMean}")

        # print(f"indifferent after {loop} iterations")
        # print(f"wealthAccepting:{wealthAccepting}, wealthRejecting:{wealthRejecting}")
        return violations, False, loop

    def sample(self, depth, weight, transitionSequence, agentActions, violation):
        # basic return condition
        if depth == 0:
            if self.state.violation or violation:
                return weight, transitionSequence
            else:
                return 0, transitionSequence
        else:
            depth -= 1

        if agentActions[0] not in self.actions:
            self.transitions.update(
                self.state.get_possible_actions_with_probabilities(
                    agentActions[0], self.horizon
                )
            )
            self.actions.append(agentActions[0])

        # calculating everything
        probabilities = self.calculate_probabilities_normalized(agentActions[0])
        # print(f"probabilities in iteration {iteration}: {probabilities}")
        selectedTransition, proposed_probability = self.select_action(probabilities)
        transitionSequence.append(selectedTransition)

        # update the weight for importance sampling
        weight *= self.transitions[selectedTransition] / proposed_probability
        violation = violation or self.state.violation

        if selectedTransition in self.children:
            # set new node
            outcome, transitionSequence = self.children.get(selectedTransition).sample(
                depth, weight, transitionSequence, agentActions[1:], violation
            )
        else:
            # expand
            newState = SimulationState.apply_action(self.state, selectedTransition)
            newNode = MCTSNode(newState, agentActions[0], self.horizon)
            self.children[selectedTransition] = newNode
            # simulate
            outcome, mc_transitions = newNode.simulateMC(agentActions[1:], depth)
            transitionSequence.extend(mc_transitions)
            if outcome > 0 or violation:
                outcome = weight

        if outcome > 0:
            self.value += 1
        self.visitCount += 1

        self.updateMaxWeight(agentActions)

        return outcome, transitionSequence

    def get_child(self, state: SimulationState, action):
        for c in self.children.values():
            if state.__eq__(c.state) and action in c.actions:
                return c
        return MCTSNode(state, action, self.horizon)

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
            if a[0] != agentAction:
                continue
            if a in self.children:
                childNode = self.children[a]
                exploit = childNode.value / childNode.visitCount
                explore = MCTSNode.C * math.sqrt(
                    math.log(self.visitCount) / childNode.visitCount
                )
                values[a] = self.transitions[a] * (exploit + explore)
            else:
                values[a] = (
                    self.transitions[a]
                    * MCTSNode.C
                    * math.sqrt(math.log(self.visitCount))
                )

        total = sum(values.values())
        if total == 0:
            return {action: 1 / len(values) for action, value in values.items()}
        return {action: value / total for action, value in values.items()}

    def simulateMC(self, agentActions, depth):
        if self.state.violation:
            result = 1
        else:
            result = 0

        actions = []
        state = self.state
        for i in range(depth):
            possible_actions = state.get_possible_actions_with_probabilities(
                agentActions[i], self.horizon
            )
            a, _ = self.select_action(possible_actions)
            actions.append(a)
            state = SimulationState.apply_action(state, a)
            result += 1 if state.violation else 0

        if result > 0:
            self.value += 1
        return result, actions

    def updateMaxWeight(self, agentActions):
        maxWeight = 0
        probabilities = self.calculate_probabilities_normalized(agentActions[0])

        for a in self.transitions:
            if a[0] != agentActions[0]:
                continue
            if a in self.children:
                downstreamWeight = self.children[a].maxWeight
            else:
                downstreamWeight = 1
            branchWeight = self.transitions[a] / probabilities[a] * downstreamWeight
            maxWeight = max(maxWeight, branchWeight)

        self.maxWeight = maxWeight
