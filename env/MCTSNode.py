import math
import random
import numpy as np

from scipy.stats import norm

from env.simulation_state import SimulationState


class MCTSNode:
    C = 1
    __slots__ = ("action", "actions", "children", "state", "value", "visitCount")

    def __init__(self, state: SimulationState, action, actions):
        self.state: SimulationState = state
        self.action = action  # [agentAction, Frog1Action, Frog2Action]
        self.children = {}  # {action: MCTSNode}
        self.actions: dict = actions  # {action: probability}
        self.value = 0
        self.visitCount = 1

    def check_MCTS(self, agentActions, depth, confidence, indifference, maxVisit):
        z = norm.ppf((1 + confidence) / 2)
        runningProbability = 0.0
        violations = []  # TODO add
        sumSqared = 0.0
        minVisit = maxVisit
        while self.visitCount < minVisit and self.visitCount < maxVisit:
            nodeSequence = []
            nodeSequence.append(self)
            actionSequence = []
            weight = 1
            node: MCTSNode = self
            iteration = 0
            result = 0

            # selection process
            while iteration < depth:
                iteration += 1
                # calculate which node to choose now
                probabilities = node.calculate_probabilities_normalized()
                # print(f"probabilities in iteration {iteration}: {probabilities}")
                selectedAction, proposed_probability = node.select_action(probabilities)
                actionSequence.append(selectedAction)

                # update the weight for importance sampling
                weight = weight * (node.actions[selectedAction] / proposed_probability)
                # print(f"weight in iteration {iteration}: {weight}")

                if selectedAction in node.children:
                    # set new node
                    node = node.children.get(selectedAction)

                    # process new node
                    nodeSequence.append(node)
                    result += 1 if node.state.violation else 0
                else:
                    # expand
                    newState = SimulationState.apply_action(node.state, selectedAction)
                    newActions = newState.get_possible_actions_with_probabilities(
                        agentActions[iteration - 1]
                    )
                    newNode = MCTSNode(newState, selectedAction, newActions)
                    node.children[selectedAction] = newNode
                    # simulate
                    res, actions = newNode.simulateMC(
                        agentActions[iteration:], depth - iteration
                    )
                    actionSequence.extend(actions)
                    result += res
                    break

            # add world to violations
            if (
                result > 0 and actionSequence not in violations
            ):  # trade-off: a check each violation for minimizing the ASP Gen program
                violations.append(actionSequence)

            # print("actionSequence: ", actionSequence)
            # update visits
            for n in nodeSequence:
                n.visitCount += 1
                if result > 0:
                    n.value += 1

            # update running violation probability
            outcome = 1 if result > 0 else 0
            runningProbability = (1 / self.visitCount) * outcome * weight + (
                (self.visitCount - 1) / self.visitCount
            ) * runningProbability
            # print(
            #    f"runningProbability ({runningProbability}) = 1 / {self.visitCount} * {outcome} * {weight} * ({self.visitCount - 1}/{self.visitCount}) * {oldRunningProbability}"
            # )

            # update sum_sqared
            sumSqared += math.pow(outcome * weight, 2)

            # update variance
            variance = (
                1
                / (self.visitCount - 1)
                * (sumSqared - self.visitCount * math.pow(runningProbability, 2))
            )
            # update minimum visits
            if runningProbability > 0:
                minVisit = variance * (2 * z / (indifference * runningProbability)) ** 2
                # print(
                #    f"minVisit({minVisit}) = {variance} * (2* {z} / {indifference} * {runningProbability})**2"
                # )
            # print(f"current visit: {self.visitCount}, minVisit: {minVisit}")

        return violations, runningProbability

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

    def calculate_probabilities_normalized(self):
        values = {}
        for a in self.actions:
            if a in self.children:
                childNode = self.children[a]
                exploit = childNode.value / childNode.visitCount
                explore = MCTSNode.C * math.sqrt(
                    math.log(self.visitCount) / childNode.visitCount
                )
                values[a] = self.actions[a] * (exploit + explore)
            else:
                values[a] = (
                    self.actions[a] * MCTSNode.C * math.sqrt(math.log(self.visitCount))
                )

        total = sum(values.values())
        if total == 0:
            return {action: 1 / len(values) for action, value in values.items()}
        return {action: value / total for action, value in values.items()}

    def simulateMC(self, agentActions, depth):
        actions = []
        state = self.state
        result = 0

        for i in range(depth):
            result += 1 if state.violation else 0
            possible_actions = state.get_possible_actions_with_probabilities(
                agentActions[i]
            )
            a, _ = self.select_action(possible_actions)
            actions.append(a)
            state = SimulationState.apply_action(state, a)

        if result > 0:
            self.value += 1
        return result, actions
