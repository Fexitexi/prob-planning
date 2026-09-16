import math
import random

from env.simulation_state import SimulationState


class MCTSNode:
    C = 1
    __slots__ = (
        "action",
        "actions",
        "children",
        "maxWeight",
        "state",
        "value",
        "visitCount",
    )

    def __init__(self, state: SimulationState, action, actions):
        self.state: SimulationState = state
        self.action = action  # [agentAction, Frog1Action, Frog2Action]
        self.children = {}  # {action: MCTSNode}
        self.actions: dict = actions  # {action: probability}
        self.value = 0
        self.visitCount = 1
        self.maxWeight = 1

    def check_MCTS(self, agentActions, depth, epsilon, maxVisit):
        violations = []
        lambdaAccepting = 1
        lambdaRejecting = 1
        wealthAccepting = 1
        wealthRejecting = 1
        acceptingSumSquared = 1
        rejectingSumSquared = 1
        highestOutcome = 0
        highestPossibleOutcome = 0
        loop = 0
        while loop < maxVisit:
            oldMaxWeight = self.maxWeight
            outcome, actionSequence = self.sample(depth, 1, [], agentActions)

            highestPossibleOutcome = max(highestPossibleOutcome, self.maxWeight)

            if outcome > 0:
                violations.append(actionSequence)
                highestOutcome = max(highestOutcome, outcome)

            # update martingales
            difference = outcome - epsilon
            acceptingReward = 1 - (lambdaAccepting * difference)
            rejectingReward = 1 + (lambdaRejecting * difference)
            if acceptingReward <= 0:
                print(
                    f"ERROR {loop}: acceptingReward:{acceptingReward}, lambdaAccepting:{lambdaAccepting}, outcome:{difference}, highestPossibleOutcome:{self.maxWeight}, prevhighestOutcome:{oldMaxWeight}"
                )
            if rejectingReward <= 0:
                print(
                    f"ERROR rejectingReward:{rejectingReward}, lambdaRejecting:{lambdaRejecting}, outcome:{outcome}, highestPossibleOutcome:{self.maxWeight}, prevhighestOutcome:{oldMaxWeight}"
                )
            wealthAccepting *= acceptingReward
            wealthRejecting *= rejectingReward

            if wealthRejecting > 1 / epsilon:
                # print(f"rejected after {loop} iterations")
                return violations, False, loop
            if wealthAccepting > 1 / epsilon:
                # print(f"accepted after {loop} iterations")
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
                min(lambdaAccepting, 0.99 / (self.maxWeight - epsilon)),
                -1 / (1 - epsilon),
            )
            lambdaRejecting = min(
                max(lambdaRejecting, -0.99 / (self.maxWeight - epsilon)),
                1 / epsilon,
            )

            loop += 1

        # print(f"indifferent after {loop} iterations")
        # print(f"wealthAccepting:{wealthAccepting}, wealthRejecting:{wealthRejecting}")
        # print(
        # f"highest outcome:{highestOutcome}, highestPossibleOutcome:{highestPossibleOutcome}"
        # )
        return violations, False, maxVisit

    def sample(self, depth, weight, actionSequence, agentActions):
        # basic return condition
        if depth == 0:
            return self.state.violation * weight, actionSequence
        else:
            depth -= 1

        # calculating everything
        probabilities = self.calculate_probabilities_normalized()
        # print(f"probabilities in iteration {iteration}: {probabilities}")
        selectedAction, proposed_probability = self.select_action(probabilities)
        actionSequence.append(selectedAction)

        # update the weight for importance sampling
        weight *= self.actions[selectedAction] / proposed_probability

        if selectedAction in self.children:
            # set new node
            outcome, actionSequence = self.children.get(selectedAction).sample(
                depth, weight, actionSequence, agentActions
            )
        else:
            # expand
            newState = SimulationState.apply_action(self.state, selectedAction)
            newActions = newState.get_possible_actions_with_probabilities(
                agentActions[-depth]
            )
            newNode = MCTSNode(newState, selectedAction, newActions)
            self.children[selectedAction] = newNode
            # simulate
            outcome, actionSequence = newNode.simulateMC(agentActions[-depth:], depth)
            if outcome > 0:
                outcome = weight

        if outcome > 0:
            self.value += 1
        self.visitCount += 1

        self.updateMaxWeight()

        return outcome, actionSequence

    def generate_fix(self):
        pass

    def get_child(self, action):
        return self.children.get(action)

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
        if self.state.violation:
            result = 1
        else:
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

    def updateMaxWeight(self):
        probabilities = self.calculate_probabilities_normalized()

        maxWeight = 0
        for a in self.actions:
            if a in self.children:
                downstreamWeight = self.children[a].maxWeight
            else:
                downstreamWeight = 1
            branchWeight = self.actions[a] / probabilities[a] * downstreamWeight
            maxWeight = max(maxWeight, branchWeight)

        self.maxWeight = maxWeight
