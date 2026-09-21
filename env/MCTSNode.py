import math
import random

from env.simulation_state import SimulationState


class MCTSNode:
    C = 1
    __slots__ = (
        "action",
        "children",
        "maxWeight",
        "state",
        "transitions",
        "value",
        "visitCount",
    )

    def __init__(self, state: SimulationState, action):
        self.state: SimulationState = state
        self.action = action  # [agentAction]
        self.children = {}  # {action: MCTSNode}
        self.transitions: dict = state.get_possible_actions_with_probabilities(
            action
        )  # {action: probability}
        self.value = 0
        self.visitCount = 1
        self.maxWeight = 1

    def check_MCTS(self, agentActions, depth, epsilon, maxVisit):
        violations = []
        lambdaAccepting = max(
            min(1, 0.99 / (self.maxWeight - epsilon)),
            -1 / (1 - epsilon),
        )
        lambdaRejecting = min(
            max(1, -0.99 / (self.maxWeight - epsilon)),
            1 / epsilon,
        )
        wealthAccepting = 1
        wealthRejecting = 1
        acceptingSumSquared = 1
        rejectingSumSquared = 1
        highestPossibleOutcome = 0
        loop = 0
        while loop < maxVisit:
            oldMaxWeight = self.maxWeight
            outcome, transitionSequence = self.sample(depth, 1, [], agentActions, False)
            if len(transitionSequence) != depth:
                print(f"ERROR: depth: {depth}, length:{len(transitionSequence)}")

            highestPossibleOutcome = max(highestPossibleOutcome, self.maxWeight)

            if outcome > 0:
                violations.append(transitionSequence)

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
        return violations, False, maxVisit

    def sample(self, depth, weight, transitionSequence, agentActions, violation):
        # basic return condition
        if depth == 0:
            if self.state.violation or violation:
                return weight, transitionSequence
            else:
                return 0, transitionSequence
        else:
            depth -= 1

        # calculating everything
        probabilities = self.calculate_probabilities_normalized()
        # print(f"probabilities in iteration {iteration}: {probabilities}")
        selectedTransition, proposed_probability = self.select_action(probabilities)
        transitionSequence.append(selectedTransition)

        # update the weight for importance sampling
        weight *= self.transitions[selectedTransition] / proposed_probability
        violation = violation or self.state.violation

        if selectedTransition in self.children:
            # set new node
            outcome, transitionSequence = self.children.get(selectedTransition).sample(
                depth, weight, transitionSequence, agentActions, violation
            )
        else:
            # expand
            newState = SimulationState.apply_action(self.state, selectedTransition)
            newNode = MCTSNode(newState, agentActions[-depth])
            self.children[selectedTransition] = newNode
            # simulate
            outcome, mc_transitions = newNode.simulateMC(agentActions[-depth:], depth)
            transitionSequence.extend(mc_transitions)
            if outcome > 0:
                outcome = weight

        if outcome > 0:
            self.value += 1
        self.visitCount += 1

        self.updateMaxWeight()

        return outcome, transitionSequence

    def get_child(self, state: SimulationState, action):
        for c in self.children.values():
            if state.__eq__(c.state) and action == c.action:
                return c
        return MCTSNode(state, action)

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
        for a in self.transitions:
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
        actions = []
        state = self.state
        if self.state.violation:
            result = 1
        else:
            result = 0

        for i in range(depth):
            possible_actions = state.get_possible_actions_with_probabilities(
                agentActions[i]
            )
            a, _ = self.select_action(possible_actions)
            actions.append(a)
            state = SimulationState.apply_action(state, a)
            result += 1 if state.violation else 0

        if result > 0:
            self.value += 1
        return result, actions

    def updateMaxWeight(self):
        probabilities = self.calculate_probabilities_normalized()

        maxWeight = 0
        for a in self.transitions:
            if a in self.children:
                downstreamWeight = self.children[a].maxWeight
            else:
                downstreamWeight = 1
            branchWeight = self.transitions[a] / probabilities[a] * downstreamWeight
            maxWeight = max(maxWeight, branchWeight)

        self.maxWeight = maxWeight
