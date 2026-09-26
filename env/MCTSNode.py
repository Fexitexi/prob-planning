import math
import random

from env.simulation_state import SimulationState


class MCTSNode:
    C = 1
    __slots__ = (
        "actions",
        "children",
        "maxWeight",
        "state",
        "transitions",
        "value",
        "visitCount",
    )

    def __init__(self, state: SimulationState, action):
        self.state: SimulationState = state
        self.actions = [action]  # [agentAction]
        self.children = {}  # {action: MCTSNode}
        self.transitions: dict = state.get_possible_actions_with_probabilities(
            action
        )  # {action: probability}
        self.value = 0
        self.visitCount = 1
        self.maxWeight = 1

    def check_MCTS(self, agentActions, depth, epsilon, maxVisit):
        violations = []
        outcomeSum = 0
        outcomeCOunter = 0
        empMean = 0.0
        empSum2 = 0.0
        aGRAPAlambda = 0.0
        onsLambda = 0.0
        lambdaAccepting = 0.0
        lambdaRejecting = 0.0
        wealthAccepting = 1
        wealthRejecting = 1
        sumSquared = 1
        loop = 0
        while loop < maxVisit:
            loop += 1
            outcome, transitionSequence = self.sample(depth, 1, [], agentActions, False)
            if len(transitionSequence) != depth:
                print(f"ERROR: depth: {depth}, length:{len(transitionSequence)}")

            if outcome > 0:
                outcomeCOunter += 1
                outcomeSum += outcome
                violations.append(transitionSequence)

            # update martingales
            difference = outcome - epsilon
            # z = difference / (1 - (difference * onsLambda))

            acceptingReward = 1 - (
                lambdaAccepting * difference
            )  # makes money if outcome == 0
            rejectingReward = 1 + (
                lambdaRejecting * difference
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

            if wealthRejecting > 1 / epsilon:
                # print(f"rejected after {loop} iterations")
                return violations, False, loop
            if wealthAccepting > 1 / epsilon:
                # print(f"accepted after {loop} iterations")
                return violations, True, loop

            # calculate lambdas for next iteration via ONS from Waudby-Smith
            # sumSquared += z**2

            # if onsLambda is large we expect a 0 outcome
            # if onsLambda is negative we expect a 1 outcome
            # onsLambda -= (2 * z) / ((2 - math.log(3)) * sumSquared)

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
            # lambdaAccepting = max(
            #    0, min(onsLambda, 0.75 / (self.maxWeight - epsilon))
            # )  # makes money if outcome < m, so large lambda if onsLambda is positive

            # lambdaRejecting = max(
            #    0, min(-onsLambda, 0.75 / epsilon)
            # )  # makes money if outcome > m, so large lambda if onsLambda is negative

            # onsLambda = max(
            #    min(onsLambda, 0.75 / (self.maxWeight - epsilon)), -0.75 / epsilon
            # )

        if loop == 0:
            return [], False, 0
        # print(f"indifferent after {loop} iterations")
        # print(
        #     f"onsLambda:{onsLambda}, lambdaAccepting:{lambdaAccepting}, maxWeight:{self.maxWeight}"
        # )
        # print(
        #    f"empirical mean:{outcomeSum / loop}, number of nonzero outcomes:{outcomeCOunter}"
        # )
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

        if agentActions[0] not in self.actions:
            self.transitions.update(
                self.state.get_possible_actions_with_probabilities(agentActions[0])
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

        self.updateMaxWeight(agentActions)

        return outcome, transitionSequence

    def get_child(self, state: SimulationState, action):
        for c in self.children.values():
            if state.__eq__(c.state) and action in c.actions:
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
