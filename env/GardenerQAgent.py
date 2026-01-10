from env import util
from env.FeatureExtractor import FeatureExtractor
import numpy as np
import random

from env.dynamics import get_action_mask


class GardenerQAgent:

    def __init__(self, seed=None, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0,):
        self._weights = util.Counter()
        self.featExtractor = FeatureExtractor(seed)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.numTraining = numTraining
        self.episodesSoFar = 0

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.get_features(state, action)
        weights = self._weights
        return sum([features[key] * weights[key] for key in features])

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        try:
            return max([self.getQValue(state, action)
                        for action in np.where(get_action_mask(state) == 1)[0]])
        except  ValueError:
            return 0.0

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        futureValue = reward + self.discount * self.getValue(nextState)
        currentValue = self.getQValue(state, action)
        predictedReward = futureValue - currentValue
        features = self.featExtractor.get_features(state, action)
        for key in self._weights:
            self._weights[key] += self.alpha * predictedReward * features[key]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        if self.epsilon > 0.0:
            legalActions = np.where(get_action_mask(state) == 1)[0]
            if util.flipCoin(self.epsilon):
                #print("Random Action")
                return random.choice(legalActions)

        return self.computeActionFromQValues(state)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actionValuePairs = [(action, self.getQValue(state, action))
                            for action in np.where(get_action_mask(state) == 1)[0]]
        if actionValuePairs == []:
            return None
        maxValue = max(actionValuePairs, key=lambda x: x[1])[1]

        # Filter into list with same max value.
        bestActions = list(
            filter(lambda x: x[1] == maxValue, actionValuePairs))
        #print(f"Best actions: {bestActions}, value: {maxValue}")
        return bestActions[0][0]

    def getBestActions(self, state):
        actionValuePairs = [(action, self.getQValue(state, action))
                            for action in np.where(get_action_mask(state) == 1)[0]]
        if actionValuePairs == []:
            return None
        maxValue = max(actionValuePairs, key=lambda x: x[1])[1]

        # Filter into list with same max value.
        bestActions = list(
            filter(lambda x: x[1] == maxValue, actionValuePairs))
        first_elements = [a for (a, _) in bestActions]
        return first_elements

    def observeTransition(self, action, nextState, deltaReward):
        """
            Called by the environment after each step
        """
        self.episodeRewards += deltaReward
        self.update(self.lastState, action, nextState, deltaReward)
        self.lastState = nextState.fast_clone()

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def stopLearning(self):
        self.epsilon = 0.0  # no exploration
        self.alpha = 0.0

    def registerInitialState(self, state):
        self.startEpisode()
        self.lastState = state
        # todo log
        #if self.episodesSoFar == 0:
        #    print('Beginning %d episodes of Training' % (self.numTraining))

    def save_weights(self, filepath):
        """
        Save learned weights to a text file.
        Each line is stored as: key<TAB>value
        """
        with open(filepath, "w") as f:
            for key, value in self._weights.items():
                f.write(f"{key}\t{value}\n")

    def load_weights(self, filepath):
        """
        Load weights from a text file saved with save_weights.
        """
        self._weights.clear()
        with open(filepath, "r") as f:
            for line in f:
                key, value = line.strip().split("\t")
                self._weights[key] = float(value)
        #print(self._weights)