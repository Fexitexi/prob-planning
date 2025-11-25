from env import util
from env.FeatureExtractor import FeatureExtractor


class GardenerQAgent:

    def __init__(self, seed=None):
        self._weights = util.Counter()
        self.featExtractor = FeatureExtractor(seed)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.get_features(state, action)
        weights = self._weights
        return sum([features[key] * weights[key] for key in features])