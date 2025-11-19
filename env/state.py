import numpy as np

class GardenerState:
    agent: np.ndarray
    frogs: np.ndarray
    lakes: np.ndarray
    lake_full: np.ndarray
    lake_timer: np.ndarray
    target: np.ndarray
    size: int

    def fast_clone(self):
        new = GardenerState()
        new.size = self.size
        new.agent = self.agent.copy()
        new.target = self.target.copy()
        new.frogs = self.frogs.copy()
        new.lakes = self.lakes.copy()
        new.lake_full = self.lake_full.copy()
        new.lake_timer = self.lake_timer.copy()
        return new