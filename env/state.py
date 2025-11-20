import numpy as np

class GardenerState:
    agent: np.ndarray
    frogs: np.ndarray
    lakes: np.ndarray
    lake_full: np.ndarray
    lake_timer: np.ndarray
    grass: np.ndarray
    walls: np.ndarray
    size: int
    active_grass: int

    def fast_clone(self):
        new = GardenerState()
        new.size = self.size
        new.active_grass = self.active_grass
        new.agent = self.agent.copy()
        new.grass = self.grass.copy()
        new.walls = self.walls.copy()
        new.frogs = self.frogs.copy()
        new.lakes = self.lakes.copy()
        new.lake_full = self.lake_full.copy()
        new.lake_timer = self.lake_timer.copy()
        return new