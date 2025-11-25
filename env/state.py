from dataclasses import dataclass

import numpy as np

class GardenerState:
    agent: np.ndarray
    frogs: np.ndarray
    lakes: np.ndarray
    lakes_full: np.ndarray
    lake_dist: np.ndarray
    lake_best_step: np.ndarray
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
        new.lake_dist = self.lake_dist.copy()
        new.lake_best_step = self.lake_best_step.copy()
        new.lakes_full = self.lakes_full.copy()
        new.lake_timer = self.lake_timer.copy()
        return new

@dataclass
class ObservationState:
    agent: np.ndarray
    frogs: np.ndarray
    lakes: np.ndarray
    lakes_full: np.ndarray
    grass: np.ndarray
    walls: np.ndarray
    size: int
    active_grass: int

    @staticmethod
    def from_obs(obs):
        return ObservationState(
            agent=obs["agent"],
            frogs=obs["frogs"],
            lakes=obs["lakes"],
            lakes_full=obs["lakes_full"],
            grass=obs["grass"],
            active_grass=obs["active_grass"],
            walls=obs["walls"],
            size=obs["size"]
        )

    def fast_clone(self):
        return ObservationState(
            agent=self.agent.copy(),
            frogs=self.frogs.copy(),
            lakes=self.lakes.copy(),
            lakes_full=self.lakes_full.copy(),
            grass=self.grass.copy(),
            walls=self.walls.copy(),
            size=self.size,
            active_grass=self.active_grass
        )