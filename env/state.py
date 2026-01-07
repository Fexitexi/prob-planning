from dataclasses import dataclass

import numpy as np

class GardenerState:
    agent: np.ndarray
    pos_actions: dict
    frogs: np.ndarray
    lakes: np.ndarray
    lakes_full: np.ndarray
    lake_dist: np.ndarray
    lake_best_step: np.ndarray
    lake_dict: dict
    lake_timer: np.ndarray
    grass: np.ndarray
    grass_active: np.ndarray
    grass_timer: np.ndarray
    walls: np.ndarray
    size: int
    grass_respawn: int
    lake_respawn: int
    score: int

    def fast_clone(self):
        new = GardenerState()
        new.size = self.size
        new.grass_respawn = self.grass_respawn
        new.lake_respawn = self.lake_respawn
        new.score = self.score
        new.agent = self.agent.copy()
        new.pos_actions = self.pos_actions.copy()
        new.grass = self.grass.copy()
        new.grass_active = self.grass_active.copy()
        new.grass_timer = self.grass_timer.copy()
        new.walls = self.walls.copy()
        new.frogs = self.frogs.copy()
        new.lakes = self.lakes.copy()
        new.lake_dist = self.lake_dist.copy()
        new.lake_best_step = self.lake_best_step.copy()
        new.lakes_full = self.lakes_full.copy()
        new.lake_dict = self.lake_dict.copy()
        new.lake_timer = self.lake_timer.copy()
        return new

@dataclass
class ObservationState:
    agent: np.ndarray
    frogs: np.ndarray
    lakes: np.ndarray
    lakes_full: np.ndarray
    lake_timer: np.ndarray
    grass: np.ndarray
    grass_active: np.ndarray
    grass_timer: np.ndarray
    walls: np.ndarray
    size: int
    grass_respawn: int
    lake_respawn: int

    @staticmethod
    def from_obs(obs):
        return ObservationState(
            agent=obs["agent"],
            frogs=obs["frogs"],
            lakes=obs["lakes"],
            lakes_full=obs["lakes_full"],
            lake_timer=obs["lake_timer"],
            grass=obs["grass"],
            grass_active=obs["grass_active"],
            grass_timer=obs["grass_timer"],
            walls=obs["walls"],
            size=obs["size"],
            grass_respawn=obs["grass_respawn"],
            lake_respawn=obs["lake_respawn"]
        )

    def fast_clone(self):
        return ObservationState(
            agent=self.agent.copy(),
            frogs=self.frogs.copy(),
            lakes=self.lakes.copy(),
            lakes_full=self.lakes_full.copy(),
            lake_timer=self.lake_timer.copy(),
            grass=self.grass.copy(),
            grass_active=self.grass_active.copy(),
            grass_timer=self.grass_timer.copy(),
            walls=self.walls.copy(),
            size=self.size,
            grass_respawn=self.grass_respawn,
            lake_respawn=self.lake_respawn,
        )