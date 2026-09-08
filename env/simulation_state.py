from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from env import dynamics


@dataclass(eq=False, frozen=True)
class SimulationState:
    PREFERENCE_WEIGHT: ClassVar[float] = 0.7
    agent: np.ndarray  # [X, Y]
    frogs: np.ndarray  # [[X, Y], [X, Y]]
    dead_frogs: np.ndarray
    captured_frogs: np.ndarray
    lakes: np.ndarray  # [[X, Y], [X, Y]]
    lakes_full: np.ndarray  # [True/False, True/False]
    lake_timer: np.ndarray  # [0, 1, 2]
    lake_respawn: int
    frog_timer: np.ndarray  # [0, 1, 2]
    walls: np.ndarray  # [[X, Y], [X, Y], [X, Y]]
    pos_actions: np.ndarray
    lake_dist: list
    lake_action_grid: np.ndarray
    size: int
    ctd: bool = False
    violation: bool = False

    def __post_init__(self):
        arrays = (
            self.agent,
            self.frogs,
            self.dead_frogs,
            self.captured_frogs,
            self.lakes,
            self.lakes_full,
            self.lake_timer,
            self.frog_timer,
            self.walls,
            self.pos_actions,
            self.lake_action_grid,
        )
        for arr in arrays:
            arr.flags.writeable = False

        for lake_dist_arr in self.lake_dist:
            lake_dist_arr.flags.writeable = False

        key = (
            b"".join(a.tobytes() for a in arrays)
            + self.size.to_bytes(4, "little")
            + self.lake_respawn.to_bytes(4, "little")
            + self.ctd.to_bytes(1, "little")
            + bool(self.violation).to_bytes(1, "little")
        )
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_hash", hash(key))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, SimulationState):
            return NotImplemented
        return self._key == other._key

    def get_possible_actions_with_probabilities(self, agentAction):
        # Convert the agent action to a hashable tuple.
        possible = {(agentAction,): 1.0}

        for i, frog in enumerate(self.frogs):
            frog_actions = {}
            # skip stunned frogs
            if self.frog_timer[i] > 0 or self.captured_frogs[i] or self.dead_frogs[i]:
                possible = {k + (4,): v for k, v in possible.items()}
                continue

            preferred_dir, other_dirs = self.get_frog_actions(frog)

            if preferred_dir is not None and other_dirs:
                # preferred = 70%, other options share the 30%
                frog_actions[preferred_dir] = SimulationState.PREFERENCE_WEIGHT
                other_prob = (1.0 - SimulationState.PREFERENCE_WEIGHT) / len(other_dirs)
                for d in other_dirs:
                    frog_actions[d] = other_prob
            elif preferred_dir is not None and not other_dirs:
                # only preferred choice => probability = 100%
                frog_actions[preferred_dir] = 1.0
            elif preferred_dir is None and other_dirs:
                # no preferred direction, evenly split among all choices
                prob = 1.0 / len(other_dirs)
                for d in other_dirs:
                    frog_actions[d] = prob
            else:
                # No valid movement; frog stays in place.
                frog_actions[4] = 1.0

            # Expand every existing action sequence with this frog's possible moves.
            new_possible = {}
            for action_seq, prob in possible.items():
                for frog_action, frog_prob in frog_actions.items():
                    new_seq = action_seq + (frog_action,)
                    new_possible[new_seq] = (
                        new_possible.get(new_seq, 0.0) + prob * frog_prob
                    )

            possible = new_possible

        return possible

    def get_frog_actions(self, frog):
        x, y = frog
        unreachable = np.iinfo(np.int32).max

        # Distance from (x, y) to every lake.
        dists = self.lake_dist[:, x, y]
        # dists = np.array([d[x, y] for d in self.lake_dist], dtype=np.int32)
        masked = np.where(self.lakes_full, dists, unreachable)
        nearest_idx = int(masked.argmin())

        preferred_idx = None
        if masked[nearest_idx] != unreachable:
            preferred_idx = int(self.lake_action_grid[nearest_idx, x, y])

        valid = self.pos_actions[x, y, :4]
        if (
            preferred_idx is not None
            and preferred_idx < 4
            and bool(valid[preferred_idx])
        ):
            preferred_dir = preferred_idx
            other_dirs = [a for a in range(4) if valid[a] and a != preferred_idx]
        else:
            preferred_dir = None
            other_dirs = [a for a in range(4) if valid[a]]
        return preferred_dir, other_dirs

    def apply_action(state, action):
        updates = {}

        updates["agent"] = state.agent.copy() + dynamics._action_to_direction[action[0]]
        if len(action) > 1:
            updates["frogs"] = state.frogs.copy() + [
                dynamics._action_to_direction[x] for x in action[1:]
            ]
        else:
            updates["frogs"] = state.frogs.copy()

        updates["violation"] = replace(state, **updates).has_violation()

        same_pos = (updates["agent"] == updates["frogs"]).all(axis=-1)
        updates["captured_frogs"] = (
            same_pos & (state.frog_timer > 0) & np.logical_not(state.dead_frogs)
        ) | state.captured_frogs

        updates["dead_frogs"] = (
            same_pos & np.logical_not(updates["captured_frogs"])
        ) | state.dead_frogs

        agent_empties = (np.abs(updates["agent"] - state.lakes).sum(axis=1) == 1) & (
            state.lake_timer == 0
        )
        updates["lake_timer"] = np.where(
            agent_empties, state.lake_respawn, np.maximum(state.lake_timer - 1, 0)
        )
        updates["lakes_full"] = updates["lake_timer"] == 0

        diff_frogs = np.abs(updates["frogs"][:, None, :] - state.lakes[None, :, :])
        near_lake = diff_frogs.max(axis=2) == 1
        near_emptied_lake = near_lake & agent_empties[None, :]
        frog_near_emptied_lake = near_emptied_lake.any(axis=1)

        updates["frog_timer"] = np.where(
            frog_near_emptied_lake, 5, np.maximum(state.frog_timer - 1, 0)
        )

        return replace(state, **updates)

    def has_violation(self):
        active = np.logical_not(self.dead_frogs | self.captured_frogs)
        if np.any(np.all(self.agent == self.frogs, axis=1) & active):
            return True

        if self.ctd:
            diff_agent = np.abs(self.agent - self.lakes)
            agent_orthogonal = diff_agent.sum(axis=1) == 1

            if not agent_orthogonal.any():
                return False

            active_frogs = np.logical_not(self.dead_frogs | self.captured_frogs)
            diff_frogs = np.abs(self.frogs[:, None, :] - self.lakes[None, :, :])
            frog_near_lake = (
                (diff_frogs.max(axis=2) == 1) & active_frogs[:, None]
            ).any(axis=0)

            combined_per_lake = agent_orthogonal & frog_near_lake & self.lakes_full
            return combined_per_lake.any()

        return False

    def calculate_violation_probability_brute_force(self, agentActions: list):
        violation_probability = 0.0
        # short circuit if an early violation happens
        if self.violation:
            return 1

        # return 0 if a leaf node (and no violation due to the if above)
        if len(agentActions) == 0:
            return 0

        for a, p in self.get_possible_actions_with_probabilities(
            agentActions[0]
        ).items():
            state = self.apply_action(a)
            violation_probability += (
                p * state.calculate_violation_probability_brute_force(agentActions[1:])
            )
        return violation_probability

    @staticmethod
    def _compute_lake_action_grid(lake_best_step):
        if not lake_best_step:
            return np.empty((0, 0, 0), dtype=np.int8)
        lake_steps = np.stack(lake_best_step, axis=0)
        action_lut = np.full((3, 3), -1, dtype=np.int8)
        action_lut[2, 1] = 0  # [ 1,  0]
        action_lut[1, 2] = 1  # [ 0,  1]
        action_lut[0, 1] = 2  # [-1,  0]
        action_lut[1, 0] = 3  # [ 0, -1]
        action_lut[1, 1] = 4  # [ 0,  0]
        return action_lut[lake_steps[..., 0] + 1, lake_steps[..., 1] + 1]

    @staticmethod
    def from_state(state, ctd: bool = False):
        return SimulationState(
            agent=state.agent.copy(),
            frogs=state.frogs.copy(),
            dead_frogs=state.dead_frogs.copy(),
            captured_frogs=state.capt_frogs.copy(),
            lakes=state.lakes.copy(),
            lakes_full=state.lakes_full.copy(),
            lake_timer=state.lake_timer.copy(),
            lake_respawn=state.lake_respawn,
            frog_timer=state.frog_timer.copy(),
            walls=state.walls.copy(),
            pos_actions=state.pos_actions,
            lake_dist=np.stack(state.lake_dist),
            lake_action_grid=SimulationState._compute_lake_action_grid(
                state.lake_best_step
            ),
            size=state.size,
            ctd=ctd,
            violation=False,
        )

    def fast_clone(self):
        return replace(self)
