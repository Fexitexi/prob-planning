from env.dynamics import GardenerDynamics, _action_to_direction
import numpy as np


class FeatureExtractor:

    def __init__(self, seed):
        self._dynamics = GardenerDynamics(seed)

    def get_features(self, state, action):
        features = {
            "mows_lawn": self.will_reach_target_after_action(state, action, "grass"),
            "sips_lake": self.will_reach_target_after_action(state, action, "lake"),
            "dist_lawn": self.shortest_path_after_action(state, action, "grass"),
            "dist_lake": self.shortest_path_after_action(state, action, "lake"),
        }
        return features

    def shortest_path_after_action(self, state, action, target_type):
        """
        Compute the shortest-path distance for the agent after executing `action`
        to either:
            - the nearest active grass cell  (target_type == "grass")
            - the nearest full lake cell     (target_type == "lake")

        Returns:
            int: number of steps in shortest path, or None if unreachable.
        """
        # simulate agent move
        direction = _action_to_direction[action]
        ax, ay = state.agent + direction
        
        # Check if move is valid (not into wall, not out of bounds)
        # Assuming state.pos_actions is available in observation state or similar
        # If not, we might need to check bounds and walls manually.
        # However, the Q-learning agent usually only calls this for legal actions.
        # Let's assume the action is legal or at least check bounds.
        if not (0 <= ax < state.size and 0 <= ay < state.size):
             return 0.0 # Or handle invalid move appropriately

        size = state.size

        if target_type == "grass":
            min_dist = np.iinfo(np.int32).max
            for i, active in enumerate(state.grass_active):
                if active:
                    # Use precomputed distance from state
                    # state.grass_dist is a list of 2D arrays, one for each grass patch
                    d = state.grass_dist[i][ax, ay]
                    if d < min_dist:
                        min_dist = d
            
            if min_dist == np.iinfo(np.int32).max:
                return 0.0
            return 1 - (min_dist / (size * size))

        elif target_type == "lake":
            min_dist = np.iinfo(np.int32).max
            for i, full in enumerate(state.lakes_full):
                if full:
                    # Use precomputed distance from state
                    # state.lake_dist is a list of 2D arrays, one for each lake
                    d = state.lake_dist[i][ax, ay]
                    if d < min_dist:
                        min_dist = d
            
            if min_dist == np.iinfo(np.int32).max:
                return 0.0
            
            if min_dist < 5:
                return 1 - (min_dist / 5)
            else:
                return 0.0
        else:
            return None

    def will_reach_target_after_action(self, state, action, target_type):
        """
        Returns True if executing `action` places the agent on:
            - the active grass patch               (target_type == "grass")
            - any cell adjacent to a full lake     (target_type == "lake")
        Otherwise returns False.
        """
        direction = _action_to_direction[action]
        ax, ay = state.agent + direction

        if target_type == "grass":
            for (gx, gy), active in zip(state.grass, state.grass_active):
                if active and (ax, ay) == (gx, gy):
                    return 1.0
            return 0.0

        if target_type == "lake":
            for i, (lx, ly) in enumerate(state.lakes):
                if state.lakes_full[i]:
                    if abs(ax - lx) + abs(ay - ly) == 1:
                        return 1.0
            return 0.0

        return 0.0