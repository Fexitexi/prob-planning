from env.dynamics import GardenerDynamics


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
        # clone state to avoid modifying real environment
        temp_state = state.fast_clone()

        # simulate agent move
        self._dynamics.move_agent(temp_state, action)

        # Build obstacle set
        size = temp_state.size
        walls = {tuple(w) for w in temp_state.walls}
        lakes = {tuple(l) for l in temp_state.lakes}

        # choose target set
        if target_type == "grass":
            targets = [tuple(temp_state.grass[temp_state.active_grass])]
        elif target_type == "lake":
            targets = []
            for (lx, ly), full in zip(temp_state.lakes, temp_state.lakes_full):
                if not full:
                    continue
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    tx, ty = lx + dx, ly + dy
                    if 0 <= tx < size and 0 <= ty < size:
                        if (tx, ty) not in walls and (tx, ty) not in lakes:
                            targets.append((tx, ty))
        else:
            return None

        from collections import deque
        ax, ay = temp_state.agent
        start = (ax, ay)

        # BFS
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            (x, y), d = q.popleft()
            if (x, y) in targets:
                # normalize distance here to the instance size
                if target_type == "grass":
                    return 1 - (d / (size * size))
                if target_type == "lake":
                    if d < 5:
                        return 1 - (d / 5)
                    else:
                        return 0.0
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if (nx, ny) in visited:
                        continue
                    if (nx, ny) in walls:
                        continue
                    if (nx, ny) in lakes:
                        continue
                    visited.add((nx, ny))
                    q.append(((nx, ny), d + 1))
        return 0.0

    def will_reach_target_after_action(self, state, action, target_type):
        """
        Returns True if executing `action` places the agent on:
            - the active grass patch               (target_type == "grass")
            - any cell adjacent to a full lake     (target_type == "lake")
        Otherwise returns False.
        """
        temp_state = state.fast_clone()
        temp_state_prev = state.fast_clone()
        self._dynamics.move_agent(temp_state, action)

        ax, ay = temp_state.agent

        if target_type == "grass":
            gx, gy = temp_state_prev.grass[temp_state_prev.active_grass]
            return 1.0 if (ax, ay) == (gx, gy) else 0.0

        if target_type == "lake":
            for i, (lx, ly) in enumerate(temp_state_prev.lakes):
                if temp_state_prev.lakes_full[i]:
                    if abs(ax - lx) + abs(ay - ly) == 1:
                        return 1.0
            return 0.0

        return 0.0