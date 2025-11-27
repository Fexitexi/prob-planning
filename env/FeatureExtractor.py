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
            "exists_lake": 1.0 if any(state.lakes_full) else 0.0, }
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
            targets = [tuple(l) for i, l in enumerate(temp_state.lakes)
                       if temp_state.lakes_full[i]]
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
                return d / (temp_state.size * temp_state.size)
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if (nx, ny) in visited:
                        continue
                    if (nx, ny) in walls:
                        continue
                    if (nx, ny) in lakes and target_type != "lake":
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