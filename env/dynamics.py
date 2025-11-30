import numpy as np


class GardenerDynamics:

    # Map action numbers to actual movements on the grid
    # This makes the code more readable than using raw numbers

    _action_to_direction = {0: np.array([1, 0]),
                                     # Move right (positive x)
                                     1: np.array([0, 1]),
                                     # Move up (positive y)
                                     2: np.array([-1, 0]),
                                     # Move left (negative x)
                                     3: np.array([0, -1]),
                                     # Move down (negative y)
                                     4: np.array([0, 0]),  # Do nothing
                                     }

    def __init__(self, seed):
        if seed is None:
            seed = np.random.SeedSequence().entropy
        self.np_random = np.random.Generator(np.random.PCG64(seed))

    def get_action_mask(self, state, pos):
        mask = np.ones(len(self._action_to_direction), dtype=np.int8)
        x, y = pos
        # Prevent moves that leave the grid
        if x == state.size - 1: mask[0] = 0
        if y == state.size - 1: mask[1] = 0
        if x == 0: mask[2] = 0
        if y == 0: mask[3] = 0
        # Prevent moves that would step onto a lake
        for action, direction in self._action_to_direction.items():
            nx, ny = pos + direction
            if any((nx == lx and ny == ly) for lx, ly in state.lakes):
                mask[action] = 0
            if any((nx == wx and ny == wy) for wx,wy in state.walls):
                mask[action] = 0
        return mask

    def move_agent(self, state, action):
        mask = self.get_action_mask(state, state.agent)

        if mask[action] == 0:
            raise ValueError(f"Illegal action {action}: action_mask={mask}")

        # Map the discrete action (0-4) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        state.agent = state.agent + direction

        # Check if agent reached any active grass patch
        terminated = False
        for (gx, gy), active in zip(state.grass, state.grass_active):
            if active and np.array_equal(state.agent, np.array([gx, gy])):
                terminated = True
                break

        return terminated


    def move_frogs(self, state):
        frogs = state.frogs  # shape (N, 2)
        size = state.size

        if frogs.size == 0:
            return

        # Directions for actions 0-3 as a (4, 2) array
        directions = np.stack([self._action_to_direction[a] for a in range(4)], axis=0)

        # ------------------------------------------------------------------
        # Compute which actions are allowed for all frogs at once
        # ------------------------------------------------------------------
        # Candidate next positions for each frog and action: (N, 4, 2)
        cand_pos = frogs[:, None, :] + directions[None, :, :]

        # Stay within grid bounds
        in_bounds_x = (cand_pos[:, :, 0] >= 0) & (cand_pos[:, :, 0] < size)
        in_bounds_y = (cand_pos[:, :, 1] >= 0) & (cand_pos[:, :, 1] < size)
        in_bounds = in_bounds_x & in_bounds_y  # (N, 4)

        # Avoid stepping onto any lake
        lakes = state.lakes
        if lakes.size > 0:
            # cand_pos: (N,4,2) vs lakes: (L,2) -> broadcast to (N,4,L,2)
            diff = cand_pos[:, :, None, :] - lakes[None, None, :, :]
            on_lake = (diff == 0).all(axis=3).any(axis=2)  # (N, 4)
        else:
            on_lake = np.zeros_like(in_bounds, dtype=bool)

        # Avoid walls
        walls = state.walls
        if walls.size > 0:
            diffw = cand_pos[:, :, None, :] - walls[None, None, :, :]
            on_wall = (diffw == 0).all(axis=3).any(axis=2)
        else:
            on_wall = np.zeros_like(in_bounds, dtype=bool)

        allowed = in_bounds & (~on_lake) & (~on_wall)  # (N, 4)

        # Frogs with no allowed moves stay in place
        n_allowed = allowed.sum(axis=1)  # (N,)
        no_allowed = n_allowed == 0

        # ------------------------------------------------------------------
        # Preferred action based on BFS best step toward nearest full lake
        # ------------------------------------------------------------------
        full_lake_indices = np.where(state.lakes_full)[0]
        if len(full_lake_indices) == 0:
            preferred = np.zeros_like(allowed, dtype=bool)
        else:
            # For each frog, pick the lake with minimal BFS distance
            dstack = np.stack([state.lake_dist[k][frogs[:,0], frogs[:,1]] for k in full_lake_indices], axis=1)
            nearest_choice = np.argmin(dstack, axis=1)
            chosen_lake_idx = full_lake_indices[nearest_choice]

            # Retrieve best-step vectors
            deltas = np.stack([state.lake_best_step[k][frogs[i,0], frogs[i,1]]
                               for i, k in enumerate(chosen_lake_idx)], axis=0)

            # Match deltas to available move directions
            dx_pref = deltas[:, 0:1]
            dy_pref = deltas[:, 1:2]

            match_x = (dx_pref != 0) & (dx_pref == directions[None, :, 0])
            match_y = (dy_pref != 0) & (dy_pref == directions[None, :, 1])
            preferred = match_x | match_y

        # Only actions that are allowed can be chosen
        pref_allowed = preferred & allowed

        n_pref = pref_allowed.sum(axis=1)           # (N,)
        n_other = n_allowed - n_pref                # (N,)

        # ------------------------------------------------------------------
        # Build probability distribution over actions for each frog
        # ------------------------------------------------------------------
        weights = np.zeros_like(allowed, dtype=float)

        # Rows where we use uniform over allowed actions
        uniform = (n_pref == 0) | (n_pref == n_allowed)
        uniform &= ~no_allowed  # must also have at least one allowed action

        if np.any(uniform):
            rows = np.where(uniform)[0]
            weights[rows] = allowed[rows] / n_allowed[rows, None]

        # Rows where we use 0.7 / 0.3 split between preferred / other
        mixed = ~(uniform | no_allowed)
        if np.any(mixed):
            rows = np.where(mixed)[0]
            pref_mask = pref_allowed[rows]
            other_mask = (allowed[rows] & ~pref_allowed[rows])

            # 0.7 total mass on preferred, 0.3 on others
            w_pref = 0.7 / n_pref[rows]
            w_other = 0.3 / n_other[rows]

            weights[rows] += pref_mask * w_pref[:, None]
            weights[rows] += other_mask * w_other[:, None]

        # Sanity: due to numerical issues, renormalize where sum > 0
        sums = weights.sum(axis=1, keepdims=True)
        positive = sums[:, 0] > 0
        if np.any(positive):
            weights[positive] /= sums[positive]

        # ------------------------------------------------------------------
        # Sample one action for each frog using the row-wise distributions
        # ------------------------------------------------------------------
        N = frogs.shape[0]
        new_positions = frogs.copy()

        # Frogs with at least one allowed action
        movable = ~no_allowed
        if np.any(movable):
            rows = np.where(movable)[0]
            w = weights[rows]  # (M,4)

            # CDF sampling per row
            cdf = np.cumsum(w, axis=1)
            # Avoid tiny numerical gaps: ensure last entry is exactly 1
            cdf[:, -1] = 1.0
            r = self.np_random.random(size=len(rows))[:, None]
            chosen_actions = (cdf >= r).argmax(axis=1)  # (M,)

            moves = directions[chosen_actions]          # (M,2)
            updated = frogs[rows] + moves
            # Clip to grid
            updated = np.clip(updated, 0, size - 1)
            new_positions[rows] = updated

        state.frogs = new_positions.astype(int)
