import numpy as np


class GardenerDynamics:

    def __init__(self, action_space, action_to_direction):
        self._action_space = action_space
        self._action_to_direction = action_to_direction

    def get_action_mask(self, state, pos):
        mask = np.ones(self._action_space.n, dtype=np.int8)
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

    def move_agent(self, state, action, np_random):
        mask = self.get_action_mask(state, state.agent)

        if mask[action] == 0:
            raise ValueError(f"Illegal action {action}: action_mask={mask}")

        # Map the discrete action (0-4) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        state.agent = state.agent + direction

        # Check if agent reached the active grass patch
        terminated = np.array_equal(state.agent,
                                    state.grass[state.active_grass])

        if terminated:
            choices = [i for i in range(len(state.grass)) if
                       i != state.active_grass]
            state.active_grass = np_random.choice(choices)

        return terminated


    def move_frogs(self, state, np_random):
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
        # Compute preferred actions based on nearest full lake (if any)
        # ------------------------------------------------------------------
        full_lakes = state.lakes[state.lake_full]

        if full_lakes.size > 0:
            # Manhattan distance from each frog to each full lake: (N, K)
            dists = np.abs(frogs[:, None, :] - full_lakes[None, :, :]).sum(axis=2)
            nearest_idx = np.argmin(dists, axis=1)  # (N,)
            targets = full_lakes[nearest_idx]       # (N, 2)
            deltas = np.sign(targets - frogs)       # (N, 2)
        else:
            # No full lakes -> no preference
            deltas = np.zeros_like(frogs)

        # Build preferred matrix (N, 4): action is preferred if it moves in the
        # sign direction of dx or dy towards the chosen full lake.
        dx = deltas[:, 0:1]  # (N,1)
        dy = deltas[:, 1:1+1]  # (N,1)

        # directions[:,0] shape (4,) -> (1,4) for broadcasting
        match_x = (dx != 0) & (dx == directions[None, :, 0])
        match_y = (dy != 0) & (dy == directions[None, :, 1])
        preferred = match_x | match_y  # (N,4)

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
            r = np_random.random(size=len(rows))[:, None]
            chosen_actions = (cdf >= r).argmax(axis=1)  # (M,)

            moves = directions[chosen_actions]          # (M,2)
            updated = frogs[rows] + moves
            # Clip to grid
            updated = np.clip(updated, 0, size - 1)
            new_positions[rows] = updated

        state.frogs = new_positions.astype(int)
