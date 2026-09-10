import numpy as np


class GardenerDynamics:
    # Map action numbers to actual movements on the grid
    # This makes the code more readable than using raw numbers

    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.SeedSequence().entropy
        self.np_random = np.random.Generator(np.random.PCG64(seed))

    def move_agent(self, state, action):
        mask = get_action_mask(state)

        if mask[action] == 0:
            raise ValueError(f"Illegal action {action}: action_mask={mask}")

        # Map the discrete action (0-4) to a movement direction
        direction = _action_to_direction[action]

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
        frogs = state.frogs

        if len(frogs) == 0:
            return

        new_positions = []

        for i in range(len(frogs)):
            fx, fy = frogs[i]
            if state.dead_frogs[i] or state.capt_frogs[i] or state.frog_timer[i] > 0:
                new_positions.append([fx, fy])
                continue
            pos_tuple = (int(fx), int(fy))

            # Get valid moves from pre-computed array
            valid_moves = []
            if 0 <= fx < state.size and 0 <= fy < state.size:
                for a in range(4):
                    if state.pos_actions[int(fx), int(fy), a] == 1:
                        valid_moves.append(a)

            if not valid_moves:
                new_positions.append([fx, fy])
                continue

            preferred_action = None
            if pos_tuple in state.lake_dict:
                # List of (lake_idx, dist, action)
                for lake_idx, dist, action in state.lake_dict[pos_tuple]:
                    if state.lakes_full[lake_idx]:
                        preferred_action = action
                        break

            chosen_action = None

            # Try to take preferred action with 70% probability
            # We ensure preferred_action is actually valid
            if preferred_action is not None and preferred_action in valid_moves:
                if self.np_random.random() < 0.7:
                    chosen_action = preferred_action

            # If not chosen yet (either no preferred, or 30% chance triggered)
            if chosen_action is None:
                if preferred_action is not None:
                    # Try to pick a random "other" action
                    other_moves = [a for a in valid_moves if a != preferred_action]
                    if other_moves:
                        chosen_action = self.np_random.choice(other_moves)
                    else:
                        # If no other moves exist, fallback to preferred (if valid)
                        if preferred_action in valid_moves:
                            chosen_action = preferred_action
                else:
                    # No preferred action, just pick any random valid move
                    chosen_action = self.np_random.choice(valid_moves)

            if chosen_action is not None:
                d = _action_to_direction[chosen_action]
                new_positions.append([fx + d[0], fy + d[1]])
            else:
                new_positions.append([fx, fy])

        state.frogs = np.array(new_positions, dtype=int)


_action_to_direction = {
    0: np.array([1, 0]),
    # Move right (positive x)
    1: np.array([0, 1]),
    # Move up (positive y)
    2: np.array([-1, 0]),
    # Move left (negative x)
    3: np.array([0, -1]),
    # Move down (negative y)
    4: np.array([0, 0]),  # Do nothing
}


def get_action_mask_pos(pos, state):
    return state.pos_actions[pos[0]][pos[1]]


def get_action_mask(state):
    pos = state.agent
    return get_action_mask_pos(pos, state)
