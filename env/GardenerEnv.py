import time
from typing import Optional

import gymnasium as gym
import numpy as np

from env.dynamics import GardenerDynamics
from env.rendering import GardenerRenderer
from env.state import GardenerState


class GardenerEnv(gym.Env):

    def __init__(self, size: int = 10):
        # The size of the square grid (5x5 by default)

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._state = GardenerState()
        self._state.size = size
        num_frogs = max(1, int(size * size * 0.05))
        num_lakes = max(1, int(size * size * 0.03))
        num_grass = max(1, int(size * size * 0.03))
        num_walls = int(size * size * 0.20)
        self._state.walls = np.full((num_walls, 2), -1, dtype=int)

        self._state.agent = np.array([-1, -1], dtype=int)
        self._state.frogs = np.full((num_frogs, 2), -1, dtype=int)
        self._state.lakes = np.full((num_lakes, 2), -1, dtype=int)
        self._state.lake_full = np.ones(num_lakes, dtype=bool)
        self._state.lake_timer = np.zeros(num_lakes, dtype=int)
        self._state.grass = np.full((num_grass, 2), -1, dtype=int)
        self._state.active_grass = 0

        # Define what actions are available (4 directions + 1 do nothing)
        self.action_space = gym.spaces.Discrete(5)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {"agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
             # [x, y] coordinates
             "frogs": gym.spaces.Box(0, size - 1, shape=(num_frogs, 2),
                                     dtype=int),  # array of [x, y] coordinates
             "lakes": gym.spaces.Box(0, size - 1, shape=(num_lakes, 2),
                                     dtype=int),  # array of [x, y] coordinates
             "grass": gym.spaces.Box(0, size - 1, shape=(num_grass, 2),
                                     dtype=int),
             "active_grass": gym.spaces.Discrete(num_grass),
             "walls": gym.spaces.Box(0, size - 1, shape=(num_walls, 2),
                                     dtype=int),
             "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,),
                                           dtype=np.int8),
             })

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        action_to_direction = {0: np.array([1, 0]),
                                     # Move right (positive x)
                                     1: np.array([0, 1]),
                                     # Move up (positive y)
                                     2: np.array([-1, 0]),
                                     # Move left (negative x)
                                     3: np.array([0, -1]),
                                     # Move down (negative y)
                                     4: np.array([0, 0]),  # Do nothing
                                     }


        self._renderer = GardenerRenderer()
        self._dynamics = GardenerDynamics(self.action_space, action_to_direction)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent, target and frog positions
        """
        return {"agent": self._state.agent,
                "frogs": self._state.frogs, "lakes": self._state.lakes,
                "grass": self._state.grass,
                "active_grass": self._state.active_grass,
                "walls": self._state.walls,
                "action_mask": self._dynamics.get_action_mask(self._state,
                                                              self._state.agent)}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {"distance": np.linalg.norm(
            self._state.agent - self._state.grass[self._state.active_grass],
            ord=1)}

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._state.agent = self.np_random.integers(0, self._state.size,
                                                    size=2,
                                                    dtype=int)

        # Place frogs randomly on the grid, avoiding agent
        all_positions = {(x, y) for x in range(self._state.size) for y in
                         range(self._state.size)}
        all_positions.discard(tuple(self._state.agent))
        frog_positions = self.np_random.choice(list(all_positions),
                                               size=len(self._state.frogs),
                                               replace=False)
        for frog_pos in frog_positions:
            all_positions.discard(tuple(frog_pos))
        lake_positions = self.np_random.choice(list(all_positions),
                                               size=len(self._state.lakes),
                                               replace=False)

        for lake_pos in lake_positions:
            all_positions.discard(tuple(lake_pos))
            lx, ly = lake_pos
            neighbors = [
                (lx + 1, ly), (lx - 1, ly),
                (lx, ly + 1), (lx, ly - 1)
            ]
            for nx, ny in neighbors:
                if 0 <= nx < self._state.size and 0 <= ny < self._state.size:
                    all_positions.discard((nx, ny))
        grass_positions = self.np_random.choice(list(all_positions),
                                                size=len(self._state.grass),
                                                replace=False)


        for grass_pos in grass_positions:
            all_positions.discard(tuple(grass_pos))

        # Generate walls ensuring accessibility of all non-lake, non-wall cells
        def is_accessible(excluded):
            # BFS over free cells
            free = {(x, y) for x in range(self._state.size)
                    for y in range(self._state.size)}
            free -= set(map(tuple, lake_positions))
            free -= set(excluded)
            if not free:
                return True
            start = next(iter(free))
            stack = [start]
            visited = set([start])
            while stack:
                cx, cy = stack.pop()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in free and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            return visited == free

        wall_positions = []
        remaining_positions = list(all_positions)
        self.np_random.shuffle(remaining_positions)
        for pos in remaining_positions:
            if len(wall_positions) == len(self._state.walls):
                break
            trial = wall_positions + [tuple(pos)]
            if is_accessible(trial):
                wall_positions.append(tuple(pos))
        self._state.walls = np.array(wall_positions, dtype=int)


        # np_random.choice returns a 1D array if input is 1D, so convert to
        # 2D array of positions
        self._state.frogs = np.array(frog_positions, dtype=int)
        self._state.lakes = np.array(lake_positions, dtype=int)
        self._state.grass = np.array(grass_positions, dtype=int)
        self._state.active_grass = self.np_random.integers(0, len(self._state.grass))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def sample(self, horizon, size):
        # create a copy of current random variable that does not influence og
        np_random = np.random.Generator(self.np_random.bit_generator.jumped())

        start_time = time.time()

        for i in range(size):
            # deep copy of full environment state
            state = self._state.fast_clone()
            for h in range(horizon):
                self._dynamics.move_frogs(state, np_random)

        elapsed = time.time() - start_time
        print(f"sample() took {elapsed:.6f} seconds for size={size}, horizon={horizon}")

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        terminated = self._dynamics.move_agent(self._state, action, self.np_random)
        self._dynamics.move_frogs(self._state, self.np_random)

        # Update lake states based on frog adjacency
        for i, (lx, ly) in enumerate(self._state.lakes):
            # Decrease timer if running
            if self._state.lake_timer[i] > 0:
                self._state.lake_timer[i] -= 1
                if self._state.lake_timer[i] == 0:
                    self._state.lake_full[i] = True  # refill lake

            # Check adjacency to any frog (Manhattan distance 1)
            for fx, fy in self._state.frogs:
                if abs(fx - lx) + abs(fy - ly) == 1 and self._state.lake_full[
                    i]:
                    self._state.lake_full[i] = False
                    self._state.lake_timer[i] = 5
                    break

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to
        # encourage efficiency
        reward = 1 if terminated else -0.01

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        self._renderer.draw(self._state)
