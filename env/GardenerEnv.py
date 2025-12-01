import time
from typing import Optional

import gymnasium as gym
import numpy as np

from env.dynamics import GardenerDynamics
from env.rendering import GardenerRenderer
from env.state import GardenerState


class GardenerEnv(gym.Env):

    def __init__(self, size: int = 15, grass_respawn: int = 50, lake_respawn: int = 20):
        # The size of the square grid (5x5 by default)

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._state = GardenerState()
        self._state.size = size
        self._state.grass_respawn = grass_respawn
        self._state.lake_respawn = lake_respawn
        num_frogs = max(1, int(size * size * 0.01))
        num_lakes = max(1, int(size * size * 0.02))
        num_grass = max(1, int(size * size * 0.04))
        num_walls = int(size * size * 0.20)
        self._state.walls = np.full((num_walls, 2), -1, dtype=int)

        self._state.agent = np.array([-1, -1], dtype=int)
        self._state.frogs = np.full((num_frogs, 2), -1, dtype=int)
        self._state.lakes = np.full((num_lakes, 2), -1, dtype=int)
        self._state.lakes_full = np.ones(num_lakes, dtype=bool)
        self._state.lake_timer = np.zeros(num_lakes, dtype=int)
        self._state.grass = np.full((num_grass, 2), -1, dtype=int)
        # Each grass patch starts active. A timer is used for reactivation.
        self._state.grass_active = np.ones(num_grass, dtype=bool)
        self._state.grass_timer = np.zeros(num_grass, dtype=int)
        self._state.score = 0

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
             "size": gym.spaces.Discrete(size + 1),
             "grass_respawn": gym.spaces.Discrete(grass_respawn + 1),
             "lake_respawn": gym.spaces.Discrete(lake_respawn + 1),
             "lakes_full": gym.spaces.Box(0, 1, shape=(num_lakes,),
                                          dtype=bool),
             "lake_timer": gym.spaces.Box(0, lake_respawn, shape=(num_lakes,), dtype=int),
             "grass": gym.spaces.Box(0, size - 1, shape=(num_grass, 2),
                                     dtype=int),
             "grass_active": gym.spaces.Box(0, 1, shape=(num_grass,), dtype=bool),
             "grass_timer": gym.spaces.Box(0, grass_respawn, shape=(num_grass,), dtype=int),
             "walls": gym.spaces.Box(0, size - 1, shape=(num_walls, 2),
                                     dtype=int),
             "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,),
                                           dtype=np.int8),
             })


        self._renderer = GardenerRenderer()
        self._dynamics = GardenerDynamics(self._np_random_seed)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent, target and frog positions
        """
        return {"agent": self._state.agent,
                "frogs": self._state.frogs,
                "size": self._state.size,
                "lake_respawn": self._state.lake_respawn,
                "grass_respawn": self._state.grass_respawn,
                "lakes": self._state.lakes,
                "lakes_full": self._state.lakes_full,
                "lake_timer": self._state.lake_timer,
                "grass": self._state.grass,
                "grass_active": self._state.grass_active,
                "grass_timer": self._state.grass_timer,
                "walls": self._state.walls,
                "action_mask": self._dynamics.get_action_mask(self._state,
                                                              self._state.agent)}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {"distance": np.linalg.norm(
            self._state.agent - self._state.grass[self._state.grass_active.argmax()],
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

        self._dynamics = GardenerDynamics(seed=seed)

        self._state.score = 0

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
        if len(wall_positions) < len(self._state.walls):
            return self.reset(seed=seed)
        self._state.walls = np.array(wall_positions, dtype=int)

        print("free candidates:", len(remaining_positions))
        print("accepted:", len(wall_positions))

        # -------------------------------------------------------------
        # Precompute shortest-path distance and best-step fields
        # -------------------------------------------------------------
        size = self._state.size
        walls_set = {tuple(w) for w in self._state.walls}
        lake_dist = []
        lake_best_step = []

        from collections import deque

        for (lx, ly) in lake_positions:
            dist = np.full((size, size), np.iinfo(np.int32).max, dtype=np.int32)
            best = np.zeros((size, size, 2), dtype=np.int8)

            q = deque()
            q.append((lx, ly))
            dist[lx, ly] = 0

            while q:
                x, y = q.popleft()
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if (nx, ny) in walls_set:
                            continue
                        if dist[nx, ny] > dist[x, y] + 1:
                            dist[nx, ny] = dist[x, y] + 1
                            best[nx, ny] = np.array([-dx, -dy], dtype=np.int8)
                            q.append((nx, ny))

            lake_dist.append(dist)
            lake_best_step.append(best)

        self._state.lake_dist = lake_dist
        self._state.lake_best_step = lake_best_step


        # np_random.choice returns a 1D array if input is 1D, so convert to
        # 2D array of positions
        self._state.frogs = np.array(frog_positions, dtype=int)
        self._state.lakes = np.array(lake_positions, dtype=int)
        self._state.lake_timer = np.ones(len(self._state.lakes), dtype=int)
        self._state.grass = np.array(grass_positions, dtype=int)
        self._state.grass_active[:] = True
        self._state.grass_timer[:] = 0

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
                self._dynamics.move_frogs(state)

        elapsed = time.time() - start_time
        print(f"sample() took {elapsed:.6f} seconds for size={size}, horizon={horizon}")


    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        grass_patch = self._dynamics.move_agent(self._state, action)

        reward = 0

        # Update grass states
        for i, (gx, gy) in enumerate(self._state.grass):
            ax, ay = self._state.agent
            if ax == gx and ay == gy:
                if self._state.grass_active[i]:
                    self._state.grass_active[i] = False
                    reward += 10
                    self._state.grass_timer[i] = self._state.grass_respawn
            else:
                if not self._state.grass_active[i] and self._state.grass_timer[i] > 0:
                    self._state.grass_timer[i] -= 1
                    if self._state.grass_timer[i] == 0:
                        self._state.grass_active[i] = True

        self._dynamics.move_frogs(self._state)

        # Update lake states based on frog adjacency
        for i, (lx, ly) in enumerate(self._state.lakes):
            # Decrease timer if running
            if self._state.lake_timer[i] > 0:
                self._state.lake_timer[i] -= 1
                if self._state.lake_timer[i] == 0:
                    self._state.lakes_full[i] = True  # refill lake

            # Check adjacency to any frog (Manhattan distance 1)
            #for fx, fy in self._state.frogs:
            #    if abs(fx - lx) + abs(fy - ly) == 1 and self._state.lakes_full[i]:
            #        self._state.lakes_full[i] = False
            #        self._state.lake_timer[i] = 20
            #        break

            # Check adjacency to the agent (Manhattan distance 1)
            ax, ay = self._state.agent
            if self._state.lakes_full[i] and abs(ax - lx) + abs(ay - ly) == 1:
                # Additional reward for being adjacent (Manhattan distance 1) to any full lake
                reward += 5
                self._state.lakes_full[i] = False
                self._state.lake_timer[i] = self._state.lake_respawn

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        self._state.score += reward

        observation = self._get_obs()
        info = self._get_info()

        # Terminate if point limit reached
        terminated = True if self._state.score >= 300 else False

        return observation, reward, terminated, truncated, info

    def render(self):
        self._renderer.draw(self._state)
