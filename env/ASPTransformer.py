import clingo
import time
import random
import numpy as np

from env.dynamics import GardenerDynamics


class ASPTransformer:
    """
    Transform a GardenerState into an ASP program with:
      - constant atoms   (walls, grass, lakes)
      - variable atoms   (agent, grass_timer, lake_timer)
    """

    def __init__(self, q_agent):
        self._static = None
        self._dynamic = None
        self._state = None
        self._dynamics = GardenerDynamics()
        self._q_agent = q_agent
        self._latest_model = None
        self._rnd = None
        self._lake_dict = None
        self._grass_dict = None
        self._dyn_lake_dict = None
        self._constraints = []
        self._check = None
        self._generate = None
        self._horizon = None
        self._frogs = []
        self._ctd = None


    def reset(self, ctd):
        self._state = None
        self._rnd = None
        self._ctd = ctd
        self._dyn_lake_dict = None
        self._latest_model = None
        self._constraints.clear()
        self._check = None
        self._generate = None
        self._dynamic = None
        self._frogs.clear()


    def build_static(self, state, horizon) -> str:
        lines = []

        self._lake_dict = {}
        self._grass_dict = {}

        lake_dist = []
        grass_dist = []
        lake_best_step = []
        walls_set = {tuple(w) for w in state.walls}
        lakes_set = {tuple(w) for w in state.lakes}

        from collections import deque

        for (lx, ly) in state.lakes:
            dist = np.full((state.size, state.size), np.iinfo(np.int32).max,
                           dtype=np.int32)
            best = np.zeros((state.size, state.size, 2), dtype=np.int8)

            q = deque()
            q.append((lx, ly))
            dist[lx, ly] = 0

            while q:
                x, y = q.popleft()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < state.size and 0 <= ny < state.size:
                        if (nx, ny) in walls_set or (nx, ny) in lakes_set:
                            continue
                        if dist[nx, ny] > dist[x, y] + 1:
                            dist[nx, ny] = dist[x, y] + 1
                            if dist[x, y] == 0:
                                # (x, y) is the lake. We are at (nx, ny) next to it.
                                # Set best move to [0, 0] to stop/interact here.
                                best[nx, ny] = np.array([0, 0], dtype=np.int8)
                            else:
                                # (x, y) is a safe path tile.
                                # Point backwards to it: (-dx, -dy).
                                best[nx, ny] = np.array([-dx, -dy],
                                                        dtype=np.int8)
                            q.append((nx, ny))

            lake_dist.append(dist)
            lake_best_step.append(best)

        for (lx, ly) in state.grass:
            dist = np.full((state.size, state.size), np.iinfo(np.int32).max,
                           dtype=np.int32)

            q = deque()
            q.append((lx, ly))
            dist[lx, ly] = 0

            while q:
                x, y = q.popleft()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < state.size and 0 <= ny < state.size:
                        if (nx, ny) in walls_set or (nx, ny) in lakes_set:
                            continue
                        if dist[nx, ny] > dist[x, y] + 1:
                            dist[nx, ny] = dist[x, y] + 1
                            q.append((nx, ny))

            grass_dist.append(dist)

        for c in range(state.size):
            for r in range(state.size):
                if (c, r) in walls_set or (c, r) in lakes_set:
                    continue
                lakes_info = []
                grass_info = []
                for i in range(len(state.lakes)):
                    dist = lake_dist[i][c, r]
                    step = lake_best_step[i][c, r]

                    action = -1
                    if step[0] == 1 and step[1] == 0:
                        action = 0
                    elif step[0] == 0 and step[1] == 1:
                        action = 1
                    elif step[0] == -1 and step[1] == 0:
                        action = 2
                    elif step[0] == 0 and step[1] == -1:
                        action = 3

                    lakes_info.append((i, dist, action))
                for i in range(len(state.grass)):
                    dist = grass_dist[i][c, r]
                    grass_info.append((i, dist))

                grass_info.sort(key=lambda x: x[1])
                lakes_info.sort(key=lambda x: x[1])
                self._lake_dict[(c, r)] = lakes_info
                self._grass_dict[(c, r)] = grass_info

        # constants
        lines.append(f"#const size={state.size}.")
        lines.append(f"#const horizon={horizon}.")
        lines.append(f"#const grass_respawn={state.grass_respawn}.")
        lines.append(f"#const lake_respawn={state.lake_respawn}.")
        lines.append("")
        self._horizon = horizon

        ## constant atoms: lakes
        line = ""
        for i, (c, r) in enumerate(state.lakes):
            line += f"lake({c}, {r}, {i})."
        lines.append(line)
        ## constant atoms: grass
        line = ""
        for i, (c, r) in enumerate(state.grass):
            line += f"grass({c}, {r}, {i})."
        lines.append(line)

        self._static = "\n".join(lines)
        return "\n".join(lines)

    def add_constraint(self, actions):
        line = ":-"
        for i,a in enumerate(actions):
            line += f" action({a}, {i}),"
        line = line.removesuffix(",")
        line+="."
        if line not in self._constraints:
            #print(f"Adding constraint: {line}")
            self._constraints.insert(0, line)
            return False
        else:
            return True

    def build_dynamic(self, state) -> str:
        self._state = state
        lines = []

        # agent position
        lines.append(f"agent({state.agent[0]}, {state.agent[1]}, 0).")

        # lake timer
        line = ""
        for i, c in enumerate(state.lake_timer):
            line += f"lake_timer({i}, {c+1}, 0)."
        lines.append(line)

        # grass timer
        line = ""
        for i, c in enumerate(state.grass_timer):
            line += f"grass_timer({i}, {c+1}, 0)."
        lines.append(line)


        # frog timer
        for f in range(len(state.frog_timer)):
            if state.frog_timer[f] > 0:
                for i in range(state.frog_timer[f]):
                    lines.append(f"frog_timer({f}, {i}).")

        return "\n".join(lines)

    def build_dynamic_worlds(self, state, num_worlds, horizon, sips) -> str:
        self._state = state
        lines = []

        # agent position
        lines.append(f"agent({state.agent[0]}, {state.agent[1]}, 0).")

        for s in sips:
            lines.append(f"ctd({sips[s][0]}, {s}, {sips[s][1]}).")

        # check the frogs in the "sphere" of the agent
        # agent window
        c_min = state.agent[0] - 2 * horizon
        c_max = state.agent[0] + 2 * horizon
        r_min = state.agent[1] - 2 * horizon
        r_max = state.agent[1] + 2 * horizon
        for c in range(c_min, c_max + 1):
            for r in range(r_min, r_max + 1):
                dist = abs(state.agent[0] - c) + abs(state.agent[1] - r)
                if dist <= 2 * horizon:
                    indices = np.where(np.all(state.frogs == [c, r], axis=1))[
                        0]
                    for i in indices:
                        if i not in self._frogs and not self._state.dead_frogs[i]:
                            self._frogs.append(i)

        self._rnd = {}
        # frogs
        for i in range(num_worlds):
            self._rnd[i] = {}  # Initialize the world level
            for f in self._frogs:
                self._rnd[i][f] = {}  # Initialize the frog level
                for t in range(horizon):
                    self._rnd[i][f][t] = random.random()

        done = []
        for f_i, (c_f, r_f) in enumerate(state.frogs):
            if f_i not in self._frogs: continue
            c_min = c_f - horizon
            c_max = c_f + horizon
            r_min = r_f - horizon
            r_max = r_f + horizon
            for c in range(c_min, c_max + 1):
                if 0 <= c < state.size:
                    for r in range(r_min, r_max + 1):
                        if 0 <= r < state.size:
                            dist = abs(c_f - c) + abs(r_f - r)
                            if (c, r) not in done and dist < horizon:
                                done.append((c, r))
                                # here new code
                                if (c, r) in self._lake_dict:
                                    for i, lake in enumerate(self._lake_dict[(c, r)]):
                                        #todo make this dynamic
                                        if i > 10: continue
                                        lines.append(f"lake_action({c}, {r}, {lake[0]}, {lake[2]}).")
                                        lines.append(f"lake_order({c}, {r}, {lake[0]}, {i}).")
                                is_wall = np.any(
                                    np.all(state.walls == [c, r], axis=1))
                                is_lake = np.any(
                                    np.all(state.lakes == [c, r], axis=1))
                                if not is_wall and not is_lake:
                                    pos_actions = []
                                    # Use the precomputed pos_actions array from state
                                    for i in range(4):
                                        if state.pos_actions[c, r, i] == 1:
                                            pos_actions.append(i)
                                    for i, action in enumerate(pos_actions):
                                        lines.append(f"act_pos({c}, {r}, {action}, {i}, {len(pos_actions)}).")

        # agent window
        c_min = state.agent[0] - horizon
        c_max = state.agent[0] + horizon
        r_min = state.agent[1] - horizon
        r_max = state.agent[1] + horizon
        for c in range(c_min, c_max + 1):
            for r in range(r_min, r_max + 1):
                dist = abs(state.agent[0] - c) + abs(state.agent[1] - r)
                if dist <= horizon:
                    if (c, r) in self._lake_dict:
                        for i, lake in enumerate(self._lake_dict[(c, r)]):
                            #todo make this dynamic
                            if i > 10: continue
                            lines.append(
                                f"lake_action({c}, {r}, {lake[0]}, {lake[2]}).")
                            lines.append(
                                f"lake_order({c}, {r}, {lake[0]}, {i}).")
                            lines.append(
                                f"lake_dist({c}, {r}, {lake[1]}, {lake[0]}).")
                    if (c, r) in self._grass_dict:
                        for i, grass in enumerate(self._grass_dict[(c, r)]):
                            # todo make this dynamic
                            if i > 10: continue
                            lines.append(
                                f"grass_order({c}, {r}, {grass[0]}, {i}).")
                            lines.append(
                                f"grass_dist({c}, {r}, {grass[1]}, {grass[0]}).")
                    is_wall = np.any(
                        np.all(state.walls == [c, r], axis=1))
                    is_lake = np.any(
                        np.all(state.lakes == [c, r], axis=1))
                    if is_wall:
                        lines.append(f"wall({c}, {r}).")
                    elif is_lake:
                        lines.append(f"lake({c}, {r}).")


        self._dynamic = "\n".join(lines)
        return "\n".join(lines)


    def call_clingo_generate(self, state, violations):
        lines = []
        dyn = self.build_dynamic(state)
        #for constraint in self._constraints:
        #    lines.append(constraint)
        if violations:
            # activate worlds here
            for i in violations:
                for f, (c,r) in enumerate(state.frogs):
                    if f not in self._frogs: continue
                    lines.append(
                        f"frog({c}, {r}, {f}, 0, {i}).")
                    for t in range(self._horizon):
                        ran = self._rnd[i][f][t]
                        lines.append(
                            f"f_rnd({f}, {t}, {i}, {int(ran * 100)}).")
        worlds = "\n".join(lines)
        self._latest_model = None
        with open("generate.lp", "r") as f:
            program = f.read()
        with open("common.lp", "r") as f:
            common = f.read()
        if self._ctd:
            with open("norms-ctd.lp", "r") as f:
                norms = f.read()
        else:
            with open("norms.lp", "r") as f:
                norms = f.read()
        self._generate = clingo.Control()
        self._generate.add("base", [], f"{self._static}\n{self._dynamic}\n{program}\n{common}\n{worlds}\n{dyn}\n{norms}")
        self._generate.ground([("base", [])], context=self)
        self._generate.solve(on_model=self.on_model)
        actions = [-1] * self._horizon
        #todo what if latest model is none
        #print(self._latest_model)
        for sym in self._latest_model:
            if sym.name == "action" and len(sym.arguments) == 2:
                actions[sym.arguments[1].number] = sym.arguments[0].number
        return actions

    def call_clingo_check(self, state, actions, exclude_worlds, n_rot, rot_count, sips):
        if rot_count != -1:
            max_world = rot_count * n_rot
        else:
            max_world = len(self._rnd)
        self._latest_model = None
        dyn = self.build_dynamic(state)
        with open("check.lp", "r") as f:
            program = f.read()
        with open("common.lp", "r") as f:
            common = f.read()
        if self._ctd:
            with open("norms-ctd.lp", "r") as f:
                norms = f.read()
        else:
            with open("norms.lp", "r") as f:
                norms = f.read()
        self._check = clingo.Control()
        self._check.add("base", [], f"{self._static}\n{program}\n{dyn}\n{common}\n{norms}")

        # add frogs and agent
        lines = []
        lines.append(f"agent({state.agent[0]}, {state.agent[1]}, 0).")

        for s in sips:
            lines.append(f"ctd({sips[s][0]}, {s}, {sips[s][1]}).")

        for i, a in enumerate(actions):
            lines.append(f"action({a}, {i}).")
            done = []
            for f_i, (c_f,r_f) in enumerate(state.frogs):
                if f_i not in self._frogs: continue
                c_min = c_f - self._horizon
                c_max = c_f + self._horizon
                r_min = r_f - self._horizon
                r_max = r_f + self._horizon
                for c in range(c_min, c_max + 1):
                    for r in range(r_min, r_max + 1):
                        dist = abs(c_f - c) + abs(r_f - r)
                        if (c,r) not in done and dist < self._horizon:
                            done.append((c,r))
                            # here new code
                            if (c, r) in self._lake_dict:
                                for j, lake in enumerate(
                                        self._lake_dict[(c, r)]):
                                    #todo make this dynamic
                                    if j > 10: continue
                                    lines.append(
                                        f"lake_action({c}, {r}, {lake[0]}, {lake[2]}).")
                                    lines.append(
                                        f"lake_order({c}, {r}, {lake[0]}, {j}).")

        for i in range(max_world):
            if i in exclude_worlds: continue
            for f, (c,r) in enumerate(state.frogs):
                if f not in self._frogs: continue
                lines.append(f"frog({c}, {r}, {f}, 0, {i}).")
                for t in range(self._horizon):
                    ran = self._rnd[i][f][t]
                    lines.append(f"f_rnd({f}, {t}, {i}, {int(ran * 100)}).")


        done = []
        for i_f, (c_f, r_f) in enumerate(state.frogs):
            if i_f not in self._frogs: continue
            c_min = c_f - self._horizon
            c_max = c_f + self._horizon
            r_min = r_f - self._horizon
            r_max = r_f + self._horizon
            for c in range(c_min, c_max + 1):
                if 0 <= c < state.size:
                    for r in range(r_min, r_max + 1):
                        if 0 <= r < state.size:
                            dist = abs(c_f - c) + abs(r_f - r)
                            if (c, r) not in done and dist < self._horizon:
                                done.append((c, r))
                                is_wall = np.any(
                                    np.all(state.walls == [c, r], axis=1))
                                is_lake = np.any(
                                    np.all(state.lakes == [c, r], axis=1))
                                if not is_wall and not is_lake:
                                    pos_actions = []
                                    # Use the precomputed pos_actions array from state
                                    for i in range(4):
                                        if state.pos_actions[c, r, i] == 1:
                                            pos_actions.append(i)
                                    for i, action in enumerate(pos_actions):
                                        lines.append(f"act_pos({c}, {r}, {action}, {i}, {len(pos_actions)}).")

        self._check.add("base", [], "\n".join(lines))
        self._check.ground([("base", [])], context=self)
        self._check.solve(on_model=self.on_model)
        violations = []
        rot = True
        #print(self._latest_model)
        for sym in self._latest_model:
            if sym.name == "norm_violation" and len(sym.arguments) == 1:
                world = sym.arguments[0].number
                violations.append(world)
                if world > max_world - n_rot:
                    rot = False
        return violations, rot

    def compute_reward_new(self, lawn, lake, dist_lawn, dist_lake):
        lawn = lawn.number
        lake = lake.number
        dist_lawn = dist_lawn.number
        dist_lake = dist_lake.number
        if dist_lake < 5 and dist_lake != 0:
            dist_lake = 1 - (dist_lake / 5)
        else:
            dist_lake = 0
        dist_lawn = 1 - (dist_lawn / (self._state.size * self._state.size))
        features = {
            "mows_lawn": lawn,
            "sips_lake": lake,
            "dist_lawn": dist_lawn,
            "dist_lake": dist_lake,
        }
        q_value = self._q_agent.getQValueFromFeatures(features)
        return clingo.Number(int(q_value * 10000))


    def on_model(self, m):
        self._latest_model = m.symbols(shown=True)