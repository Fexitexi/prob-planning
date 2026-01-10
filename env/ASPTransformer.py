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
        self._dyn_lake_dict = None
        self._constraints = []
        self._check = None
        self._generate = None
        self._horizon = None

    def reset(self):
        self._state = None
        self._rnd = None
        self._dyn_lake_dict = None
        self._latest_model = None
        self._constraints.clear()
        self._check = None
        self._generate = None
        self._dynamic = None


    def build_static(self, state, horizon) -> str:
        lines = []

        self._lake_dict = {}
        # build the lake dict

        lake_dist = []
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
                            best[nx, ny] = np.array([-dx, -dy], dtype=np.int8)
                            q.append((nx, ny))

            lake_dist.append(dist)
            lake_best_step.append(best)

        for c in range(state.size):
            for r in range(state.size):
                if (c, r) in walls_set or (c, r) in lakes_set:
                    continue
                lakes_info = []
                for i in range(len(state.lakes)):
                    dist = lake_dist[i][c, r]
                    step = lake_best_step[i][c, r]

                    action = 4
                    if step[0] == 1 and step[1] == 0:
                        action = 0
                    elif step[0] == 0 and step[1] == 1:
                        action = 1
                    elif step[0] == -1 and step[1] == 0:
                        action = 2
                    elif step[0] == 0 and step[1] == -1:
                        action = 3

                    lakes_info.append((i, dist, action))

                lakes_info.sort(key=lambda x: x[1])
                self._lake_dict[(c, r)] = lakes_info

        # constants
        lines.append(f"#const size={state.size}.")
        lines.append(f"#const horizon={horizon}.")
        lines.append(f"#const grass_respawn={state.grass_respawn}.")
        lines.append(f"#const lake_respawn={state.lake_respawn}.")
        lines.append("")
        self._horizon = horizon

        # encode possible actions of frogs
        line = ""

        # moved to dynamic
        #done = []
        #for (c_f,r_f) in state.frogs:
        #    c_min = c_f - horizon
        #    c_max = c_f + horizon
        #    r_min = r_f - horizon
        #    r_max = r_f + horizon
        #    for c in range(c_min, c_max + 1):
        #        if 0 <= c < state.size:
        #            for r in range(r_min, r_max + 1):
        #                if 0 <= r < state.size:
        #                    dist = abs(c_f - c) + abs(r_f - r)
        #                    if (c,r) not in done and dist <= horizon:
        #                        done.append((c,r))
        #                        is_wall = np.any(np.all(state.walls == [c, r], axis=1))
        #                        is_lake = np.any(np.all(state.lakes == [c, r], axis=1))
        #                        if not is_wall and not is_lake:
        #                            pos_actions = []
        #                            # Use the precomputed pos_actions array from state
        #                            for i in range(4):
        #                                if state.pos_actions[c, r, i] == 1:
        #                                    pos_actions.append(i)
        #                            for i, action in enumerate(pos_actions):
        #                                line += f"act_pos({c}, {r}, {action}, {i}, {len(pos_actions)})."
        #lines.append(line)

        ## constant atoms: lakes
        #line = ""
        #for i, (c, r) in enumerate(state.lakes):
        #    line += f"lake({c}, {r}, {i})."
        #lines.append(line)
        ## constant atoms: grass
        #line = ""
        #for i, (c, r) in enumerate(state.grass):
        #    line += f"grass({c}, {r}, {i})."
        #lines.append(line)
        ## constant atoms: walls
        #line = ""
        #for (c, r) in state.walls:
        #    line += f"wall({c}, {r})."
        #lines.append(line)

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
            line += f"lake_timer({i}, {c}, 0)."
        lines.append(line)

        # grass timer
        line = ""
        for i, c in enumerate(state.grass_timer):
            line += f"grass_timer({i}, {c}, 0)."
        lines.append(line)

        return "\n".join(lines)

    def build_dynamic_worlds(self, state, num_worlds, horizon, append_lines=True) -> str:
        self._state = state
        lines = []

        # agent position
        lines.append(f"agent({state.agent[0]}, {state.agent[1]}, 0).")

        self._dyn_lake_dict = {}

        words = ["1"]
        final_words = []
        for h in range(horizon):
            new_words = []
            for w in words:
                for a in range(5):
                    word = f"{w}{a}"
                    new_words.append(word)
            words = new_words
            final_words.extend(new_words)

        for w in final_words:
            w = w.removeprefix("1")
            copy_state = state.fast_clone()
            valid = True
            for a in w:
                try:
                    self._dynamics.move_agent(copy_state, int(a))
                except:
                    valid = False
                    break
                # todo this should be in dynamics
                for i, (gx, gy) in enumerate(copy_state.grass):
                    ax, ay = copy_state.agent
                    if ax == gx and ay == gy:
                        if copy_state.grass_active[i]:
                            copy_state.grass_active[i] = False
                            copy_state.grass_timer[i] = copy_state.grass_respawn
                    else:
                        if not copy_state.grass_active[i] and copy_state.grass_timer[
                            i] > 0:
                            copy_state.grass_timer[i] -= 1
                            if copy_state.grass_timer[i] == 0:
                                copy_state.grass_active[i] = True

                # Update lake states based on frog adjacency
                for i, (lx, ly) in enumerate(copy_state.lakes):
                    # Decrease timer if running
                    if copy_state.lake_timer[i] > 0:
                        copy_state.lake_timer[i] -= 1
                        if copy_state.lake_timer[i] == 0:
                            copy_state.lakes_full[i] = True  # refill lake


                    # Check adjacency to the agent (Manhattan distance 1)
                    ax, ay = copy_state.agent
                    if copy_state.lakes_full[i] and abs(ax - lx) + abs(
                            ay - ly) == 1:
                        # Additional reward for being adjacent (Manhattan distance 1) to any full lake
                        copy_state.lakes_full[i] = False
                        copy_state.lake_timer[i] = copy_state.lake_respawn
            if valid:
                self._dyn_lake_dict[f"1{w}"] = copy_state.lakes_full

        for h in self._dyn_lake_dict:
            full_lakes = self._dyn_lake_dict[h]
            done = []
            for (c_f,r_f) in state.frogs:
                c_min = c_f - horizon
                c_max = c_f + horizon
                r_min = r_f - horizon
                r_max = r_f + horizon
                for c in range(c_min, c_max + 1):
                    for r in range(r_min, r_max + 1):
                        dist = abs(c_f - c) + abs(r_f - r)
                        if (c,r) not in done and dist < horizon:
                            done.append((c,r))
                            if (c, r) in self._lake_dict:
                                lakes = self._lake_dict[(c, r)]
                                lake = None
                                for i in lakes:
                                    if full_lakes[i[0]]:
                                        lake = i
                                        break
                                if lake is not None:
                                    lines.append(
                                        f"pref_act({c},{r},{h},{lake[2]}).")
                                else:
                                    lines.append(
                                        f"pref_act({c},{r},{h},{-1}).")

        self._rnd = []
        # frogs
        for i in range(num_worlds):
            random_world = []
            for f, (c,r) in enumerate(state.frogs):
                random_frog = []
                if append_lines: lines.append(f"frog({c}, {r}, {f}, 0, {i}).")
                for t in range(horizon):
                    ran = random.random()
                    if append_lines: lines.append(f"f_rnd({f}, {t}, {i}, {int(ran * 100)}).")
                    random_frog.append(ran)
                random_world.append(random_frog)
            self._rnd.append(random_world)

        done = []
        for (c_f, r_f) in state.frogs:
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

    def build_worlds(self, worlds) -> str:
        lines = []

        for w, world in enumerate(worlds):
            line = ""
            for t, state in enumerate(world):
                for i, (c,r) in enumerate(state.frogs):
                    line += f"frog({c}, {r}, {i}, {t}, {w})."
            lines.append(line)

        return "\n".join(lines)

    def call_clingo_generate(self, state, violations):
        lines = []
        for constraint in self._constraints:
            lines.append(constraint)
        if violations:
            # activate worlds here
            for i in violations:
                for f, (c,r) in enumerate(state.frogs):
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
        self._generate = clingo.Control()
        self._generate.add("base", [], f"{self._static}\n{self._dynamic}\n{program}\n{worlds}")
        self._generate.ground([("base", [])], context=self)
        self._generate.solve(on_model=self.on_model)
        actions = [-1] * self._horizon
        #todo what if latest model is none
        for sym in self._latest_model:
            if sym.name == "action" and len(sym.arguments) == 2:
                actions[sym.arguments[1].number] = sym.arguments[0].number
        return actions

    def call_clingo_check(self, state, actions, exclude_worlds):
        self._latest_model = None
        with open("check.lp", "r") as f:
            program = f.read()
        self._check = clingo.Control()
        self._check.add("base", [], f"{self._static}\n{program}")

        # add frogs and agent
        lines = []
        lines.append(f"agent({state.agent[0]}, {state.agent[1]}, 0).")

        h = "1"
        for i, a in enumerate(actions):
            lines.append(f"action({a}, {i}).")
            h += str(a)
            full_lakes = self._dyn_lake_dict[h]
            for (c, r) in self._lake_dict:
                lakes = self._lake_dict[(c,r)]
                lake = None
                for j in lakes:
                    if full_lakes[j[0]]:
                        lake = j
                        break
                if lake is not None:
                    lines.append(f"pref_act({c},{r},{i},{lake[2]}).")
                else:
                    lines.append(f"pref_act({c},{r},{i},{-1}).")

        for i in range(len(self._rnd)):
            if i in exclude_worlds: continue
            for f, (c,r) in enumerate(state.frogs):
                lines.append(f"frog({c}, {r}, {f}, 0, {i}).")
                for t in range(self._horizon):
                    ran = self._rnd[i][f][t]
                    lines.append(f"f_rnd({f}, {t}, {i}, {int(ran * 100)}).")


        done = []
        for (c_f, r_f) in state.frogs:
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
        #print(self._latest_model)
        for sym in self._latest_model:
            if sym.name == "norm_violation" and len(sym.arguments) == 1:
                violations.append(sym.arguments[0].number)
        return violations


    def call_clingo_new(self, static, dynamic, horizon):
        self._latest_model = None
        with open("fixed-logic.lp", "r") as f:
            fixed_program = f.read()
        ctl = clingo.Control()
        ctl.add("base", [], f"{static}\n{dynamic}\n{fixed_program}")
        ctl.ground([("base", [])], context=self)
        ctl.solve(on_model=self.on_model)
        actions = [-1] * horizon
        for sym in self._latest_model:
            if sym.name == "action" and len(sym.arguments) == 2:
                actions[sym.arguments[1].number] = sym.arguments[0].number
        return actions

    def call_clingo(self, static, dynamic, worlds, horizon):
        start_time = time.time()
        self._latest_model = None
        with open("fixed.lp", "r") as f:
            fixed_program = f.read()
        ctl = clingo.Control()

        ctl.add("base", [], f"{static}\n{dynamic}\n{worlds}\n{fixed_program}")
        ctl.ground([("base", [])], context=self)
        ctl.solve(on_model=self.on_model)
        actions = [-1] * horizon
        for sym in self._latest_model:
            if sym.name == "action" and len(sym.arguments) == 2:
                actions[sym.arguments[1].number] = sym.arguments[0].number
        #print(f"First action: {first_action}")

        elapsed = time.time() - start_time
        #print(f"clingo took {elapsed:.6f} seconds")
        return actions

    def compute_reward(self, h):
        actions = []
        history = h.number
        while history > 0:
            actions.append(history % 10)
            history //= 10
        state = self._state.fast_clone()
        actions.pop()
        success = True
        while len(actions) > 1:
            action = actions.pop()
            try:
                self._dynamics.move_agent(state, action)
            except:
                success = False
                break
        if success:
            try:
                value = self._q_agent.getQValue(state, actions[0])
                return clingo.Number(int(value * 10000))
            except:
                return clingo.Number(0)
        else:
            return clingo.Number(0)

    def on_model(self, m):
        self._latest_model = m.symbols(shown=True)