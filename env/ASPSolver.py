import random

import clingo
import numpy as np
from clingo.symbol import Symbol

from env.simulation_state import SimulationState
from env.state import GardenerState


class ASPSolver:
    def __init__(self, ctd, horizon):
        self._latest_model = None
        self.horizon = horizon
        self.agent_movement = []
        self.agent_symbols = {}
        self.frog_movement = []
        self.f_rnd_symbols = {}
        # create instance
        self._check = clingo.Control()
        # add external files
        self._check.load("mss_check.lp")
        # if ctd:
        # self._check.load("norms-ctd.lp")
        # else:
        #    self._check.load("norms.lp")
        # self._generate = clingo.Control()

    def instantiate_level(self, state: GardenerState):
        lines = []

        # constants
        lines.append(f"#const size={state.size}.")
        lines.append(f"#const horizon={self.horizon}.")
        # lines.append(f"#const grass_respawn={state.grass_respawn}.")
        lines.append(f"#const lake_respawn={state.lake_respawn}.")
        lines.append("")

        # constant atoms: walls
        line = ""
        for i, (c, r) in enumerate(state.walls):
            line += f"wall({c}, {r})."
        lines.append(line)
        # constant atoms: lakes
        line = ""
        for i, (c, r) in enumerate(state.lakes):
            line += f"lake({c}, {r}, {i})."
        lines.append(line)
        # constant atoms: grass
        line = ""
        # for i, (c, r) in enumerate(state.grass):
        #    line += f"grass({c}, {r}, {i})."
        # lines.append(line)

        self._check.add("base", [], "\n".join(lines))
        self._check.ground([("base", [])], context=self)
        # self._generate.add("base", [], lines)

    def instantiate_state(self, state: GardenerState, strata):
        lines = []
        # adding agent and agent actions
        lines.append(f"agent({state.agent[0]}, {state.agent[1]}, 0).")

        # lake timer
        line = ""
        for i, c in enumerate(state.lake_timer):
            line += f"lake_timer({i}, {c}, 0)."
        lines.append(line)

        # adding frogs
        for f, (c, r) in enumerate(state.frogs):
            if state.capt_frogs[f] or state.dead_frogs[f]:
                continue
            for i in range(strata):
                lines.append(f"frog({c}, {r}, {f}, 0, {i}).")

        # adding frog_timer
        for f in range(len(state.frog_timer)):
            if state.frog_timer[f] > 0:
                for i in range(state.frog_timer[f]):
                    lines.append(f"frog_timer({f}, {i}).")
                    lines.append(f"ctd({f}, {i - 1}).")

        # adding windowed distance maps and preferred and possible frog actions
        done = []
        for i_f, (c_f, r_f) in enumerate(state.frogs):
            if state.capt_frogs[i_f] or state.dead_frogs[i_f]:
                continue
            c_min = c_f - self.horizon
            c_max = c_f + self.horizon
            r_min = r_f - self.horizon
            r_max = r_f + self.horizon
            for c in range(c_min, c_max + 1):
                if 0 <= c < state.size:
                    for r in range(r_min, r_max + 1):
                        if 0 <= r < state.size:
                            dist = abs(c_f - c) + abs(r_f - r)
                            if (c, r) not in done and dist < self.horizon:
                                done.append((c, r))
                                # append frog navigation help
                                if (c, r) in state.lake_dict:
                                    for j, lake in enumerate(state.lake_dict[(c, r)]):
                                        # todo make this dynamic
                                        if j > 10:
                                            continue
                                        lines.append(
                                            f"lake_action({c}, {r}, {lake[0]}, {lake[2]})."
                                        )
                                        lines.append(
                                            f"lake_order({c}, {r}, {lake[0]}, {j})."
                                        )

                                is_wall = np.any(np.all(state.walls == [c, r], axis=1))
                                is_lake = np.any(np.all(state.lakes == [c, r], axis=1))
                                if not is_wall and not is_lake:
                                    pos_actions = []
                                    # Use the precomputed pos_actions array from state
                                    for i in range(4):
                                        if state.pos_actions[c, r, i] == 1:
                                            pos_actions.append(i)
                                    for i, action in enumerate(pos_actions):
                                        lines.append(
                                            f"act_pos({c}, {r}, {action}, {i}, {
                                                len(pos_actions)
                                            })."
                                        )
        self._check.add("state", [], "\n".join(lines))
        self._check.ground([("state", [])], context=self)

    def prepare_agent_movement(self):
        with self._check.backend() as backend:
            for t in range(self.horizon):
                actions = []
                for a in range(5):
                    sym = clingo.Function(
                        "action",
                        [clingo.Number(a), clingo.Number(t)],
                    )

                    atom_id = backend.add_atom(sym)
                    backend.add_external(atom_id, clingo.TruthValue.False_)
                    actions.append(sym)

                self.agent_symbols[t] = actions

        self._check.ground([("agent_movement", [])], context=self)

    def instantiate_agent_movement(self, actions):
        for t, a in enumerate(actions):
            sym = self.agent_symbols[t][a]

            self._check.assign_external(sym, True)
            self.agent_movement.append(sym)

    def clear_agent_movement(self):
        for i in self.agent_movement:
            self._check.assign_external(i, False)

        self.agent_movement = []

    def prepare_frog_action(self, state, strata):
        with self._check.backend() as backend:
            for f in range(len(state.frogs)):
                if state.dead_frogs[f] or state.capt_frogs[f]:
                    continue
                for t in range(self.horizon):
                    for i in range(strata):
                        for j in range(5):
                            sym = clingo.Function(
                                "f_action",
                                [
                                    clingo.Number(j),
                                    clingo.Number(f),
                                    clingo.Number(t),
                                    clingo.Number(i),
                                ],
                            )
                            atom_id = backend.add_atom(sym)
                            backend.add_external(atom_id, clingo.TruthValue.False_)
                            self.f_rnd_symbols[(f, t, i, j)] = sym

        self._check.ground([("frog_movement", [])], context=self)

    def instantiate_frog_actions(self, state: SimulationState, strata):
        for i, f in enumerate(state.frogs):
            if state.dead_frogs[i] or state.captured_frogs[i]:
                continue
            pref, other = state.get_frog_actions(f)
            for t in range(self.horizon):
                for w in range(strata):
                    ran = (w + random.random()) / strata

                    r = int(ran * 100)
                    if pref is None:
                        action = other[int(ran * len(other))]
                    elif r < 70:
                        action = pref
                    else:
                        idx = int((r - 70) / (30.0 / len(other)))
                        action = other[idx]

                    sym = self.f_rnd_symbols[(i, t, w, action)]
                    self._check.assign_external(sym, True)
                    self.frog_movement.append(sym)

    def prepare_frog_movement(self, state, strata):
        with self._check.backend() as backend:
            for f in range(len(state.frogs)):
                if state.dead_frogs[f] or state.capt_frogs[f]:
                    continue
                for t in range(self.horizon):
                    for i in range(strata):
                        for j in range(100):
                            sym = clingo.Function(
                                "f_rnd",
                                [
                                    clingo.Number(f),
                                    clingo.Number(t),
                                    clingo.Number(i),
                                    clingo.Number(j),
                                ],
                            )

                            atom_id = backend.add_atom(sym)
                            backend.add_external(atom_id, clingo.TruthValue.False_)
                            self.f_rnd_symbols[(f, t, i, j)] = sym

        self._check.ground([("frog_movement", [])], context=self)

    def instantiate_frog_movement(self, state, strata):
        for f in range(len(state.frogs)):
            if state.dead_frogs[f] or state.captured_frogs[f]:
                continue
            for t in range(self.horizon):
                for w in range(strata):
                    ran = (w + random.random()) / strata
                    r = int(ran * 100)
                    sym: Symbol = self.f_rnd_symbols[(f, t, w, r)]
                    self._check.assign_external(sym, True)
                    self.frog_movement.append(sym)

    def clear_frog_movement(self):
        for i in self.frog_movement:
            self._check.assign_external(i, False)

        self.frog_movement = []

    def check(self):
        self._latest_model = None
        self._check.solve(on_model=self.on_model)
        # print(self._latest_model)
        violations = []
        lines = []
        for sym in self._latest_model:
            if sym.name == "norm_violation" and len(sym.arguments) == 1:
                world = sym.arguments[0].number
                violations.append(world)

        for sym in self.frog_movement:
            if sym.arguments[3].number in violations:
                lines.append(sym)

        return lines

    def generate(self):
        pass

    def on_model(self, m):
        self._latest_model = m.symbols(shown=True)
