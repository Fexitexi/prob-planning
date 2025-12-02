import clingo

from env.dynamics import GardenerDynamics


class ASPTransformer:
    """
    Transform a GardenerState into an ASP program with:
      - constant atoms   (walls, grass, lakes)
      - variable atoms   (agent, grass_timer, lake_timer)
    """

    def __init__(self, q_agent):
        self._static = None
        self._state = None
        self._dynamics = GardenerDynamics()
        self._q_agent = q_agent
        self._latest_model = None

    def build_static(self, state, horizon) -> str:
        lines = []

        # constants
        lines.append(f"#const size={state.size}.")
        lines.append(f"#const horizon={horizon}.")
        lines.append(f"#const grass_respawn={state.grass_respawn}.")
        lines.append(f"#const lake_respawn={state.lake_respawn}.")
        lines.append("")

        # constant atoms: walls
        line = ""
        for (c, r) in state.walls:
            line += f"wall({c}, {r})."
        lines.append(line)

        # constant atoms: lakes
        line = ""
        for i, (c, r) in enumerate(state.lakes):
            line += f"lake({c}, {r}, {i})."
        lines.append(line)

        # constant atoms: grass
        line = ""
        for i, (c, r) in enumerate(state.grass):
            line += f"grass({c}, {r}, {i})."
        lines.append(line)

        self.static = "\n".join(lines)
        return "\n".join(lines)

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

    def build_worlds(self, worlds) -> str:
        lines = []

        for w, world in enumerate(worlds):
            line = ""
            for t, state in enumerate(world):
                for i, (c,r) in enumerate(state.frogs):
                    line += f"frog({c}, {r}, {i}, {t}, {w})."
            lines.append(line)

        return "\n".join(lines)


    def call_clingo(self, static, dynamic, worlds):
        self._latest_model = None
        with open("fixed.lp", "r") as f:
            fixed_program = f.read()
        ctl = clingo.Control()

        ctl.add("base", [], f"{static}\n{dynamic}\n{worlds}\n{fixed_program}\n")
        ctl.ground([("base", [])], context=self)
        ctl.solve(on_model=self.on_model)
        print(self._latest_model)
        first_action = None
        for sym in self._latest_model:
            if sym.name == "action" and len(sym.arguments) == 2:
                if sym.arguments[1].number == 0:
                    first_action = sym.arguments[0].number
                    break
        #print(f"First action: {first_action}")
        return first_action

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