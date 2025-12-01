class ASPTransformer:
    """
    Transform a GardenerState into an ASP program with:
      - constant atoms   (walls, grass, lakes)
      - variable atoms   (agent, grass_timer, lake_timer)
    """

    def __init__(self):
        self.static = None

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

        print("\n".join(lines))
