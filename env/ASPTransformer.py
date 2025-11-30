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
        lines.append("")

        # constant atoms: walls
        for (c, r) in state.walls:
            lines.append(f"wall({c}, {r}).")

        # constant atoms: lakes
        for i, (c, r) in enumerate(state.lakes):
            lines.append(f"lake({c}, {r}, {i}).")

        # constant atoms: grass
        for i, (c, r) in enumerate(state.grass):
            lines.append(f"grass({c}, {r}, {i}).")

        self.static = "\n".join(lines)
        print(self.static)
        return "\n".join(lines)