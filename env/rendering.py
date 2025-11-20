import pygame


class GardenerRenderer:
    def draw(self, state):
        if not hasattr(self, "screen"):
            pygame.init()
            self.window_size = 600
            self.cell_size = self.window_size // state.size
            self.screen = pygame.display.set_mode(
                (self.window_size, self.window_size))
        self.screen.fill((255, 255, 255))
        # Draw grid lines
        for i in range(state.size + 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (i * self.cell_size, 0),
                             (i * self.cell_size, self.window_size))
            pygame.draw.line(self.screen, (200, 200, 200),
                             (0, i * self.cell_size),
                             (self.window_size, i * self.cell_size))
        # Draw agent
        ax, ay = state.agent
        pygame.draw.rect(self.screen, (255, 0, 255),
                         pygame.Rect(ax * self.cell_size, ay * self.cell_size,
                                     self.cell_size, self.cell_size))

        # Draw grass patches
        for idx, (gx, gy) in enumerate(state.grass):
            color = (0, 100, 0) if idx == state.active_grass else (150, 100,
                                                                   50)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(gx * self.cell_size,
                                         gy * self.cell_size, self.cell_size,
                                         self.cell_size))

        # Draw frogs
        for fx, fy in state.frogs:
            pygame.draw.rect(self.screen, (0, 255, 0),
                             pygame.Rect(fx * self.cell_size,
                                         fy * self.cell_size, self.cell_size,
                                         self.cell_size))
        # Draw lakes
        for idx, (lx, ly) in enumerate(state.lakes):
            color = (0, 0, 255) if state.lake_full[idx] else (100, 100, 255)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(lx * self.cell_size,
                                         ly * self.cell_size, self.cell_size,
                                         self.cell_size))

        # Draw walls
        for wx, wy in state.walls:
            pygame.draw.rect(self.screen, (120, 120, 120),
                             pygame.Rect(wx * self.cell_size,
                                         wy * self.cell_size, self.cell_size,
                                         self.cell_size))

        pygame.display.flip()
        pygame.display.set_caption("GardenerEnv")
        pygame.event.pump()
