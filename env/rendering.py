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

        # Draw grass patches
        for idx, (gx, gy) in enumerate(state.grass):
            color = (0, 100, 0) if state.grass_active[idx] else (150, 100, 50)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(gx * self.cell_size,
                                         gy * self.cell_size, self.cell_size,
                                         self.cell_size))
        # Draw lakes
        for idx, (lx, ly) in enumerate(state.lakes):
            color = (0, 0, 255) if state.lakes_full[idx] else (100, 100, 255)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(lx * self.cell_size,
                                         ly * self.cell_size, self.cell_size,
                                         self.cell_size))

        # Draw frogs
        for i, (fx, fy) in enumerate(state.frogs):
            color = (0, 255, 0)
            if state.capt_frogs[i]: color = (255, 255, 0)
            elif state.dead_frogs[i]: color = (255, 0, 0)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(fx * self.cell_size,
                                         fy * self.cell_size, self.cell_size,
                                         self.cell_size))

        # Draw walls
        for wx, wy in state.walls:
            pygame.draw.rect(self.screen, (120, 120, 120),
                             pygame.Rect(wx * self.cell_size,
                                         wy * self.cell_size, self.cell_size,
                                         self.cell_size))
        # Draw agent
        ax, ay = state.agent
        agent_size = int(self.cell_size * 0.6)
        offset = (self.cell_size - agent_size) // 2
        pygame.draw.rect(self.screen, (255, 0, 255),
                         pygame.Rect(ax * self.cell_size + offset, ay * self.cell_size + offset,
                                     agent_size, agent_size))

        # Draw score
        if not hasattr(self, "font"):
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)
        score_surf = self.font.render(f"Score: {state.score}", True, (0, 0, 0))
        self.screen.blit(score_surf, (5, 5))

        pygame.display.flip()
        pygame.display.set_caption("GardenerEnv")
        pygame.event.pump()
