import time

import clingo
import numpy as np
from clingo import Number, Function


def get_current_time_ms():
    return round(time.time() * 1000)


class ClingoHelperOld:
    def __init__(self, state, learning, horizon, radius):
        self.state = state
        self.learning = learning
        self.radius = radius
        self.horizon = horizon
        self.time_diff = []
        self.ctl = None
        self.cache = []

    def on_model(self, m):
        self.save_solution(m)
        show = " ".join([str(i) for i in m.symbols(shown=True)])
        #print("Answer:\n{}".format(show))

    def save_solution(self, m):
        self.player = []
        self.final = False
        for sym in m.symbols(shown=True):
            if sym.name == "action_taken" and len(sym.arguments) == 2:
                while len(self.player) < sym.arguments[1].number + 1:
                    self.player.append(0)
                self.player[sym.arguments[1].number] = sym.arguments[
                    0].number

    def setup(self):
        self.max_frogs = min(pow((self.radius * 2 + 1), 2) - 1,
                             len(self.state.frogs))
        self.ctl = clingo.Control(["-c", f"horizon={self.horizon}", "-c",
                                   f"frogs={self.max_frogs}", "-c",
                                   f"radius={self.radius}", "-c",
                                   f"size={self.state.size}"])
        self.ctl.load('env/old/program.lp')
        self.ctl.ground([("base", [])], context=self)

    def reset_clingo_externals(self):
        for i in range(1, self.state.size + 1):
            for j in range(1, self.state.size + 1):
                self.ctl.assign_external(Function("player", [
                    Number(i),
                    Number(j)
                ]), False)
        for f, (c,r) in enumerate(self.state.frogs):
            self.ctl.assign_external(Function("foutside", [
                Number(f)
            ]), False)
        for i in range(self.radius + 1):
            for f, (c,r) in enumerate(self.state.frogs):
                self.ctl.assign_external(Function("fcol", [
                    Number(f),
                    Number(i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("fcol", [
                    Number(f),
                    Number(-i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("frow", [
                    Number(f),
                    Number(i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("frow", [
                    Number(f),
                    Number(-i),
                    Number(0)
                ]), False)
            for j in range(self.radius + 1):
                self.ctl.assign_external(Function("wall", [
                    Number(i),
                    Number(j)
                ]), False)
                self.ctl.assign_external(Function("wall", [
                    Number(-i),
                    Number(j)
                ]), False)
                self.ctl.assign_external(Function("wall", [
                    Number(i),
                    Number(-j)
                ]), False)
                self.ctl.assign_external(Function("wall", [
                    Number(-i),
                    Number(-j)
                ]), False)
            for a in range(5):
                for r in range(5):
                    self.ctl.assign_external(Function("action", [
                        Number(a),
                        Number(r)
                    ]), False)

    def set_clingo_externals(self):
        midpoint = self.state.agent

        #todo we can only do this for the current action with this method
        actionValuePairs = self.learning.get_actionValuePairs(self.state)
        sortedActionValuePairs = sorted(actionValuePairs, key=lambda x: x[1],
                                        reverse=True)

        for i, actionValuePair in enumerate(sortedActionValuePairs):
            self.ctl.assign_external(Function("action",
                                              [Number(actionValuePair[0]),
                                               Number(4-i)]), True)
        #print(sortedActionValuePairs)

        # walls & plants
        for i in range(self.radius + 1):
            for j in range(self.radius + 1):
                absolut = (midpoint[0] + i, midpoint[1] + j)
                if np.any(np.all(self.state.walls == [absolut[0], absolut[1]], axis=1)):
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(j)]), True)
                if np.any(np.all(self.state.lakes == [absolut[0], absolut[1]], axis=1)):
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(j)]), True)
                if absolut[0] < 0 or absolut[1] < 0 or absolut[
                    0] >= self.state.size or absolut[
                    1] >= self.state.size:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(j)]), True)

                absolut = (midpoint[0] - i, midpoint[1] + j)
                if np.any(np.all(self.state.walls == [absolut[0], absolut[1]], axis=1)):
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(j)]), True)
                if np.any(np.all(self.state.lakes == [absolut[0], absolut[1]], axis=1)):
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(j)]), True)
                if absolut[0] < 0 or absolut[1] < 0 or absolut[
                    0] >= self.state.size or absolut[
                    1] >= self.state.size:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(j)]), True)

                absolut = (midpoint[0] + i, midpoint[1] - j)
                if np.any(np.all(self.state.walls == [absolut[0], absolut[1]], axis=1)):
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(-j)]), True)
                if np.any(np.all(self.state.lakes == [absolut[0], absolut[1]], axis=1)):
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(-j)]), True)
                if absolut[0] < 0 or absolut[1] < 0 or absolut[
                    0] >= self.state.size or absolut[
                    1] >= self.state.size:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(-j)]), True)

                absolut = (midpoint[0] - i, midpoint[1] - j)
                if np.any(np.all(self.state.walls == [absolut[0], absolut[1]], axis=1)):
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(-j)]), True)
                if np.any(np.all(self.state.lakes == [absolut[0], absolut[1]], axis=1)):
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(-j)]), True)
                if absolut[0] < 0 or absolut[1] < 0 or absolut[
                    0] >= self.state.size or absolut[
                    1] >= self.state.size:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(-j)]), True)

        # frogs
        count_frog = 0
        for f, (c, r) in enumerate(self.state.frogs):
            relative = (c - midpoint[0], r - midpoint[1])
            if abs(relative[0]) > self.radius or abs(
                    relative[
                        1]) > self.radius or self.state.dead_frogs[f]:
                #    self.ctl.assign_external(Function("foutside", [Number(c)]),
                #                             True)
                pass
            else:
                self.ctl.assign_external(
                    Function("fcol",
                             [Number(count_frog), Number(relative[1]),
                              Number(0)]),
                    True)
                self.ctl.assign_external(
                    Function("frow",
                             [Number(count_frog), Number(relative[0]),
                              Number(0)]),
                    True)
                count_frog += 1
        for i in range(count_frog, self.max_frogs):
            self.ctl.assign_external(Function("foutside", [Number(i)]),
                                     True)

    def get_action(self, state):
        self.state = state

        time_pre_compute = get_current_time_ms()

        self.player = []

        # Reset the clingo window
        self.reset_clingo_externals()

        # Set all externals of the currently considered window (see section
        # "Optimization-> Windowing" in the main document for more information)
        # Externals include cell information (walls, frogs, plants) as well as
        # information about policy preferences
        self.set_clingo_externals()

        # solve the LP
        self.ctl.solve(on_model=self.on_model)

        # measure the computation time
        time_post_compute = get_current_time_ms()
        time_diff = time_post_compute - time_pre_compute
        self.time_diff.append(time_diff)

        # cache further actions if desired and return current next action
        #print(self.player)
        action = self.player[0]
        return action
