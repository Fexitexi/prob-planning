import numpy as np

from env import dynamics
from env.state import GardenerState


class SimulationCache:
    __slots__ = (
        "indexAgent",
        "indexCaptFrogs",
        "indexDeadFrogs",
        "indexFrogTimer",
        "indexFrogs",
        "indexLakeTimer",
        "indexViolation",
        "numberOfFrogs",
        "numberOfLakes",
    )

    def __init__(self, env: GardenerState):
        self.numberOfFrogs = len(env.frogs)
        self.numberOfLakes = len(env.lakes)
        self.indexAgent = 0
        self.indexFrogs = 2
        self.indexDeadFrogs = 2 + 2 * self.numberOfFrogs
        self.indexCaptFrogs = 2 + 3 * self.numberOfFrogs
        self.indexFrogTimer = 2 + 4 * self.numberOfFrogs
        self.indexLakeTimer = 2 + 5 * self.numberOfFrogs
        self.indexViolation = 2 + 5 * self.numberOfFrogs + self.numberOfLakes


class SimulationState:
    __slots__ = ("_hash", "_key", "arr", "cache", "env")

    def __init__(self, arr, cache, env):
        self.arr = arr
        self.cache = cache
        self.env: GardenerState = env
        self._key = None
        self._hash = None

    @property
    def key(self):
        k = self._key
        if k is None:
            k = self._key = self.arr.tobytes()
        return k

    def __hash__(self):
        h = self._hash
        if h is None:
            h = self._hash = hash(self.key)
        return h

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, SimulationState):
            return NotImplemented
        return self.key == other.key

    def get_possible_actions_with_probabilities(self, arr, agentAction, horizon):
        # Convert the agent action to a hashable tuple.
        possible = {(agentAction,): 1.0}

        for i in range(self.cache.numberOfFrogs):
            new_possible = {}
            x_f = arr[2 + 2 * i]
            y_f = arr[3 + 2 * i]
            dist = abs(arr[0] - x_f) + abs(arr[1] - y_f)
            # skip stunned frogs
            if (
                arr[self.cache.indexFrogTimer + i] > 0
                or arr[self.cache.indexCaptFrogs + i]
                or arr[self.cache.indexDeadFrogs + i]
                or dist > 2 * horizon
            ):
                possible = {k + (4,): v for k, v in possible.items()}
                continue

            valid_moves = self.env.pos_actions[(x_f, y_f)][:4]
            preferred_action = None
            if (x_f, y_f) in self.env.lake_dict:
                for lake_idx, dist, action in self.env.lake_dict[(x_f, y_f)]:
                    if arr[self.cache.indexLakeTimer + lake_idx] == 0 and (
                        action == 4 or valid_moves[action] == 1
                    ):
                        if action != 4:
                            preferred_action = action
                        break

            numberOfValidMoves = sum(valid_moves)
            if preferred_action is not None and numberOfValidMoves > 1:
                for idx, i in enumerate(valid_moves):
                    if i == 0:
                        continue
                    if idx == preferred_action:
                        for action_seq, prob in possible.items():
                            new_possible[action_seq + (idx,)] = prob * 0.7
                    else:
                        for action_seq, prob in possible.items():
                            new_possible[action_seq + (idx,)] = prob * (
                                0.3 / (numberOfValidMoves - 1)
                            )
            elif preferred_action is None and numberOfValidMoves > 0:
                for idx, i in enumerate(valid_moves):
                    if i == 0:
                        continue
                    for action_seq, prob in possible.items():
                        new_possible[action_seq + (idx,)] = prob * (
                            1.0 / numberOfValidMoves
                        )
            elif preferred_action is not None and numberOfValidMoves <= 1:
                for action_seq, prob in possible.items():
                    new_possible[action_seq + (preferred_action,)] = prob
            else:
                # frog is unable to move
                for action_seq, prob in possible.items():
                    new_possible[action_seq + (4,)] = prob

            possible = new_possible

        return possible

    def apply_action(self, arr, action):
        violation = False
        result_arr = arr.copy()
        directions = dynamics._action_to_direction

        for i, a in enumerate(action):
            a = int(a)
            result_arr[2 * i] += directions[a][0]
            result_arr[(2 * i) + 1] += directions[a][1]

        for i in range(self.cache.numberOfFrogs):
            if (
                result_arr[self.cache.indexDeadFrogs + i] == 1
                or result_arr[self.cache.indexCaptFrogs + i] == 1
            ):
                continue
            captured_now = False

            if (
                result_arr[2 + (2 * i)] == result_arr[0]
                and result_arr[3 + (2 * i)] == result_arr[1]
            ):
                if result_arr[self.cache.indexFrogTimer + i] > 0:
                    captured_now = True
                    result_arr[self.cache.indexCaptFrogs + i] = 1
                else:
                    violation = True
                    result_arr[self.cache.indexDeadFrogs + i] = 1

            if result_arr[self.cache.indexFrogTimer + i] == 1 and not captured_now:
                violation = True

            result_arr[self.cache.indexFrogTimer + i] = max(
                result_arr[self.cache.indexFrogTimer + i] - 1, 0
            )

        agent_empties = np.zeros(self.cache.numberOfLakes, dtype=np.bool_)
        for i in range(self.cache.numberOfLakes):
            lake_timer = max(result_arr[self.cache.indexLakeTimer + i] - 1, 0)

            agent_dist = abs(result_arr[0] - self.env.lakes[i, 0]) + abs(
                result_arr[1] - self.env.lakes[i, 1]
            )

            if agent_dist == 1 and lake_timer == 0:
                agent_empties[i] = True
                lake_timer = self.env.lake_respawn

            result_arr[self.cache.indexLakeTimer + i] = lake_timer

        for i in range(self.cache.numberOfFrogs):
            if (
                result_arr[self.cache.indexDeadFrogs + i] == 1
                or result_arr[self.cache.indexCaptFrogs + i] == 1
            ):
                continue

            for j in range(self.cache.numberOfLakes):
                if agent_empties[j]:
                    dx = abs(result_arr[2 + 2 * i] - self.env.lakes[j, 0])
                    dy = abs(result_arr[3 + 2 * i] - self.env.lakes[j, 1])
                    if max(dx, dy) == 1:
                        result_arr[self.cache.indexFrogTimer + i] = 5
                        break

        result_arr[-1] = 1 if violation else 0

        return result_arr

    def calculate_violation_probability_brute_force(
        self, arr, agentActions: list, depth
    ):
        violation_probability = 0.0
        # short circuit if an early violation happens
        if arr[-1] == 1:
            return 1.0

        # return 0 if a leaf node (and no violation due to the if above)
        if depth == 0:
            for i in range(self.cache.numberOfFrogs):
                if (
                    arr[self.cache.indexFrogTimer + i] > 0
                    and not arr[self.cache.indexCaptFrogs + i]
                ):
                    return 1.0
            return 0.0

        for a, p in self.get_possible_actions_with_probabilities(
            arr, agentActions[0], depth
        ).items():
            next_arr = self.apply_action(arr, a)
            violation_probability += (
                p
                * self.calculate_violation_probability_brute_force(
                    next_arr, agentActions[1:], depth - 1
                )
            )
        return violation_probability

    @staticmethod
    def from_state(state: GardenerState):
        # only call at the first step, otherwise use fast_clone or apply_action
        arr = np.concatenate(
            [
                state.agent,
                state.frogs.ravel(),
                state.dead_frogs,
                state.capt_frogs,
                state.frog_timer,
                np.where(
                    state.lakes_full,
                    0,
                    np.maximum(state.lake_timer, 1),
                ),
                [0],
            ]
        ).astype(np.int16)
        return SimulationState(arr, SimulationCache(state), state)

    def fast_clone(self):
        return SimulationState(self.arr.copy(), self.cache, self.env)
