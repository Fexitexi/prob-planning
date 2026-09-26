"""Typed, immutable configuration objects for the Gardener benchmark runner."""

import math
from dataclasses import dataclass, field
from enum import IntEnum


class SamplingMode(IntEnum):
    """World-sampling strategy used by the new ASP-based planner."""

    RANDOM = 0
    SEQUENTIAL = 1
    MCTS = 2


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Parameters controlling sampling / statistical guarantees.

    The ``epsilon`` and ``delta`` fields are used for every sampling mode
    because they determine the required sample sizes.  ``strata`` and
    ``indifference`` are only relevant for stratified sampling.

    ``confidence`` and ``max_visits`` are reserved for future MCTS support
    and are not currently exposed on the CLI.
    """

    mode: SamplingMode = SamplingMode.RANDOM
    epsilon: float = 0.05
    delta: float = 0.05
    strata: int = 1
    indifference: float = 0.005
    confidence: float = 0.95
    max_visits: int = 1000

    @property
    def n_rot(self) -> int:
        """Number of worlds required for the Rule-of-Three check."""
        return math.ceil(math.log(self.delta) / math.log(1.0 - self.epsilon))

    def n_asp(self, horizon: int, n_actions: int = 5) -> int:
        """Number of worlds required for ASP-Gen / dynamic world building."""
        return math.ceil(
            (1.0 / (2.0 * self.epsilon**2))
            * math.log((2.0 * n_actions**horizon) / self.delta)
        )

    def __post_init__(self):
        if not (0.0 < self.epsilon < 1.0):
            raise ValueError("epsilon must be in (0, 1)")
        if not (0.0 < self.delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        if self.strata < 1:
            raise ValueError("strata must be at least 1")
        if self.indifference <= 0.0:
            raise ValueError("indifference must be positive")


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level configuration for one benchmark run."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    horizon: int = 3
    size: int = 15
    seed: int = 42
    rounds: int = 10
    ctd: bool = False
    render: bool = False
    logLevel: int = 0

    @classmethod
    def from_args(cls, args) -> "Config":
        return cls(
            sampling=SamplingConfig(
                mode=SamplingMode(args.sampling),
                epsilon=args.epsilon,
                delta=args.delta,
                indifference=args.indifference,
            ),
            horizon=args.horizon,
            size=args.size,
            seed=args.seed,
            rounds=args.rounds,
            ctd=bool(args.ctd),
            render=bool(args.render),
            logLevel=args.logLevel,
        )
