"""
StateLibrary: bucketed on-disk storage for SimulationState objects.

States are grouped into buckets by an associated float value (e.g. a
violation probability in [0, 1]). Buckets are defined by an explicit list
of edges, so they don't have to be uniform width -- e.g. you can make
buckets much narrower near 0 and wider near 1. Each bucket is capped at a
maximum number of stored states. States are stored as individual pickle
files so that counting states in a bucket is a cheap directory listing,
not a full deserialize.

Layout on disk:

    root/
      bucket_0.0000_0.0100/
        3f2a1e7e....pkl
      bucket_0.0100_0.0500/
        ...
      bucket_0.5000_1.0000/
        ...

Each .pkl file contains {"state": SimulationState, "value": float, "extra": list[tuple]}.
"""

from __future__ import annotations

import bisect
import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class BucketKey:
    low: float
    high: float

    @property
    def dirname(self) -> str:
        return f"bucket_{self.low:.6f}_{self.high:.6f}"


@dataclass
class StoredEntry:
    """What you get back from load_bucket / load_range."""

    state: Any
    value: float
    actions: list[tuple]


def uniform_edges(
    value_min: float, value_max: float, bucket_width: float
) -> list[float]:
    """Evenly spaced edges, e.g. uniform_edges(0, 1, 0.25) -> [0, .25, .5, .75, 1.0]."""
    n = round((value_max - value_min) / bucket_width)
    return [value_min + i * bucket_width for i in range(n + 1)]


def custom_edges(*edges: float) -> list[float]:
    """Pass your own breakpoints explicitly, e.g. custom_edges(0, .01, .05, .1, .25, .5, 1.0)."""
    edges = sorted(edges)
    assert len(edges) >= 2, "need at least a low and high edge"
    return list(edges)


class StateLibrary:
    def __init__(
        self,
        root: str | Path,
        edges: Sequence[float] | None = None,
        bucket_width: float = 0.25,
        max_per_bucket: int = 1000,
        value_min: float = 0.0,
        value_max: float = 1.0,
    ):
        """
        root: directory the library lives in (created if missing)
        edges: explicit, strictly increasing bucket boundaries, e.g.
               [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0] for fine buckets near 0.
               If omitted, falls back to uniform buckets of `bucket_width`
               spanning [value_min, value_max].
        max_per_bucket: cap on stored states per bucket
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_per_bucket = max_per_bucket

        if edges is None:
            edges = uniform_edges(value_min, value_max, bucket_width)
        self.edges = sorted(edges)
        assert len(self.edges) >= 2, "need at least two edges to form one bucket"

    # ---------- bucket math ----------

    def _bucket_index(self, value: float) -> int:
        # bisect_right so a value exactly on an interior edge falls into the
        # *upper* bucket (i.e. buckets are [low, high)), matching the
        # right-open convention used everywhere below.
        idx = bisect.bisect_right(self.edges, value) - 1
        # clamp values at/over the top edge into the last bucket, and
        # anything below the bottom edge into the first bucket
        idx = max(0, min(idx, len(self.edges) - 2))
        return idx

    def _bucket_key(self, value: float) -> BucketKey:
        idx = self._bucket_index(value)
        return BucketKey(self.edges[idx], self.edges[idx + 1])

    def _bucket_dir(self, key: BucketKey, create: bool = False) -> Path:
        path = self.root / key.dirname
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _buckets_overlapping(self, low: float, high: float) -> Iterator[BucketKey]:
        idx_low = self._bucket_index(low)
        idx_high = self._bucket_index(high)
        for idx in range(idx_low, idx_high + 1):
            yield BucketKey(self.edges[idx], self.edges[idx + 1])

    # ---------- counting ----------

    def count(self, value: float) -> int:
        """Number of states currently stored in the bucket containing `value`."""
        key = self._bucket_key(value)
        d = self._bucket_dir(key)
        if not d.exists():
            return 0
        return sum(1 for _ in d.glob("*.pkl"))

    def is_full(self, value: float, cap: int | None = None) -> bool:
        cap = self.max_per_bucket if cap is None else cap
        return self.count(value) >= cap

    # ---------- saving ----------

    def try_save(
        self,
        state,
        value: float,
        actions: list[tuple],
        cap: int | None = None,
    ) -> bool:
        """
        Save `state` (plus an optional list of tuples in `extra`) tagged with
        `value`, if its bucket isn't already at capacity.
        Returns True if saved, False if the bucket was full (state discarded).
        """
        cap = self.max_per_bucket if cap is None else cap
        key = self._bucket_key(value)
        d = self._bucket_dir(key)

        # Recheck count right before writing to reduce (not eliminate) races
        # if you ever call this from multiple processes concurrently.
        current = sum(1 for _ in d.glob("*.pkl")) if d.exists() else 0
        if current >= cap:
            return False

        d = self._bucket_dir(key, create=True)
        fname = f"{uuid.uuid4().hex}.pkl"
        payload = {"state": state, "value": value, "actions": actions}
        with open(d / fname, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    # ---------- loading ----------

    def load_bucket(self, value: float) -> list[StoredEntry]:
        """Load all states whose bucket contains `value`."""
        key = self._bucket_key(value)
        d = self._bucket_dir(key)
        return self._load_dir(d)

    def load_range(self, low: float, high: float) -> list[StoredEntry]:
        """
        Load all states with stored value in [low, high], spanning as many
        buckets as needed. Filters on the exact stored value, so this works
        even if [low, high] doesn't align with bucket edges.
        """
        results: list[StoredEntry] = []
        for key in self._buckets_overlapping(low, high):
            d = self._bucket_dir(key)
            for entry in self._load_dir(d):
                if low <= entry.value <= high:
                    results.append(entry)
        return results

    @staticmethod
    def _load_dir(d: Path) -> list[StoredEntry]:
        if not d.exists():
            return []
        out = []
        for f in d.glob("*.pkl"):
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            out.append(
                StoredEntry(
                    state=data["state"], value=data["value"], actions=data["actions"]
                )
            )
        return out

    # ---------- introspection ----------

    def bucket_counts(self) -> dict[str, int]:
        """Count of states per existing bucket directory."""
        counts = {}
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name.startswith("bucket_"):
                counts[d.name] = sum(1 for _ in d.glob("*.pkl"))
        return counts
