# ASP-Based Probabilistic Policy Fixing for Norm-Compliant RL

Reference implementation and benchmark suite for the paper *ASP-Based Probabilistic Policy Fixing for Norm Compliant RL*.

The framework adapts a norm-agnostic, reward-optimizing reinforcement learning policy online by interjecting an ASP-based "emergency planner" whenever the agent risks violating a deontic norm. Norm-violation probabilities are estimated via sampled world evolutions, and statistical guarantees on near-optimality are provided through the Rule of Three and Hoeffding's inequality.

## Approach

Two ASP programs interact with a Q-learning policy in a generate-and-check loop (see `Main.py`):

- **RoT-Check** (`check.lp`) — given a fixed candidate policy, simulates `n` sampled world evolutions and reports any in which the policy violates a norm.
- **ASP-Gen** (`generate.lp`) — given the bad worlds, searches for a new policy that minimizes norm violations while staying as close as possible to the RL policy's preferences.

Sample sizes `n` and `N` are derived from the user-specified confidence `1 − δ` and error tolerance `ϵ`. Norms are encoded as ASP rules in `norms.lp` (maintenance) and `norms-ctd.lp` (contrary-to-duty).

## Repository Layout

| Path | Purpose |
|------|---------|
| `Main.py` | Entry point — runs the policy-fixing loop on the Gardener environment |
| `check.lp` | RoT-Check ASP program |
| `generate.lp` | ASP-Gen ASP program |
| `common.lp` | Shared domain rules |
| `norms.lp`, `norms-ctd.lp` | Norm encodings (MTN1, MTN2, CTD) |
| `env/` | Gardener gym environment, Q-learning agent, ASP transformer |
| `weights.pkl` | Pre-trained Q-learning weights |
| `bench_config.json`, `configs.txt`, `instances.txt` | Benchmark configurations |

## Requirements

- Python 3.9+ with `gymnasium`, `numpy`, `clingo` (5.7.1)
- A pre-trained Q-learning policy is provided in `weights.pkl`

## Usage

```bash
python3 Main.py --method=0 --rounds=100 --size=15 --horizon=3 \
                --ctd=0 --epsilon=0.05 --delta=0.05
```

Key arguments:

- `--method` — `0` PPF, `1` PPF\* (with policy caching), `2` UPF baseline ([Adam and Eiter, 2025]), `3` plain RL
- `--horizon` — planning horizon `h` (use `h ≥ 5` when CTD is enabled)
- `--size` — grid side length (e.g. 15, 50, 100)
- `--ctd` — `0` enforce only MTN1, `1` enforce MTN1+MTN2+CTD
- `--epsilon`, `--delta` — error tolerance and confidence parameter
- `--rounds`, `--seed` — number of rounds and master seed for instance generation

The full set of benchmark configurations used for the paper is listed in `configs.txt`.

## Additional Information

For full ASP encodings, extended experimental results, and further details, see [Appendix.pdf](Appendix.pdf).
