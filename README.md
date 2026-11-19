# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Heru1933/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Heru1933/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at the case target, as five quantiles (5/25/50/75/95%) on the logit scale.

The model fits, per trophy, a weighted line of logit(rate) against log-time since release over the most recent window with enough distinct reads, shrinks the slope toward the cross-trophy median, re-anchors at the last observed page, and caps the median at a tier-plausible ceiling. The spread widens honestly with the horizon (refresh noise + random walk + fit misfit + slope uncertainty + tracked-player dilution), because a week-one forecast from launch-week data deserves a wide band. See the docstring at the top of `main.py` for the full reasoning, including the approaches that were tried and rejected on synthetic backtests.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` case in the input directory and writes one `<id>.out` per case. Deterministic: two runs produce byte-identical output, in well under a second per case.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
