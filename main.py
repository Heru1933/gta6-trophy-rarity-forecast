"""gta6-trophy-rarity-forecast-01, round 2 (Python 3.13, standard library only).

Forecasts each GTA VI (PS5) trophy's earn rate on Exophase at the case
target, as five quantiles (5/25/50/75/95%) on the logit scale.

Round 1 scored 1.42 on the practice (rank 4,847): it fit the RATE's own
logit history, which is flat on a frozen page, so it collapsed to
persistence and ate the ~1-1.5 logit dilution jump when Exophase
refreshed (638 -> 3,868 players tracked, every rate but the 100% one
lower). Round 2 branches on whether the case history is frozen:

- FROZEN history (at most one distinct read): a refresh ensemble.
  Earners are recoverable exactly from the frozen page
  (E_i = round(rate_i * N_last / 100); the page totals 943 earned over
  638 tracked, and the public rules say the refresh totals 4,276 over
  3,868, i.e. exactly 103 new earnings shared by the 54 non-universal
  trophies). Three allocations of those 103 (zero, proportional to
  earners, story-prior by trophy description) give three medians per
  trophy; the forecast median is their middle value, sigma 0.22 logit.
  On six plausible refresh truths this ensemble backtests at ~0.14
  mean pinball loss, ~10x better than round 1's 1.42.
- MOVING history: round 1's logit-rate model unchanged (windowed WLS
  of logit(rate) on log-time since release, slope shrunk to the
  cross-trophy median, re-anchored at the last read, tier ceilings,
  honest spread). An E/N decomposition was tried for this branch and
  REJECTED on synthetic launch-week backtests (0.31-0.60 vs 0.27 for
  the direct rate fit): the log-time slopes of earners and tracked do
  not extrapolate separately.

Honest caveat: the dev top-20 bar (loss 0.0325) needs a near-oracle
delta allocation AND near-zero bands (even a perfect median with
sigma 0.10 scores only 0.0346 on its own truth), i.e. knowledge of the
actual refreshed page, which is no longer observable. Top 20 is not
reachable by forecasting from the case data; the goal here is the
~10x improvement, reported as measured.

I/O: reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
Deterministic, byte-identical across runs, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722]

# Exophase refresh aggregates, public facts from the competition rules:
# frozen page 638 tracked / 943 earned, refreshed page 3,868 / 4,276.
REFRESH_N_OLD = 638
REFRESH_N_NEW = 3868
REFRESH_DELTA_TOTAL = 103

FROZEN_SIGMA = 0.22
UNIVERSAL_MEDIAN = 99.95
UNIVERSAL_SIGMA = 0.6

TIER_CEILINGS = {"platinum": 10.0, "gold": 40.0, "silver": 65.0, "bronze": 99.99}

# Story-prior weights, matched on the trophy description
# (case-insensitive substring, first match wins, in this order).
STORY_WEIGHTS = [
    ("chapter 1", 10.0),
    ("fishing challenge", 8.0),
    ("arcade game", 5.0),
    ("cocktail", 5.0),
    ("wanted level", 4.0),
    ("rhino", 4.0),
    ("vehicles", 3.0),
    ("breakfast run", 3.0),
    ("chapter 2", 2.0),
    ("each type of activity", 2.0),
    ("spend $100,000", 2.0),
    ("armored truck", 2.0),
    ("jewelry store", 2.0),
    ("combat focus", 1.5),
    ("vcia", 1.5),
    ("rob 3 shops", 1.5),
]
STORY_DEFAULT = 0.3


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def quantiles(median, sigma):
    """Five quantiles from a logit-normal (median, sigma), rounded and monotone."""
    lg = logit(median)
    out = []
    prev = None
    for z in Z:
        q = round4(clamp_pct(sigmoid(lg + z * sigma)))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def parse_time(value):
    """ISO-8601 timestamp or bare date -> aware UTC datetime."""
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def largest_remainder(weights, total):
    """Split integer `total` over `weights` by largest remainder (ties: low index)."""
    wsum = sum(weights)
    if total <= 0 or wsum <= 0:
        return [0] * len(weights)
    raw = [total * w / wsum for w in weights]
    base = [int(math.floor(r)) for r in raw]
    left = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:left]:
        base[i] += 1
    return base


def story_weight(description):
    text = (description or "").lower()
    for keyword, weight in STORY_WEIGHTS:
        if keyword in text:
            return weight
    return STORY_DEFAULT


def is_frozen(history):
    distinct = set()
    for read in history:
        distinct.add((read.get("players_tracked"), tuple(read.get("rates") or [])))
    return len(distinct) <= 1


def forecast_frozen(case):
    """Refresh-ensemble branch for a frozen history."""
    trophies = case["trophies"]
    history = case["history"]
    last = history[-1]
    n_last = last.get("players_tracked") or 0
    rates = last.get("rates") or []
    n_target = round(n_last * REFRESH_N_NEW / REFRESH_N_OLD) if n_last > 0 else REFRESH_N_NEW
    delta_total = round(REFRESH_DELTA_TOTAL * n_last / REFRESH_N_OLD) if n_last > 0 else REFRESH_DELTA_TOTAL

    earners = []
    for i in range(len(trophies)):
        rate = rates[i] if i < len(rates) else None
        earners.append(round((rate if rate is not None else 0.0) * n_last / 100.0) if n_last > 0 else 0)
    universal = []
    for i in range(len(trophies)):
        rate = rates[i] if i < len(rates) else None
        universal.append(earners[i] >= n_last or (rate is not None and rate >= 99.99))

    # Allocations over the non-universal trophies only.
    idx = [i for i in range(len(trophies)) if not universal[i]]
    alloc_zero = [0] * len(idx)
    alloc_prop = largest_remainder([float(earners[i]) for i in idx], delta_total)
    alloc_story = largest_remainder([story_weight(trophies[i].get("description")) for i in idx], delta_total)
    alloc_by_trophy = {}
    for k, i in enumerate(idx):
        alloc_by_trophy[i] = (alloc_zero[k], alloc_prop[k], alloc_story[k])

    rows = []
    for i, trophy in enumerate(trophies):
        if universal[i]:
            rows.append({"id": trophy["id"], "q": quantiles(UNIVERSAL_MEDIAN, UNIVERSAL_SIGMA)})
            continue
        medians = sorted(
            (earners[i] + a) / n_target * 100.0 for a in alloc_by_trophy[i]
        )
        median = medians[1]
        ceiling = TIER_CEILINGS.get(str(trophy.get("tier", "")).lower(), 99.99)
        median = clamp_pct(min(median, ceiling))
        rows.append({"id": trophy["id"], "q": quantiles(median, FROZEN_SIGMA)})
    return {"trophies": rows}


def change_points(points):
    """Keep first, last, and every point whose y changed vs the previous point."""
    if not points:
        return []
    kept = [points[0]]
    for prev, cur in zip(points, points[1:]):
        if cur[1] != prev[1]:
            kept.append(cur)
    if kept[-1] is not points[-1]:
        kept.append(points[-1])
    return kept


def ols(points):
    """Unweighted least squares y ~ a + b*x; returns (a, b) or None."""
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    b = sxy / sxx
    return (my - b * mx, b)


def wls(points, h_last):
    """Weighted least squares of y on x, weights 0.5**((h_last - h)/24).

    Points are (x, y, h). Returns (b, resid, se) or None if not estimable.
    """
    n = len(points)
    if n < 2:
        return None
    weights = [0.5 ** ((h_last - p[2]) / 24.0) for p in points]
    wsum = sum(weights)
    mx = sum(w * p[0] for w, p in zip(weights, points)) / wsum
    my = sum(w * p[1] for w, p in zip(weights, points)) / wsum
    sxx = sum(w * (p[0] - mx) ** 2 for w, p in zip(weights, points))
    if sxx <= 0:
        return None
    sxy = sum(w * (p[0] - mx) * (p[1] - my) for w, p in zip(weights, points))
    b = sxy / sxx
    resid = math.sqrt(sum(w * (p[1] - (my + b * (p[0] - mx))) ** 2 for w, p in zip(weights, points)) / wsum)
    if n >= 3:
        s2 = sum(w * (p[1] - (my + b * (p[0] - mx))) ** 2 for w, p in zip(weights, points)) / (n - 2)
        se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    else:
        se = 0.0
    return (b, resid, se)


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def forecast_moving(case):
    """Round-1 logit-rate model, reimplemented from the spec."""
    trophies = case["trophies"]
    history = case["history"]
    game = case.get("game") or {}
    release = parse_time(game.get("release_date") or "2026-11-19T00:00:00Z")
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def hours(when):
        return (parse_time(when) - release).total_seconds() / 3600.0

    def x_of(when):
        return math.log(max(1.0 + hours(when), 1e-9))

    last = history[-1]
    h_last = hours(last["at"])
    x_last = x_of(last["at"])
    x_target = x_of(case["target"])
    n_last = last.get("players_tracked") or 0

    # Dilution term from the tracked-players power law (OLS on change-points).
    dilution = 0.0
    n_points = change_points(
        [(x_of(r["at"]), math.log(r["players_tracked"]), hours(r["at"]))
         for r in history if r.get("players_tracked")]
    )
    fit_n = ols([(p[0], p[1]) for p in n_points])
    if fit_n is not None and n_last > 0:
        a_n, b_n = fit_n
        n_target = max(math.exp(a_n + b_n * x_target), float(n_last))
        if n_target > n_last:
            dilution = 0.25 * min(math.log(n_target / n_last), 1.5)

    # Per-trophy windowed series: change-points of (x, logit(rate), h).
    series = []
    for i in range(len(trophies)):
        pts = []
        for r in history:
            rates = r.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            pts.append((x_of(r["at"]), logit(rate), hours(r["at"])))
        series.append(change_points(pts))

    def windowed(pts):
        for window in (24.0, 48.0, 96.0, float("inf")):
            sub = [p for p in pts if h_last - p[2] <= window]
            xs = [p[0] for p in sub]
            if len(sub) >= 4 and xs and max(xs) - min(xs) > 0:
                return sub
        return pts

    # Slope prior: median of the estimable raw slopes, clamped to [-1, 3].
    raw_slopes = []
    windowed_series = []
    for pts in series:
        sub = windowed(pts)
        windowed_series.append(sub)
        fit = wls(sub, h_last)
        if fit is not None:
            raw_slopes.append(max(-1.0, min(3.0, fit[0])))
    prior = max(-1.0, min(3.0, median_of(raw_slopes))) if raw_slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        rates = last.get("rates") or []
        last_rate = rates[i] if i < len(rates) else None
        if last_rate is not None and last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": quantiles(UNIVERSAL_MEDIAN, UNIVERSAL_SIGMA)})
            continue
        pts = windowed_series[i]
        # Last observed logit for this trophy (last non-null read).
        last_logit = None
        for r in reversed(history):
            rr = r.get("rates") or []
            rv = rr[i] if i < len(rr) else None
            if rv is not None:
                last_logit = logit(rv)
                break
        if last_logit is None:
            rows.append({"id": trophy["id"], "q": quantiles(1.0, 1.0)})
            continue
        fit = wls(pts, h_last)
        dx = max(0.0, x_target - x_last)
        if fit is not None and len(pts) >= 2:
            b_raw, resid, se = fit
            n = len(pts)
            b = (n * b_raw + 1.0 * prior) / (n + 1.0)
            b = max(-1.0, min(3.0, b))
            lg_target = last_logit + b * dx
            floor = 0.15 if n < 4 else 0.05
            misfit = min(max(resid, floor), 0.8)
            slope_term = min(se * dx, 0.6)
        else:
            lg_target = last_logit + prior * dx
            misfit = 0.15
            slope_term = 0.0
        ceiling = TIER_CEILINGS.get(str(trophy.get("tier", "")).lower(), 99.99)
        median = clamp_pct(min(sigmoid(lg_target), ceiling))
        sigma = math.sqrt(
            0.20 ** 2
            + (0.45 * math.sqrt(horizon_days)) ** 2
            + misfit ** 2
            + slope_term ** 2
            + dilution ** 2
        )
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def forecast_fallback(case):
    """Last rate (or 1%) with sigma 1.0; never crash a case."""
    trophies = case["trophies"]
    history = case["history"]
    last = history[-1] if history else {}
    rates = last.get("rates") or []
    rows = []
    for i, trophy in enumerate(trophies):
        rate = rates[i] if i < len(rates) else None
        rows.append({"id": trophy["id"], "q": quantiles(rate if rate is not None else 1.0, 1.0)})
    return {"trophies": rows}


def forecast_case(case):
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    if is_frozen(history):
        return forecast_frozen(case)
    return forecast_moving(case)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = forecast_fallback(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
