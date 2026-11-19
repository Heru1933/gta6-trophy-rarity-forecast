"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Approach
--------
An earn rate is a ratio, rate = earners / players_tracked, and the two
sides move for different reasons: tracked players flood in at launch
(launch night: 638 tracked, then 3,868 within hours, which *lowered*
every rate on the page even though more people had earned each trophy),
while earners grow as players actually play. So the rate's own history
is the object to model, on the logit scale against log-time:

  logit(rate_i) = a_i + b_i * ln(1 + h),   h = hours since release.

That is a logistic curve in log-time: rates rise steeply early, then
bend toward saturation, which is how trophy earn rates actually behave
over a launch week. The fit is weighted least squares on the case
history compressed to its change points (Exophase only refreshes its
counts occasionally, so most reads repeat the same page and would
otherwise overweight one stale value), with a 24-hour half-life so the
recent trend -- the one that survives into the target week -- dominates.
Crucially, the fit uses only the most recent window that still holds
enough distinct reads (48 hours, else 96, else the whole history): a
trophy whose rate has flattened as it saturates must show a flat
recent slope, not the steep slope of its early rise -- fitting the
whole history over-extrapolates fast trophies far past saturation
(checked on synthetic launch-week data: whole-history fitting put
saturated bronze trophies near 99% when the truth was ~70%).
Each trophy's slope is shrunk toward the cross-trophy median slope (empirical-Bayes style): a trophy
with only one or two distinct reads borrows strength from the rest, and
a trophy whose history is perfectly flat falls back to persistence --
with no movement observed, no movement is assumed. The intercept is
re-anchored at the last observed read, so the forecast starts from the
latest level. The median is capped at a tier-plausible ceiling
(platinum 10%, gold 40%, silver 65%, bronze ~100%): a week after
launch almost nobody has finished a game this size, so an extrapolated
platinum rate of 30% would be a fitting artefact, not a forecast. The
100% trophy (everyone earns it on starting) is handled separately with
a tight band near 100%.

Why not fit earners and tracked separately? I tried it: a power law
for earners extrapolates the steep early rise of an S-curve far past
saturation (on synthetic launch-week data it put fast bronze trophies
at 99.9% when the truth was 40-70%). The logit-rate fit is the
conservative one, and the tracked-players series is still used -- for
the dilution term of the spread, below.

Spread (logit scale, honest about a 4.5-day horizon)
----------------------------------------------------
sigma^2 = refresh^2 + walk^2 + misfit^2 + slope-uncertainty^2 + dilution^2
  refresh  0.20  -- one Exophase refresh can move a rate a lot (launch
                  night's first refresh moved the typical rate by ~1.5
                  logit units); this term does not shrink with horizon.
  walk     0.45 * sqrt(days) -- random-walk growth of uncertainty.
  misfit   residual std of the trophy's own logit fit.
  slope    standard error of the fitted slope times the log-time distance
           extrapolated beyond the last read.
  dilution 0.25 * ln(N_target / N_last), capped -- if the tracked base
           is still growing fast, rates can be diluted down again.
           N_target comes from a power-law fit to players_tracked.

Quantiles are the median shifted by z_tau * sigma on the logit scale,
so the 5%-95% band for the final spans roughly a factor of
e^(2*1.645*sigma) in odds -- wide, because a week-one forecast from
launch-week data deserves to be wide. No randomness anywhere: two runs
are byte-identical.

Python 3.13, standard library only, no network. Reads every *.in file
in $MAKEFUN_INPUT_DIR (else /input) and writes one <id>.out per case
to $MAKEFUN_OUTPUT_DIR (else /output).
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
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

RELEASE_FALLBACK = "2026-11-19T00:00:00Z"

# Median-rate ceilings by tier (percent). Soft caps on the median only;
# the upper quantiles may still exceed them.
TIER_CEILING = {"platinum": 10.0, "gold": 40.0, "silver": 65.0,
                "bronze": 99.99}

SIGMA_REFRESH = 0.20
SIGMA_WALK_PER_SQRT_DAY = 0.45
SIGMA_DILUTION = 0.25
DEFAULT_MISFIT = 0.15
SHRINKAGE_POINTS = 1.0  # pseudo-count pulling a slope to the prior
HALF_LIFE_HOURS = 24.0  # recency weight half-life for the logit fit
FIT_WINDOWS_HOURS = (24.0, 48.0, 96.0, float("inf"))  # shortest window
# with enough distinct reads wins; a saturated trophy's flat recent
# slope must not be overwritten by its steep early rise.
MIN_WINDOW_POINTS = 4


def parse_time(text):
    """ISO 8601 timestamp -> aware datetime (UTC). A bare date or a
    timestamp with no offset is read as UTC."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clamp_pct(x):
    return min(99.99, max(0.01, float(x)))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def wols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid, se).

    resid is the (unweighted) residual std, 0 with <3 points; se the
    standard error of b, 0 when it cannot be estimated.
    """
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n >= 3:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        resid = math.sqrt(rss / (n - 2))
        se = resid / math.sqrt(sxx)
    else:
        resid = 0.0
        se = 0.0
    return a, b, resid, se


def ols(xs, ys):
    """Unweighted least squares; returns (a, b, resid, se, sxx)."""
    a, b, resid, se = wols(xs, ys, [1.0] * len(xs))
    mx = sum(xs) / len(xs)
    return a, b, resid, se, sum((x - mx) ** 2 for x in xs)


def median_of(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def change_points(xs, ys, hs=None):
    """Keep the first point, the last point, and every point where y
    changed from the previously kept value. Flat stretches of identical
    reads collapse to one point, so a stale page cannot dominate a fit.
    Returns (x, y, h) triples, h defaulting to 0 when not given."""
    if hs is None:
        hs = [0.0] * len(xs)
    pts = []
    for x, y, h in zip(xs, ys, hs):
        if not pts or y != pts[-1][1]:
            pts.append((x, y, h))
    if pts and pts[-1] != (xs[-1], ys[-1], hs[-1]):
        pts.append((xs[-1], ys[-1], hs[-1]))
    return pts


def windowed(pts, h_last):
    """Shortest recent window holding enough distinct reads.

    pts are (x, logit, hours) triples, oldest first. Returns the
    trailing slice in the first window (48h, 96h, all) with at least
    MIN_WINDOW_POINTS points and a nonzero x-span, else all points.
    """
    for window in FIT_WINDOWS_HOURS:
        cut = [p for p in pts if h_last - p[2] <= window]
        if len(cut) >= MIN_WINDOW_POINTS and cut[-1][0] > cut[0][0]:
            return cut
    return pts


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    release_raw = (case.get("game") or {}).get("release_date") or RELEASE_FALLBACK
    release = parse_time(release_raw)
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def hours_of(when):
        return (parse_time(when) - release).total_seconds() / 3600.0

    def x_of(when):
        return math.log1p(max(0.0, hours_of(when)))

    x_target = math.log1p(max(
        0.0, (target - release).total_seconds() / 3600.0))
    h_last = hours_of(history[-1]["at"])
    x_last = x_of(history[-1]["at"])

    # --- players tracked: one power law, for the dilution term --------
    n_xs, n_ys = [], []
    for read in history:
        n = read.get("players_tracked")
        if n and n > 0:
            n_xs.append(x_of(read["at"]))
            n_ys.append(math.log(n))
    n_pts = change_points(n_xs, n_ys)
    n_last = history[-1].get("players_tracked") or 0
    if len(n_pts) >= 2:
        n_a, n_b, _, _, _ = ols([p[0] for p in n_pts], [p[1] for p in n_pts])
        n_target = math.exp(n_a + n_b * x_target)
    else:
        n_target = float(n_last) if n_last else 1.0
    n_target = max(n_target, float(n_last) if n_last else 1.0)
    dilution = SIGMA_DILUTION * min(
        math.log(n_target / (n_last or n_target)), 1.5) \
        if n_target > (n_last or 0) else 0.0

    # --- per-trophy logit-rate series ----------------------------------
    series = []
    for i in range(len(trophies)):
        xs, ys, hs = [], [], []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            xs.append(x_of(read["at"]))
            ys.append(logit(rate))
            hs.append(hours_of(read["at"]))
        pts = change_points(xs, ys, hs) if xs else []
        series.append({"pts": pts,
                       "last_rate": history[-1]["rates"][i]})

    # Cross-trophy slope prior: median of the estimable slopes.
    raw_slopes = []
    for s in series:
        pts = windowed(s["pts"], h_last)
        if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
            ws = [0.5 ** ((h_last - p[2]) / HALF_LIFE_HOURS) for p in pts]
            _, b, _, _ = wols([p[0] for p in pts], [p[1] for p in pts], ws)
            raw_slopes.append(b)
    slope_prior = median_of(raw_slopes) if raw_slopes else 0.0
    slope_prior = min(max(slope_prior, -1.0), 3.0)

    rows = []
    for i, trophy in enumerate(trophies):
        s = series[i]
        pts = s["pts"]
        last_rate = s["last_rate"]
        if last_rate is None:
            last_rate = 0.0
        tier = trophy.get("tier", "bronze")
        ceiling = TIER_CEILING.get(tier, 99.99)

        if last_rate >= 99.99:
            # Everybody already has it. It should stay near-universal,
            # but the tracked base can pick up players who have not
            # started yet, so the band reaches down to ~99.8% rather
            # than insisting on 99.99%.
            median, sigma = 99.95, 0.6
        else:
            pts = windowed(s["pts"], h_last)
            if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ws = [0.5 ** ((h_last - p[2]) / HALF_LIFE_HOURS)
                      for p in pts]
                _, b_raw, resid, se = wols(xs, ys, ws)
                n_pts = len(pts)
                # Shrink the slope toward the cross-trophy prior; the
                # fewer distinct reads, the more shrinkage.
                b = (n_pts * b_raw + SHRINKAGE_POINTS * slope_prior) / (
                    n_pts + SHRINKAGE_POINTS)
                b = min(max(b, -1.0), 3.0)
                # Re-anchor at the last observed level, so the forecast
                # starts from the latest page, not the fitted line.
                lg_target = ys[-1] + b * (x_target - xs[-1])
                # A two-point fit has zero residuals by construction;
                # do not let that read as certainty.
                floor = 0.15 if n_pts < 4 else 0.05
                misfit = min(max(resid, floor), 0.8)
                slope_term = min(se * max(0.0, x_target - xs[-1]), 0.6)
            else:
                # No movement observed: persistence, grown at the prior
                # slope from the last observed rate.
                lg_target = logit(last_rate) + slope_prior * max(
                    0.0, x_target - x_last)
                misfit = DEFAULT_MISFIT
                slope_term = 0.0
            median = sigmoid(lg_target)
            median = min(median, ceiling)
            median = clamp_pct(median)
            sigma = math.sqrt(
                SIGMA_REFRESH ** 2
                + (SIGMA_WALK_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                + misfit ** 2
                + slope_term ** 2
                + dilution ** 2
            )
        lg = logit(median)
        qs = []
        for z in Z:
            q = round4(sigmoid(lg + z * sigma))
            qs.append(min(99.99, max(0.01, q)))
        # Enforce non-decreasing after rounding/clamping.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Last-resort forecast: last rate (or 1%) with a wide band. Used
    only if the main forecast raises, so a run never crashes a case."""
    rows = []
    last = case["history"][-1] if case.get("history") else None
    for i, trophy in enumerate(case["trophies"]):
        rate = None
        if last and last.get("rates"):
            rate = last["rates"][i]
        median = clamp_pct(rate if rate is not None else 1.0)
        lg = logit(median)
        qs = [round4(sigmoid(lg + z * 1.0)) for z in Z]
        qs = [min(99.99, max(0.01, q)) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception as exc:  # keep the case valid, never crash
            print(f"{path.name}: forecast failed ({exc}); using fallback",
                  file=sys.stderr)
            forecast = fallback_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
