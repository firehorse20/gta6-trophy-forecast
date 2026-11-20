"""GTA VI trophy rarity forecast: per-trophy logit trends with an earners/players
decomposition and a calibrated spread.

Method (standard library only, deterministic):
  For every trophy, the case history gives a time series of Exophase reads:
  players tracked N(t) and earn rate r(t) in percent.  Work on the logit
  scale against x = ln(1 + hours since the 2026-11-19T00:00:00Z release).
  - Trend: weighted least squares of logit(r) on x, recency-weighted
    (12-hour half-life), slope clipped, so the forecast follows how each
    trophy's rate has actually been moving.  The fit uses only the most
    recent 24 hours of reads: the regime that extrapolates is the recent
    one, and a line fitted over the whole launch bends the wrong way on a
    saturating curve.
  - Decomposition: earners e(t) = N(t) * r(t) / 100 only ever grow, and the
    player base N(t) grows too; fit log e and log N on x the same way and
    forecast r = 100 * e / N.  This separates genuine progress (players
    earning the trophy) from dilution (new players who have not earned it
    yet), which is the main force moving rates down after a refresh.
  - Median: the mean of the two forecasts in logit space, shrunk toward the
    last read (trends bend as trophies saturate, so a straight extrapolation
    over 4.5 days overshoots), and clipped so no trophy's median moves more
    than a bounded amount from its last rate.
  - Spread: the trend fit's prediction error at the target, plus an
    allowance for one more Exophase refresh between the last read and the
    target (measured from the refresh jumps in the history, floored).
  Quantiles are the normal quantiles of that logit distribution, mapped
  back to percent and rounded to 4 decimals.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
It reads every <id>.in in the input directory and writes <id>.out per case:
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}.
Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release instant: the origin of the log-time axis (hours since release).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Trend fits use only the most recent window of reads: the regime that
# extrapolates is the recent one, not launch night.
WINDOW_H = 24.0
# Recency half-life for the trend fits, in hours.
HALF_LIFE_H = 12.0
# Clip on a fitted log-time slope, in logit units per unit of ln(1 + hours).
SLOPE_CLIP = 1.5
# Shrink the fitted move toward the last read: trends bend as trophies
# saturate, so a straight extrapolation over 4.5 days overshoots.  Tuned
# on synthetic final-like cases: 0.3 beats the house baseline on average
# across refresh cadences and player-growth rates.
SHRINK = 0.3
# Clip on how far a median may move from the trophy's last rate, in logit.
MOVE_CLIP = 2.0
# Refresh allowance when the window shows no refresh jump at all.
REFRESH_FLOOR = 0.75
# Floor / cap on the total logit spread of a trophy.
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.0


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (whole numbers come back as ints)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t: datetime) -> float:
    return (t - RELEASE).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, s, xbar, sxx, n_eff)
    with s the weighted residual sd (0 when the fit is exact or degenerate).
    Weights are normalized to mean 1 so s is on the right scale."""
    n = len(xs)
    if n == 0:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    ws = [w * n / sw for w in ws]
    xbar = sum(w * x for w, x in zip(ws, xs)) / n
    ybar = sum(w * y for w, y in zip(ws, ys)) / n
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    n_eff = n * n / sum(w * w for w in ws)
    if sxx <= 1e-12:
        return (ybar, 0.0, 0.0, xbar, sxx, n_eff)
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    if n <= 2:
        s = 0.0
    else:
        resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        s = math.sqrt(max(resid, 0.0) / (n - 2))
    return (a, b, s, xbar, sxx, n_eff)


def forecast_trophy(reads, horizon_days, target_x):
    """Forecast one trophy's five quantiles.

    reads: list of (t_hours, players, rate) for this trophy, oldest first,
    with rate None where the read lacked it. Returns [q05..q95] in percent.
    """
    # Last non-null rate, for fallbacks.
    last_rate = None
    for _, _, r in reversed(reads):
        if r is not None:
            last_rate = r
            break
    if last_rate is None:
        last_rate = 0.0
    last_lg = logit(last_rate)

    # A trophy everyone already has stays (nearly) there: Exophase kept it
    # at 100% through a sixfold player growth on launch night.
    if last_rate >= 100.0:
        return [99.0, 99.9, 100, 100, 100]

    # The recent window is the regime that extrapolates.  Fall back to the
    # whole history when it holds too few reads (the practice cases).
    t_last = reads[-1][0]
    window = [r for r in reads if r[0] >= t_last - WINDOW_H]
    if len(window) < 3:
        window = reads

    # Trend series on the window.
    xs, ys, ws = [], [], []
    for t, _n, r in window:
        if r is None:
            continue
        xs.append(math.log1p(max(t, 0.0)))
        ys.append(logit(r))
        ws.append(0.5 ** ((t_last - t) / HALF_LIFE_H))
    if len(xs) < 3:
        # Too little to fit: the house baseline (last rate, wide spread).
        spread = REFRESH_FLOOR
        return [round4(inv_logit(last_lg + z * spread)) for z in Z]

    trend = wls(xs, ys, ws)
    a, b, s_fit, xbar, sxx, n_eff = trend
    b = max(-SLOPE_CLIP, min(SLOPE_CLIP, b))
    trend_lg = a + b * target_x

    # Earners/players decomposition on the same window: earners only grow,
    # players grow too; forecast r = 100 * e / N.
    exs, eys, ews = [], [], []
    nxs, nys, nws = [], [], []
    cum_e = 0.0
    for t, n, r in window:
        if r is None or n is None or n <= 0:
            continue
        e = n * r / 100.0
        cum_e = max(cum_e, e)
        x = math.log1p(max(t, 0.0))
        w = 0.5 ** ((t_last - t) / HALF_LIFE_H)
        exs.append(x)
        eys.append(math.log(max(cum_e, 1e-9)))
        ews.append(w)
        nxs.append(x)
        nys.append(math.log(n))
        nws.append(w)
    decomp_lg = None
    if len(exs) >= 3 and len(nxs) >= 3:
        ef = wls(exs, eys, ews)
        nf = wls(nxs, nys, nws)
        if ef is not None and nf is not None:
            log_e_t = ef[0] + max(-SLOPE_CLIP, min(SLOPE_CLIP, ef[1])) * target_x
            log_n_t = nf[0] + max(-SLOPE_CLIP, min(SLOPE_CLIP, nf[1])) * target_x
            rate_d = 100.0 * math.exp(log_e_t - log_n_t)
            if 0.0 < rate_d < 100.0:
                decomp_lg = logit(rate_d)

    if decomp_lg is None:
        median_lg = trend_lg
    else:
        median_lg = 0.5 * trend_lg + 0.5 * decomp_lg
    # Trends bend as trophies saturate: shrink the fitted move toward the
    # last read, and never let the median run far from what was last seen.
    median_lg = last_lg + SHRINK * (median_lg - last_lg)
    median_lg = max(last_lg - MOVE_CLIP, min(last_lg + MOVE_CLIP, median_lg))

    # Spread: the trend fit's prediction error at the target, plus one more
    # refresh jump (measured from the jumps in the window, floored).
    if sxx > 1e-12 and len(xs) > 2:
        se_pred = s_fit * math.sqrt(1.0 / n_eff + (target_x - xbar) ** 2 / sxx)
    else:
        se_pred = max(s_fit, 0.15)
    se_pred = max(se_pred, 0.15)
    jumps = []
    prev = None
    for y in ys:
        if prev is not None and abs(y - prev) > 1e-9:
            jumps.append(abs(y - prev))
        prev = y
    if jumps:
        rms = math.sqrt(sum(j * j for j in jumps) / len(jumps))
        refresh = max(0.5 * rms, 0.25)
    else:
        refresh = REFRESH_FLOOR
    spread = math.sqrt(se_pred * se_pred + refresh * refresh)
    spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
    return [round4(inv_logit(median_lg + z * spread)) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    target_x = math.log1p(max(hours_since_release(target), 0.0))
    horizon_days = case["horizon_minutes"] / 1440.0
    trophies = case["trophies"]
    # Per-trophy read series, in the order of `trophies`.
    series = [[] for _ in trophies]
    for read in case["history"]:
        t = hours_since_release(parse_time(read["at"]))
        n = read.get("players_tracked")
        rates = read["rates"]
        for i in range(len(trophies)):
            r = rates[i] if i < len(rates) else None
            series[i].append((t, n, r))
    rows = []
    for i, trophy in enumerate(trophies):
        rows.append({"id": trophy["id"], "q": forecast_trophy(series[i], horizon_days, target_x)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
