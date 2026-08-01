"""Project revenue benchmarks -- grounded in real JMAN project economics (RM-
provided ground truth on typical team size, duration, and revenue), not
derived from real allocation-hours data.

An earlier version of this derived revenue from real allocation data (hours
x rate-card rate x a per-CoE "bill multiplier"), which looked plausible but
was ~10x too high: a real `project_code` in the source data pools every
allocation across a project's full multi-month lifecycle (extensions,
replacements, multiple phases), averaging ~8 distinct employees and ~102
days per project -- nothing like the ~4-person, ~5-week unit "a project"
actually means operationally. Stacking a 1.6-3.0x bill multiplier on top of
that already-inflated cost compounded the error further. Replaced with the
real templates below instead of trying to re-segment the messy historical
data into clean phases.
"""

WEEKS_PER_MONTH = 365.25 / 7 / 12  # ~4.33

# A typical DELIVERY project: ~5 weeks, ~$35k revenue, team = 2x engineer
# (Senior Software Engineer or Software Engineer) @ 100% each, 1x Solutions
# Enabler @ 100% (sometimes 50% in practice), 1x Consultant (Consultant or
# Senior Consultant). Org-wide average project duration runs longer (~8
# weeks) since some projects extend past this typical case -- kept here as
# context, not used in the revenue math below.
DELIVERY_TEMPLATE = {
    "duration_weeks": 5,
    "org_avg_duration_weeks": 8,
    "revenue_usd": 35_000,
    "role_mix": {
        "Senior Software Engineer": 2.0,
        "Solutions Enabler": 1.0,
        "Consultant": 1.0,
    },
}

# A typical DESIGN & DISCOVERY (D&D) engagement -- the precursor phase most
# delivery projects go through before a client commits to the full delivery
# project. Clients pay for D&D, but conversion to a delivery project isn't
# guaranteed (no real conversion-rate figure exists, so this module doesn't
# fabricate one -- D&D is surfaced as prerequisite context, not folded into
# the delivery revenue math).
DND_TEMPLATE = {
    "duration_weeks": 3,
    "revenue_usd_low": 10_000,
    "revenue_usd_high": 15_000,
    "revenue_usd": 12_500,  # midpoint, used wherever a single figure is needed
    "role_mix": {
        "Associate Consultant": 1.0,
        "Senior Software Engineer": 1.0,
        "Technical Architect": 0.25,
        "Consultant": 0.5,
    },
}


def _fte_months(role_mix: dict[str, float], duration_weeks: float) -> float:
    return sum(role_mix.values()) * (duration_weeks / WEEKS_PER_MONTH)


DELIVERY_TEMPLATE["avg_revenue_per_fte_month"] = round(
    DELIVERY_TEMPLATE["revenue_usd"] / _fte_months(DELIVERY_TEMPLATE["role_mix"], DELIVERY_TEMPLATE["duration_weeks"]), 0
)
DND_TEMPLATE["avg_revenue_per_fte_month"] = round(
    DND_TEMPLATE["revenue_usd"] / _fte_months(DND_TEMPLATE["role_mix"], DND_TEMPLATE["duration_weeks"]), 0
)

# $35k/$10-15k are rates anchored to a 5-week delivery / 3-week D&D engagement,
# not flat per-project constants -- a longer engagement bills more. Expressed
# per-week so any actual requested duration scales proportionally.
DELIVERY_RATE_PER_WEEK_USD = DELIVERY_TEMPLATE["revenue_usd"] / DELIVERY_TEMPLATE["duration_weeks"]
DND_RATE_PER_WEEK_LOW_USD = DND_TEMPLATE["revenue_usd_low"] / DND_TEMPLATE["duration_weeks"]
DND_RATE_PER_WEEK_HIGH_USD = DND_TEMPLATE["revenue_usd_high"] / DND_TEMPLATE["duration_weeks"]

# Real D&D engagements commonly run longer than the 3-week rate anchor above --
# 4-6 weeks is common, 6 being the most typical case per RM ground truth. Used
# as the default duration for the D&D estimate when none is specified, not as
# a change to the underlying $/week rate.
DND_TYPICAL_DURATION_WEEKS = 6


def delivery_revenue_for_duration(duration_weeks: float | None) -> float:
    weeks = duration_weeks if duration_weeks and duration_weeks > 0 else DELIVERY_TEMPLATE["duration_weeks"]
    return round(DELIVERY_RATE_PER_WEEK_USD * weeks, 0)


def dnd_revenue_range_for_duration(duration_weeks: float | None) -> tuple[float, float]:
    weeks = duration_weeks if duration_weeks and duration_weeks > 0 else DND_TYPICAL_DURATION_WEEKS
    return round(DND_RATE_PER_WEEK_LOW_USD * weeks, 0), round(DND_RATE_PER_WEEK_HIGH_USD * weeks, 0)


def _real_project_sample_sizes_by_coe() -> dict[str, int]:
    """How many real historical COMPLETE/ACTIVE projects exist per CoE --
    legitimate context (a trust signal for "have we actually done this kind
    of work before"), kept from real data even though the revenue figure
    itself is now template-based rather than derived from real allocation
    hours."""
    from app.core.adapter import get_adapter
    from app.engines.role_mix_engine import canonical_project_coe

    adapter = get_adapter()
    projects = adapter.get_projects()
    real = projects[
        (projects["date_source"].isin(["given", "derived_allocation"]))
        & (projects["project_status"].isin(["COMPLETE", "ACTIVE"]))
    ].copy()
    real["coe"] = real["tech_coe"].apply(canonical_project_coe)
    counts = real.dropna(subset=["coe"]).groupby("coe")["project_code"].nunique()
    return counts.to_dict()


def _real_project_durations_weeks() -> list[float]:
    """Real per-project duration_weeks (end - start) for every real, dated
    project -- the actual historical distribution this org runs at, used to
    derive Short/Mid/Long buckets from data rather than a guessed split."""
    from app.core.adapter import get_adapter

    adapter = get_adapter()
    projects = adapter.get_projects()
    real = projects[
        (projects["date_source"].isin(["given", "derived_allocation"]))
        & (projects["project_status"].isin(["COMPLETE", "ACTIVE"]))
    ].copy()
    real = real.dropna(subset=["project_start_date", "project_end_date"])
    weeks = (real["project_end_date"] - real["project_start_date"]).dt.days / 7
    return [w for w in weeks.tolist() if w and w > 0]


def compute_duration_buckets() -> dict:
    """Split the real historical project-duration distribution into Short /
    Mid / Long terciles (boundaries and mix % come from the data itself, not
    a fixed cutoff), so the frontend's duration-mix slider can default to
    what this org actually runs and let an RM simulate deviating from it."""
    import statistics

    durations = sorted(_real_project_durations_weeks())
    n = len(durations)
    if n < 3:
        return {"buckets": {}, "total_sample_size": n}

    def _quantile(data: list[float], q: float) -> float:
        return statistics.quantiles(data, n=100)[min(98, max(0, round(q * 100) - 1))] if len(data) > 1 else data[0]

    p33 = _quantile(durations, 1 / 3)
    p67 = _quantile(durations, 2 / 3)

    short = [d for d in durations if d <= p33]
    mid = [d for d in durations if p33 < d <= p67]
    long_ = [d for d in durations if d > p67]

    def _bucket(vals: list[float], lo: float | None, hi: float | None) -> dict:
        return {
            "min_weeks": round(lo, 1) if lo is not None else None,
            "max_weeks": round(hi, 1) if hi is not None else None,
            "avg_weeks": round(statistics.mean(vals), 1) if vals else round((lo or 0), 1),
            "historical_mix_pct": round(100 * len(vals) / n, 1),
            "sample_size": len(vals),
        }

    return {
        "buckets": {
            "short": _bucket(short, min(durations), p33),
            "mid": _bucket(mid, p33, p67),
            "long": _bucket(long_, p67, max(durations)),
        },
        "total_sample_size": n,
    }


def get_revenue_benchmarks_by_coe(duration_weeks: float | None = None) -> dict[str, dict]:
    """canonical CoE -> {avg_revenue_per_project, avg_revenue_per_fte_month,
    sample_size}. This is the inversion table the revenue-target forecast
    reads backwards: target_revenue / avg_revenue_per_project[coe] ~= how
    many delivery projects of that CoE are needed. The revenue figures are
    the same real DELIVERY_TEMPLATE rate for every CoE (team size and
    duration drive revenue more than the specific tech stack, per the real
    template this is grounded in), scaled to `duration_weeks` if given
    (defaults to the template's own 5-week anchor) -- sample_size is the
    real count of historical projects in that CoE."""
    sample_sizes = _real_project_sample_sizes_by_coe()
    if not sample_sizes:
        return {}
    revenue = delivery_revenue_for_duration(duration_weeks)
    weeks_used = duration_weeks if duration_weeks and duration_weeks > 0 else DELIVERY_TEMPLATE["duration_weeks"]
    fte_month_value = round(revenue / _fte_months(DELIVERY_TEMPLATE["role_mix"], weeks_used), 0)
    return {
        coe: {
            "avg_revenue_per_project": revenue,
            "avg_revenue_per_fte_month": fte_month_value,
            "sample_size": count,
        }
        for coe, count in sample_sizes.items()
    }
