# Canonical, location-agnostic seniority ladder. UK/USA and India use two different
# naming families for the SAME real org levels (confirmed by RM ground truth,
# e.g. "Associate Consultant" (UK/USA) = "Software Engineer" (India)) -- location
# does NOT reliably predict which family a given employee's title belongs to (both
# families show up at every real office), so titles are leveled directly, not
# filtered by employee location. India's ladder is one rung shorter at the top --
# there is no India equivalent of Associate Partner.
LEVELS: list[dict] = [
    {"level": 1, "titles": ["Intern", "Trainee Software Engineer"]},
    {"level": 2, "titles": ["Associate Consultant", "Software Engineer"]},
    {"level": 3, "titles": ["Senior Associate Consultant", "Senior Software Engineer"]},
    {"level": 4, "titles": ["Consultant", "Solutions Enabler"]},
    {"level": 5, "titles": ["Senior Consultant", "Solutions Consultant"]},
    # "Technical Solutions Architect" / "Technology Solutions Architect" are treated as
    # the same level -- near-duplicate spellings in the source data, most likely
    # inconsistent data entry rather than two distinct real grades.
    {"level": 6, "titles": ["Manager", "Senior Solutions Consultant", "Technical Solutions Architect", "Technology Solutions Architect"]},
    {"level": 7, "titles": ["Principal", "Principal Architect", "Principal Technology Architect"]},
    {"level": 8, "titles": ["Associate Partner", "Leadership"]},
    {"level": 9, "titles": ["Partner"]},
]

TITLE_TO_LEVEL: dict[str, int] = {title: entry["level"] for entry in LEVELS for title in entry["titles"]}
TITLES_AT_LEVEL: dict[int, list[str]] = {entry["level"]: entry["titles"] for entry in LEVELS}

# Manager/Principal/Associate Partner/Partner/Leadership are oversight and
# client-relationship roles, not hands-on technical ICs -- gating their availability
# on the same technical skill requirements (SQL, Python, etc.) as a Senior Software
# Engineer produces false "need to hire a Partner" signals for every project that
# asks for any technical skill. Scoped to this consulting/oversight function, not a
# level cutoff -- Architect-family titles stay skill-gated like any technical IC
# regardless of seniority, since architects remain hands-on technical roles.
LEADERSHIP_DESIGNATIONS: frozenset[str] = frozenset(
    {"Manager", "Principal", "Associate Partner", "Partner", "Leadership"}
)


def level_of(designation: str) -> int | None:
    return TITLE_TO_LEVEL.get(designation)


def same_level_peers(designation: str) -> list[str]:
    """Other real titles at the exact same level (the other naming family's
    equivalent) -- a genuine equivalent, not a downgrade/upgrade fallback."""
    level = TITLE_TO_LEVEL.get(designation)
    if level is None:
        return []
    return [t for t in TITLES_AT_LEVEL[level] if t != designation]


def adjacent_designations(designation: str, max_levels: int = 1) -> list[tuple[str, int]]:
    level = TITLE_TO_LEVEL.get(designation)
    if level is None:
        return []
    out: list[tuple[str, int]] = []
    for offset in range(-max_levels, max_levels + 1):
        if offset == 0:
            continue
        for title in TITLES_AT_LEVEL.get(level + offset, []):
            out.append((title, offset))
    return out
