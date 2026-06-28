
DESIGNATION_LADDERS: list[list[str]] = [
    ["Trainee Software Engineer", "Software Engineer", "Senior Software Engineer"],
    ["Associate Consultant", "Senior Associate Consultant", "Consultant", "Senior Consultant"],
    ["Solutions Enabler", "Solutions Consultant", "Senior Solutions Consultant"],
    ["Technology Solutions Architect", "Principal Technology Architect"],
    ["Manager", "Principal", "Associate Partner", "Partner"],
]

def adjacent_designations(designation: str, max_levels: int = 1) -> list[tuple[str, int]]:
    for ladder in DESIGNATION_LADDERS:
        if designation not in ladder:
            continue
        idx = ladder.index(designation)
        out = []
        for offset in range(-max_levels, max_levels + 1):
            if offset == 0:
                continue
            j = idx + offset
            if 0 <= j < len(ladder):
                out.append((ladder[j], offset))
        return out
    return []
