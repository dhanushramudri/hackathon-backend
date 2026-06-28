
RESOURCE_CODE_MAP: dict[str, list[str]] = {
    "AC": ["Associate Consultant"],
    "AC (UK)": ["Associate Consultant"],
    "AP": ["Associate Partner"],
    "AP/P": ["Associate Partner", "Partner"],
    "C": ["Consultant"],
    "C/SAC/AC": ["Consultant", "Senior Associate Consultant", "Associate Consultant"],
    "EM": [],
    "Enabler": ["Solutions Enabler"],
    "GTM Architect": [],
    "M": ["Manager"],
    "P": ["Partner"],
    "PA": ["Principal Architect"],
    "SAC": ["Senior Associate Consultant"],
    "SAC - C": ["Senior Associate Consultant", "Consultant"],
    "SAC or AC": ["Senior Associate Consultant", "Associate Consultant"],
    "SAC/AC": ["Senior Associate Consultant", "Associate Consultant"],
    "SC": ["Solutions Consultant"],
    "SC (EM)": ["Solutions Consultant"],
    "SC or C - EM": ["Solutions Consultant", "Consultant"],
    "SE": ["Software Engineer"],
    "SSE": ["Senior Software Engineer"],
    "SSE  or SE": ["Senior Software Engineer", "Software Engineer"],
    "SSE or SE": ["Senior Software Engineer", "Software Engineer"],
    "Snr Sol Con": ["Senior Solutions Consultant"],
    "Sol Con": ["Solutions Consultant"],
    "Sol Con/Enabler/SSE": ["Solutions Consultant", "Solutions Enabler", "Senior Software Engineer"],
    "Sr DS SME": [],
    "Sr Sol Con": ["Senior Solutions Consultant"],
}

def decode_resource_code(code) -> list[str]:
    if not isinstance(code, str):
        return []
    return list(RESOURCE_CODE_MAP.get(code.strip(), []))

def group_label(code) -> str:
    designations = decode_resource_code(code)
    if not designations:
        return f"{code} (no resolvable designation)"
    return " or ".join(designations)
