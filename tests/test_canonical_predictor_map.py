from scripts.canonical_predictor_map import canonical_map, normalize_canonical


def test_ilsa_codes():
    assert canonical_map("x", variable_code="ESCS") == "SES_COMPOSITE"
    assert canonical_map("x", variable_code="BSDGEDUP") == "PARENTAL_EDUCATION"


def test_keywords():
    assert canonical_map("Teacher support for students") == "TEACHER_QUALITY"
    assert canonical_map("Home educational resources") == "HOME_RESOURCES"


def test_aliases():
    assert normalize_canonical("BELONGING") == "SCHOOL_BELONGING"
