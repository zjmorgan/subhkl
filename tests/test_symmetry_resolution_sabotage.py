import pytest
from subhkl.core.spacegroup import is_systematically_absent, get_centering


def test_centering_mapping_vulnerability():
    """
    Reproduces the vulnerability where custom centring codes ('F', 'I', etc.)
    might not be correctly resolved to their expected extinction rules.
    """
    # Test 'F' centring (Face-centered)
    # Mixed parity h,k,l should be absent (e.g. 1,0,0)
    h, k, l = [1], [0], [0]

    # Check if 'F' correctly triggers face-centering absences
    absent = is_systematically_absent(h, k, l, "F")
    assert absent[0], "Centring 'F' should make (1,0,0) absent!"

    # Check if 'I' correctly triggers body-centering absences (h+k+l must be even)
    # (1,0,0) -> 1+0+0 = 1 (odd) -> should be absent
    absent_i = is_systematically_absent(h, k, l, "I")
    assert absent_i[0], "Centring 'I' should make (1,0,0) absent!"


def test_centering_type_resolution():
    """
    Verifies that the centering type returned matches the input code.
    """
    assert get_centering("F") == "F"
    assert get_centering("I") == "I"
    assert get_centering("A") == "A"
    assert get_centering("P") == "P"

    # 'H' is used in the code but is non-standard.
    # Gemmi might return 'P' or crash.
    try:
        assert get_centering("H") == "H"
    except Exception as e:
        pytest.fail(f"Bug Reproduced: 'H' centring failed! {e}")
