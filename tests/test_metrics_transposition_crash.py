import pytest


def test_generate_reflections_none_sg_crash():
    """
    Reproduces the crash in generate_reflections when space_group is None.
    The code defaults to 'P 1' in the signature, but if passed explicitly
    as None, it will crash.
    """
    from subhkl.core.crystallography import generate_reflections

    # Unit cell
    a, b, c = 10, 10, 10
    alpha, beta, gamma = 90, 90, 90

    # This should NOT crash with AttributeError: 'NoneType' has no attribute 'upper'
    # or similar in is_systematically_absent.
    try:
        h, k, l = generate_reflections(a, b, c, alpha, beta, gamma, space_group=None)
    except Exception as e:
        if "'NoneType' object has no attribute" in str(e):
            pytest.fail(
                f"Bug Reproduced: generate_reflections crashed with space_group=None! {e}"
            )
        raise e
