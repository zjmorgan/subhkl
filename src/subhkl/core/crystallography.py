import numpy as np
import numpy.typing as npt
import scipy.linalg

from subhkl.core.spacegroup import is_systematically_absent


def cartesian_matrix_metric_tensor(a, b, c, alpha, beta, gamma):
    """
    Calculates the B matrix (orientation matrix) and G* (reciprocal metric tensor).
    """
    G = np.array(
        [
            [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [b * a * np.cos(gamma), b**2, b * c * np.cos(alpha)],
            [c * a * np.cos(beta), c * b * np.cos(alpha), c**2],
        ]
    )
    Gstar = np.linalg.inv(G)
    B = scipy.linalg.cholesky(Gstar, lower=False)
    return B, Gstar


def generate_reflections(a, b, c, alpha, beta, gamma, space_group="P 1", d_min=2.0):
    """
    Generates unique HKL indices for a given unit cell and resolution cutoff.
    """
    if space_group is None:
        space_group = "P 1"
    constants = a, b, c, *np.deg2rad([alpha, beta, gamma])
    B, Gstar = cartesian_matrix_metric_tensor(*constants)

    astar, bstar, cstar = np.sqrt(np.diag(Gstar))

    h_max = int(np.floor(1 / d_min / astar))
    k_max = int(np.floor(1 / d_min / bstar))
    l_max = int(np.floor(1 / d_min / cstar))

    h, k, l = np.meshgrid(  # noqa: E741
        np.arange(-h_max, h_max + 1),
        np.arange(-k_max, k_max + 1),
        np.arange(-l_max, l_max + 1),
        indexing="ij",
    )

    h_flat, k_flat, l_flat = h.flatten(), k.flatten(), l.flatten()

    # Filter by resolution (1/d^2 = hkl . G* . hkl)
    hkl_sq = np.einsum(
        "ij,jl,il->l", Gstar, [h_flat, k_flat, l_flat], [h_flat, k_flat, l_flat]
    )

    with np.errstate(divide="ignore"):
        d = 1 / np.sqrt(hkl_sq)

    res_mask = (d > d_min) & (d < np.inf)
    absent_mask = is_systematically_absent(h_flat, k_flat, l_flat, space_group)

    final_mask = res_mask & (~absent_mask)

    return h_flat[final_mask], k_flat[final_mask], l_flat[final_mask]


def get_q_lab(
    h: npt.ArrayLike,
    k: npt.ArrayLike,
    l: npt.ArrayLike,  # noqa: E741
    RUB: npt.ArrayLike,
) -> npt.NDArray:
    """
    Calculate Q vectors in the Lab Frame.
    Q_lab = RUB * hkl
    RUB should be the composite matrix (R @ U @ B).
    """
    hkl = np.stack([h, k, l], axis=1)  # (N, 3)

    # Handle RUB shape: (3,3) or (N,3,3)
    if RUB.ndim == 3:
        # Einsum: n=batch, i=row, j=col. RUB[n,i,j] * hkl[n,j] -> out[n,i]
        q_lab = np.einsum("nij,nj->ni", RUB, hkl)
    else:
        # Standard matmul: hkl @ RUB.T
        q_lab = hkl @ RUB.T

    return q_lab
