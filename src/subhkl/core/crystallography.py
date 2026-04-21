from dataclasses import dataclass
import warnings

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import numpy.typing as npt
import scipy.linalg

from subhkl.core.spacegroup import is_systematically_absent
from subhkl.core.models import (
    LatticeSystem,
    _Params,
    LATTICE_CONFIG,
    LATTICE_CONSTRAINTS,
    SG_SYSTEM_MAP,
)


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


@dataclass(frozen=True, slots=True)
class Lattice:
    """
    Represents a crystal lattice and handles transformations between
    physical parameters and reciprocal space matrices.
    """

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    system: LatticeSystem = LatticeSystem.TRICLINIC

    def validate(self):
        if any(p <= 0 for p in (self.a, self.b, self.c)):
            raise ValueError(
                f"Lattice cell dimensions must be positive. Got: a={self.a}, b={self.b}, c={self.c}"
            )

        if not all(0 < ang < 180 for ang in (self.alpha, self.beta, self.gamma)):
            raise ValueError(
                f"Lattice angles must be (0, 180). Got: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}"
            )

        ar, br, gr = map(np.radians, (self.alpha, self.beta, self.gamma))
        metric_check = (
            1
            - np.cos(ar) ** 2
            - np.cos(br) ** 2
            - np.cos(gr) ** 2
            + 2 * np.cos(ar) * np.cos(br) * np.cos(gr)
        )
        if metric_check <= 0:
            raise ValueError(
                f"Invalid Lattice angles: The combination of angles ({self.alpha}, {self.beta}, {self.gamma})"
                "creates a physically impossible (collapsed) volume"
            )

    # NOTE(vivek): need to make a design decision about this at some point
    @classmethod
    def from_params(cls, params: jnp.ndarray, system: LatticeSystem):
        full_params = LATTICE_CONFIG[system]["reconstruct"](params)

        # batched input, return stacked params
        if params.ndim > 1:
            return jnp.stack(full_params, axis=-1)
        return cls(*full_params, system=system)

    @staticmethod
    def get_active_indices(system):
        return LATTICE_CONFIG[system]["active_indices"]

    def to_numpy(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])

    def to_jax(self) -> jnp.ndarray:
        return jnp.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma])

    def get_b_matrix(self):
        """
        Calculates the reciprocal lattice B matrix using the Busing-Levy convention.
        B = Cholesky(G*), where G* is the reciprocal metric tensor.
        """
        a_rad = jnp.deg2rad(self.alpha)
        b_rad = jnp.deg2rad(self.beta)
        g_rad = jnp.deg2rad(self.gamma)

        g11, g22, g33 = self.a**2, self.b**2, self.c**2
        g12 = self.a * self.b * jnp.cos(g_rad)
        g13 = self.a * self.c * jnp.cos(b_rad)
        g23 = self.b * self.c * jnp.cos(a_rad)

        G = jnp.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])

        # Reciprocal Metric Tensor G* = inv(G)
        G_star = jnp.linalg.inv(G)

        # B matrix is the upper triangular Cholesky decomposition
        # B transforms HKL indices to Cartesian reciprocal coordinates: q = B @ hkl
        return jsl.cholesky(G_star, lower=False)

    # NOTE(vivek): uses numpy
    @classmethod
    def from_b_matrix(cls, B, system: LatticeSystem = LatticeSystem.TRICLINIC):
        G_star = B.T @ B
        G = np.linalg.inv(G_star)
        a = np.sqrt(G[0, 0])
        b = np.sqrt(G[1, 1])
        c = np.sqrt(G[2, 2])

        alpha = np.rad2deg(np.arccos(G[1, 2] / (b * c)))
        beta = np.rad2deg(np.arccos(G[0, 2] / (a * c)))
        gamma = np.rad2deg(np.arccos(G[0, 1] / (a * b)))

        return cls(a, b, c, alpha, beta, gamma, system)

    @staticmethod
    def check_constraints(
        params: np.ndarray, system: LatticeSystem, atol_len: float, atol_ang: float
    ):
        """Validates parameters against symmetry rules defined in metadata."""
        constraints = LATTICE_CONSTRAINTS[system]
        violations = []

        for i, j in constraints.get("equal_lengths", []):
            if not np.isclose(params[i], params[j], atol=atol_len):
                violations.append(f"{'abc'[i]}={'abc'[j]}")

        for i, j in constraints.get("equal_angles", []):
            if not np.isclose(params[i], params[j], atol=atol_ang):
                violations.append(
                    f"{['alpha', 'beta', 'gamma'][i - 3]}={['alpha', 'beta', 'gamma'][j - 3]}"
                )

        for idx, val in constraints.get("fixed_angles", {}).items():
            if not np.isclose(params[idx], val, atol=atol_ang):
                violations.append(f"{['alpha', 'beta', 'gamma'][idx - 3]}={val}")

        return violations

    @staticmethod
    def infer_system(
        lattice_cell,
        space_group: str,
        atol_len=0.05,
        atol_ang=0.5,
        return_type: str = "enum",
    ):
        if isinstance(lattice_cell, Lattice):
            params = lattice_cell.to_numpy()
        else:
            params = np.asarray(lattice_cell)

        try:
            from subhkl.core.spacegroup import get_space_group_object

            sg = get_space_group_object(space_group)
            sys_str = str(sg.crystal_system()).split(".")[-1].lower()
            centering = sg.centring_type()

            expected = SG_SYSTEM_MAP.get(sys_str, LatticeSystem.TRICLINIC)
            if sys_str == "trigonal":
                expected = (
                    LatticeSystem.RHOMBOHEDRAL
                    if centering == "R"
                    else LatticeSystem.HEXAGONAL
                )
        except Exception:
            expected = LatticeSystem.TRICLINIC
            centering = "P"

        # check highest symmetry that satisfies constraints
        geometric = LatticeSystem.TRICLINIC
        for sys in reversed(LatticeSystem):
            if not Lattice.check_constraints(params, sys, atol_len, atol_ang):
                geometric = sys
                break

        violations = Lattice.check_constraints(params, expected, atol_len, atol_ang)
        if violations:
            warnings.warn(
                f"\n[Lattice System] Input parameters violate {space_group} ({expected}) constraints: {', '.join(violations)}.\n"
                f"optimization will enforce {expected} constraints, which may cause a jump in parameters.",
                stacklevel=2,
            )

        if expected < geometric:
            warnings.warn(
                f"Lattice System Override: Geometry suggests {geometric.name}, "
                f"but Space Group requires {expected.name}. Enforcing Lower Symmetry."
            )

        num_params = LATTICE_CONFIG[expected]["num_params"]
        match return_type:
            case "str":
                return LATTICE_CONFIG[expected]["name"], num_params
            case "enum":
                return expected, num_params
            case "lattice":
                return Lattice(*lattice_cell, system=expected), num_params
            case _:
                raise ValueError(f"Unknown return type: {return_type}")


class LatticeSOA:
    """Dedicated class for structure of array oriented lattices"""

    @staticmethod
    def compute_B_batched(batched_lattice):
        deg2rad = jnp.pi / 180.0
        a = batched_lattice[:, _Params.A]
        b = batched_lattice[:, _Params.B]
        c = batched_lattice[:, _Params.C]
        alpha = batched_lattice[:, _Params.ALPHA] * deg2rad
        beta = batched_lattice[:, _Params.BETA] * deg2rad
        gamma = batched_lattice[:, _Params.GAMMA] * deg2rad

        g11, g22, g33 = a**2, b**2, c**2
        g12 = a * b * jnp.cos(gamma)
        g13 = a * c * jnp.cos(beta)
        g23 = b * c * jnp.cos(alpha)

        row1 = jnp.stack([g11, g12, g13], axis=-1)
        row2 = jnp.stack([g12, g22, g23], axis=-1)
        row3 = jnp.stack([g13, g23, g33], axis=-1)

        G = jnp.stack([row1, row2, row3], axis=-2)
        G_star = jnp.linalg.inv(G)

        return jsl.cholesky(G_star, lower=False)
