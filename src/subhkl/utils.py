import typing

import numpy as np
import numpy.typing as npt
import scipy.linalg

from subhkl.spacegroup import is_systematically_absent

if typing.TYPE_CHECKING:
    from subhkl.detector import Detector

# ==============================================================================
# JAX Import with Fallback (shared across modules)
# ==============================================================================

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jscipy_linalg
    import jax.scipy.optimize
    import jax.scipy.signal
    from jax import jit as jit
    from jax import vmap as vmap
    from jax import lax as lax
    from evosax.algorithms import CMA_ES, PSO, DifferentialEvolution
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    HAS_JAX = True
    OPTIMIZATION_BACKEND = "jax"
except Exception:
    # Fallback shim: expose a minimal `jax`-like object and map jax.numpy
    # to the installed NumPy so code using `jnp` still works.

    class _JaxShim:
        """Minimal JAX shim for when JAX is not installed."""

        @staticmethod
        def jit(f=None, *, static_argnames=None, **kwargs):
            if f is None:
                return lambda fn: fn
            return f

        @staticmethod
        def vmap(fun, in_axes=0, out_axes=0):
            """A basic pure-Python/NumPy shim for jax.vmap."""

            def batched_fun(*args):
                # Normalize in_axes to a tuple so it pairs with args
                axes = (
                    in_axes
                    if isinstance(in_axes, (tuple, list))
                    else (in_axes,) * len(args)
                )

                # Determine the batch size by looking at the first mapped argument
                batch_size = None
                for arg, axis in zip(args, axes):
                    if axis is not None:
                        batch_size = arg.shape[axis]
                        break

                # If no axes are mapped, just return the standard function
                if batch_size is None:
                    return fun(*args)

                # Run the function in a loop over the batch size
                results = []
                for i in range(batch_size):
                    unbatched_args = []
                    for arg, axis in zip(args, axes):
                        if axis is not None:
                            # Extract the slice for this specific batch index
                            unbatched_args.append(np.take(arg, i, axis=axis))
                        else:
                            # Pass the argument as-is (e.g., for broadcasting)
                            unbatched_args.append(arg)

                    # Apply the core function to the single slice
                    results.append(fun(*unbatched_args))

                # Stack the collected results back together along the requested out_axes
                return np.stack(results, axis=out_axes)

            return batched_fun

        class _LaxShim:
            @staticmethod
            def scan(f, init, xs, length=None):
                if xs is None:
                    xs = [None] * length
                carry = init
                ys = []
                for x in xs:
                    carry, y = f(carry, x)
                    ys.append(y)
                return carry, np.stack(ys)

            @staticmethod
            def dynamic_slice(operand, start_indices, slice_sizes):
                """A NumPy shim for jax.lax.dynamic_slice."""
                operand = np.asarray(operand)

                if (
                    len(start_indices) != operand.ndim
                    or len(slice_sizes) != operand.ndim
                ):
                    raise ValueError(
                        "start_indices and slice_sizes must match the rank of the operand."
                    )

                slices = []
                for start, size, dim_size in zip(
                    start_indices, slice_sizes, operand.shape
                ):
                    # JAX clamps the start index so the requested slice size always fits
                    max_valid_start = max(0, dim_size - size)
                    clamped_start = min(max(0, start), max_valid_start)

                    # Build the slice object for this dimension
                    slices.append(slice(clamped_start, clamped_start + size))

                return operand[tuple(slices)]

            @staticmethod
            def top_k(operand, k):
                """A NumPy shim for jax.lax.top_k."""
                operand = np.asarray(operand)

                # Sort indices along the last axis.
                # np.argsort sorts ascending, so the largest values are at the end.
                sorted_indices = np.argsort(operand, axis=-1)

                # Slice the last k indices, then reverse the step ([::-1]) to make it descending
                top_indices = sorted_indices[..., -k:][..., ::-1]

                # Use the indices to gather the actual values from the original array
                top_values = np.take_along_axis(operand, top_indices, axis=-1)

                return top_values, top_indices

        class _NnShim:
            @staticmethod
            def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
                from scipy.special import logsumexp

                return logsumexp(
                    a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign
                )

        class _ScipyShim:
            class _SpecialShim:
                @staticmethod
                def logit(x):
                    from scipy.special import logit

                    return logit(x)

            class _SignalShim:
                @staticmethod
                def correlate2d(in1, in2, mode="same", boundary="fill", fillvalue=0):
                    from scipy.signal import correlate2d

                    return correlate2d(
                        in1,
                        in2,
                        mode=mode,
                        boundary=boundary,
                        fillvalue=fillvalue,
                    )

                @staticmethod
                def convolve2d(in1, in2, mode="same", boundary="fill", fillvalue=0):
                    from scipy.signal import convolve2d

                    return convolve2d(
                        in1,
                        in2,
                        mode=mode,
                        boundary=boundary,
                        fillvalue=fillvalue,
                    )

            class _OptimizeShim:
                @staticmethod
                def minimize(fun, x0, args=(), method=None, tol=None, options=None):
                    from scipy.optimize import minimize

                    return minimize(
                        fun,
                        x0,
                        args=args,
                        method=method,
                        tol=tol,
                        options=options,
                    )

            special = _SpecialShim()
            signal = _SignalShim()
            optimize = _OptimizeShim()

        lax = _LaxShim()
        nn = _NnShim()
        scipy = _ScipyShim()

    jax = _JaxShim()
    jnp = np
    jit = jax.jit

    class _JscipyLinalgShim:
        @staticmethod
        def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
            import scipy.linalg

            # Handle batching manually if needed, or use np.vectorize
            if a.ndim > 2:
                # We can use our vmap shim!
                def single_cholesky(matrix):
                    return scipy.linalg.cholesky(
                        matrix,
                        lower=lower,
                        overwrite_a=overwrite_a,
                        check_finite=check_finite,
                    )

                return jax.vmap(single_cholesky)(a)
            return scipy.linalg.cholesky(
                a,
                lower=lower,
                overwrite_a=overwrite_a,
                check_finite=check_finite,
            )

    jscipy_linalg = _JscipyLinalgShim()
    DifferentialEvolution = None
    PSO = None
    CMA_ES = None
    Mesh = None
    NamedSharding = None
    P = None
    HAS_JAX = False
    OPTIMIZATION_BACKEND = "numpy"
    jit = jax.jit
    lax = jax.lax
    vmap = jax.vmap
    nn = jax.nn
    jax_scipy = jax.scipy


def jnp_update_add(arr, idx, val):
    """Immutable update: arr[idx] += val"""
    if HAS_JAX:
        return arr.at[idx].add(val)
    else:
        res = arr.copy()
        res[idx] += val
        return res


def jnp_update_set(arr, idx, val):
    """Immutable update: arr[idx] = val"""
    if HAS_JAX:
        return arr.at[idx].set(val)
    else:
        res = arr.copy()
        res[idx] = val
        return res


def scale_coordinates(xp, yp, scale_x, scale_y, nx, ny):
    """
    Scale pixel coordinates to physical coordinates.

    Parameters
    ----------
    xp : float
        Pixel x-coordinate
    yp : float
        Pixel y-coordinate
    scale_x : float
        Scale factor in x direction (m/pixel)
    scale_y : float
        Scale factor in y direction (m/pixel)
    nx : int
        Number of pixels in x direction
    ny : int
        Number of pixels in y direction

    Returns
    -------
    x, y : tuple of float
        Physical coordinates in meters
    """
    x = (xp - nx / 2) * scale_x
    y = (yp - ny / 2) * scale_y
    return x, y


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


def calculate_angular_error(
    xyz_det: npt.NDArray,
    h: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,  # noqa: E741
    lam: npt.NDArray,
    RUB: npt.NDArray,
    sample_offset: npt.NDArray = None,
    ki_vec: npt.NDArray = None,
    R_all: npt.NDArray = None,
):
    """
    Calculate D-spacing and Angular errors for observed peaks vs predicted geometry.
    Uses the RUB matrix (R @ U @ B) for all coordinate transformations.
    """
    if sample_offset is None:
        sample_offset = np.zeros(3)
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])

    # 1. Calculate Q_calc (Lab Frame)
    q_lab_calc = get_q_lab(h, k, l, RUB)
    q_calc_norm = q_lab_calc / np.linalg.norm(q_lab_calc, axis=1, keepdims=True)

    # 2. Calculate Q_obs (Lab Frame) from Detector Pixel Position
    # v = Pixel_Position - Sample_Position
    if R_all is not None:
        if R_all.ndim == 3:
            s_lab = np.einsum("nij,j->ni", R_all, sample_offset)
        else:
            s_lab = R_all @ sample_offset
        v = xyz_det - s_lab
    else:
        v = xyz_det - sample_offset

    dist = np.linalg.norm(v, axis=1, keepdims=True)
    kf_dir = v / dist  # Unit vector pointing from sample to pixel

    ki_dir = ki_vec / np.linalg.norm(ki_vec)

    # Scattering vector direction matches delta_k = kf - ki
    delta_k = kf_dir - ki_dir[None, :]
    two_sin_theta = np.linalg.norm(delta_k, axis=1)

    # Q_obs direction
    with np.errstate(divide="ignore", invalid="ignore"):
        q_obs_norm = delta_k / two_sin_theta[:, None]

    # 3. Angular Error (Angle between Q_calc direction and Q_obs direction)
    dot = np.sum(q_obs_norm * q_calc_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_err = np.rad2deg(np.arccos(dot))

    # 4. D-Spacing Error
    # d_obs = lambda / 2sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_obs = np.divide(lam, two_sin_theta)

    # d_calc = 1 / |RUB * hkl|
    # Since R and U are rotations (unitary), they preserve length.
    # |R * U * B * h| == |B * h| == 1/d
    q_lab_mag = np.linalg.norm(q_lab_calc, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        d_calc = np.divide(1.0, q_lab_mag)

    d_err = np.abs(d_obs - d_calc)

    return d_err, ang_err


def predict_reflections_on_panel(
    detector: "Detector",
    h: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,  # noqa: E741
    RUB: npt.NDArray,
    wavelength_min: float,
    wavelength_max: float,
    sample_offset: npt.NDArray = None,
    ki_vec: npt.NDArray = None,
    R_all: npt.NDArray = None,
):
    """
    Predicts which HKLs fall on a specific detector panel using the RUB matrix.
    Returns: (row, col, h, k, l, wavelength)
    """
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])
    ki_hat = ki_vec / np.linalg.norm(ki_vec)

    # 1. Calculate Q vectors (Units: 2pi/d)
    # get_q_lab returns 1/d units. Multiply by 2pi.
    q_lab_direction = get_q_lab(h, k, l, RUB)
    Q_vecs = 2 * np.pi * q_lab_direction.T  # Shape (3, N)

    Q_sq = np.sum(Q_vecs**2, axis=0)

    # 2. Calculate Wavelength (Generalized Laue)
    # lambda = -4pi * (Q . ki) / Q^2
    Q_dot_ki = np.sum(Q_vecs * ki_hat[:, None], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        lamda = -4 * np.pi * Q_dot_ki / Q_sq

    # 3. Filter Wavelength
    mask = (lamda > wavelength_min) & (lamda < wavelength_max)
    if not np.any(mask):
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    Q_vecs = Q_vecs[:, mask]
    lamda = lamda[mask]
    h, k, l = h[mask], k[mask], l[mask]  # noqa: E741

    # 4. Calculate kf direction
    k_mag = 2 * np.pi / lamda
    kf_vecs = Q_vecs + k_mag * ki_hat[:, None]

    kf_norms = np.linalg.norm(kf_vecs, axis=0)
    kf_dirs = kf_vecs / kf_norms  # Shape (3, N_filtered)

    x, y, z = kf_dirs[0], kf_dirs[1], kf_dirs[2]

    # 5. Ray Trace intersection with Panel
    # CORRECTED: Rotate sample offset to Lab frame
    if R_all is not None and sample_offset is not None:
        if R_all.ndim == 3:
            s_lab = np.einsum("nij,j->ni", R_all, sample_offset)
        else:
            s_lab = R_all @ sample_offset
    else:
        s_lab = sample_offset

    mask_panel, row, col = detector.reflections_mask(x, y, z, sample_offset=s_lab)

    return (
        row[mask_panel],
        col[mask_panel],
        h[mask_panel],
        k[mask_panel],
        l[mask_panel],
        lamda[mask_panel],
    )
