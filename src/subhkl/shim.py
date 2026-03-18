import numpy as np

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

        class _TreeShim:
            @staticmethod
            def map(f, *trees):
                """A basic pure-Python/NumPy shim for jax.tree.map."""
                if not trees:
                    return None

                first = trees[0]
                if isinstance(first, (list, tuple)):
                    return type(first)(
                        jax.tree.map(f, *[t[i] for t in trees])
                        for i in range(len(first))
                    )
                elif isinstance(first, dict):
                    return {
                        k: jax.tree.map(f, *[t[k] for t in trees]) for k in first.keys()
                    }
                else:
                    return f(*trees)

        lax = _LaxShim()
        nn = _NnShim()
        scipy = _ScipyShim()
        tree = _TreeShim()

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

