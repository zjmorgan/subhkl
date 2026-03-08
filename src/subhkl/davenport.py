import jax.numpy as jnp
from jax import jit, vmap

@jit
def davenport_pair(v_pair, u_pair, w_pair):
    """
    Calculates the optimal quaternion and eigenvalue for two matched Zone Axes.
    """
    B = jnp.sum(w_pair[:, None, None] * (v_pair[:, :, None] @ u_pair[:, None, :]), axis=0)
    S = B + B.T
    sigma = jnp.trace(B)
    
    Z = jnp.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])
    
    K_top = jnp.concatenate([jnp.array([sigma]), Z])[None, :]
    K_bottom = jnp.concatenate([Z[:, None], S - sigma * jnp.eye(3)], axis=1)
    K = jnp.concatenate([K_top, K_bottom], axis=0)
    
    evals, evecs = jnp.linalg.eigh(K)
    q_passive = evecs[:, -1]
    
    # --- Convert Passive (Frame) to Active (Vector) Rotation ---
    q_active = jnp.array([q_passive[0], -q_passive[1], -q_passive[2], -q_passive[3]])
    
    return q_active, evals[-1]

batch_davenport = jit(vmap(davenport_pair, in_axes=(0, 0, 0)))
