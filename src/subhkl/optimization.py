import os
from functools import partial  # Added for static_argnames
import h5py
import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.interpolate

# --- JAX and Evosax Imports ---
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jscipy_linalg
# Corrected imports for DE and PSO
from evosax.algorithms import DifferentialEvolution, PSO
# ------------------------------

# Try to import tqdm for a progress bar
try:
    from tqdm import trange
except ImportError:
    trange = None

os.environ["OMP_NUM_THREADS"] = "1"


class VectorizedObjectiveJAX:
    """
    JAX-compatible vectorized objective function for evosax.
    (Replaces the numpy-based VectorizedObjective)
    """
    def __init__(self, B, kf_ki_dir, wavelength, angle_cdf, angle_t, tol=0.15):
        """
        Parameters
        ----------
        B : array (3, 3)
            B matrix (from reciprocal_lattice_B)
        kf_ki_dir : array (3, M)
            difference between incident and scattering directions for M
            reflections
        wavelength : array (2,)
            wavelength lower and upper bounds [min, max]
        angle_cdf : array
            CDF values for angle interpolation (from FindUB._angle_cdf)
        angle_t : array
            Angle values for interpolation (from FindUB._angle_t)
        tol : float
            Indexing tolerance
        """
        self.B = jnp.array(B)
        self.kf_ki_dir = jnp.array(kf_ki_dir)
        self.tol = tol
        self.angle_cdf = jnp.array(angle_cdf)
        self.angle_t = jnp.array(angle_t)

        # Ensure wavelength is a JAX array
        wavelength = jnp.array(wavelength)
        
        wl_min = jnp.full(self.kf_ki_dir.shape[1], wavelength[0])  # (M)
        wl_max = jnp.full(self.kf_ki_dir.shape[1], wavelength[1])  # (M)

        # Create 100 linearly spaced wavelengths for each reflection
        self.lamda = jnp.linspace(wl_min, wl_max, 100).T
        # (M, 100)

    def orientation_U_jax(self, param):
        """
        Compute orientation matrices (U) from angles using JAX.
        Implements Rodrigues' rotation formula.

        Parameters
        ----------
        param : array, (S, 3)
            Rotation parameters. S = population size.
            param[:, 0] = u0
            param[:, 1] = u1
            param[:, 2] = u2

        Returns
        -------
        U : array (S, 3, 3)
            Sample orientation matrices for each set of input parameters.
        """
        u0, u1, u2 = param.T # (S,) each

        theta = jnp.arccos(1 - 2 * u0)
        phi = 2 * jnp.pi * u1

        # Rotation axis (w)
        w = jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        ).T # (S, 3)

        # Rotation angle (omega)
        omega = jnp.interp(u2, self.angle_cdf, self.angle_t) # (S,)

        # JAX implementation of axis-angle to rotation matrix (Rodrigues' formula)
        wx, wy, wz = w.T # (S,) each
        c = jnp.cos(omega) # (S,)
        s = jnp.sin(omega) # (S,)
        t = 1.0 - c # (S,)

        # Identity matrices
        I = jnp.eye(3)[None, :, :].repeat(param.shape[0], axis=0) # (S, 3, 3)

        # Skew-symmetric cross-product matrix K
        K = jnp.array(
            [
                [jnp.zeros_like(wx), -wz, wy],
                [wz, jnp.zeros_like(wy), -wx],
                [-wy, wx, jnp.zeros_like(wz)]
            ]
        ) # (3, 3, S)
        K = jnp.transpose(K, (2, 0, 1)) # (S, 3, 3)

        # K^2
        K2 = jnp.einsum('sij,sjk->sik', K, K) # (S, 3, 3)
        
        # Rodrigues' formula
        U = I + s[:, None, None] * K + t[:, None, None] * K2 # (S, 3, 3)
        
        return U

    def indexer_jax(self, UB):
        """
        JAX-compatible Laue indexer for a given collection of :math:`UB` matrices

        Parameters
        ----------
        UB : array, (S, 3, 3)
            S, 3x3 sample oriented lattice matrices.

        Returns
        -------
        err : array, (S)
            Indexing cost for each UB
        num : array (S)
            Number of peaks indexed for each UB
        hkl : array (S, M, 3)
            Miller indices of peaks for each UB
        lamda : array (S, M)
            Resolved wavelength of each peak
        """
        UB_inv = jnp.linalg.inv(UB) # (S, 3, 3)

        hkl_lamda = jnp.einsum("sij,jm->sim", UB_inv, self.kf_ki_dir)
        # (S, 3, M), M = number of reflections

        hkl = hkl_lamda[:, :, :, None] / self.lamda[None, None, :, :]
        # (S, 3, M, 100)

        int_hkl = jnp.round(hkl)
        diff_hkl = hkl - int_hkl  # (S, 3, M, 100)

        dist = jnp.einsum("sij,sjmd->simd", UB, diff_hkl)
        # (S, 3, M, 100)

        dist = jnp.linalg.norm(dist, axis=1)  # (S, M, 100)
        ind = jnp.argmin(dist, axis=2, keepdims=True)  # (S, M, 1)

        err = jnp.take_along_axis(dist, ind, axis=2)[:, :, 0]  # (S, M)
        lamda = jnp.take_along_axis(self.lamda[None], ind, axis=2)[:, :, 0]
        # (S, M)

        hkl = hkl_lamda / lamda[:, None, :]  # (S, 3, M)

        int_hkl = jnp.round(hkl)
        diff_hkl = hkl - int_hkl  # (S, 3, M)

        mask = (jnp.abs(diff_hkl) < self.tol).all(axis=1)  # (S, M)
        num = jnp.sum(mask, axis=1)  # (S)

        # We minimize the sum of squared errors
        return jnp.sum(err**2, axis=1), num, int_hkl.transpose((0, 2, 1)), lamda

    # Use partial to make 'self' a static argument for JIT
    @partial(jax.jit, static_argnames='self')
    def __call__(self, x):
        """
        JIT-compiled objective function.

        Parameters
        ----------
        x : array (S, 3)
            Refineable parameters. S = population size

        Returns
        -------
        error : array (S,)
            Indexing error for each particle.
        """

        U = self.orientation_U_jax(x) # (S, 3, 3)
        
        UB = jnp.einsum("sij,jk->sik", U, self.B)
        # (S, 3, 3)

        error, num, hkl, lamda = self.indexer_jax(UB)

        # Return the error.
        # Clipping in the es_step function prevents NaNs.
        return error


class FindUB:
    """
    Optimizer of crystal orientation from peaks and known lattice parameters.
    ... (rest of class attributes) ...
    """

    def __init__(self, filename=None):
        """
        Find :math:`UB` from peaks.

        Parameters
        ----------
        filename : str, optional
            Filename of found peaks. The default is None.

        """

        if filename is not None:
            self.load_peaks(filename)

        t = np.linspace(0, np.pi, 1024)
        cdf = (t - np.sin(t)) / np.pi

        # Store data for JAX
        self._angle_cdf = cdf
        self._angle_t = t
        
        # Keep scipy interpolator for non-JAX methods
        self._angle = scipy.interpolate.interp1d(cdf, t, kind="linear")

    def load_peaks(self, filename):
        """
        Obtain peak information from .h5 file.

        Parameters
        ----------
        filename : str
            HDF5 file of peak information.

        """

        with h5py.File(os.path.abspath(filename), "r") as f:
            self.a = f["sample/a"][()]
            self.b = f["sample/b"][()]
            self.c = f["sample/c"][()]
            self.alpha = f["sample/alpha"][()]
            self.beta = f["sample/beta"][()]
            self.gamma = f["sample/gamma"][()]
            self.wavelength = f["instrument/wavelength"][()]
            self.R = f["goniometer/R"][()]
            self.two_theta = f["peaks/scattering"][()]
            self.az_phi = f["peaks/azimuthal"][()]
            self.centering = f["sample/centering"][()].decode("utf-8")

    def uncertainty_line_segements(self):
        """
        The scattering vector scaled with the (unknown) wavelength.

        Returns
        -------
        kf_ki_dir : list
            Difference between scattering and incident beam directions.

        """

        tt = np.deg2rad(self.two_theta)  # (M)
        az = np.deg2rad(self.az_phi)  # (M)

        kf_ki_dir = np.array(
            [np.sin(tt) * np.cos(az), np.sin(tt) * np.sin(az), np.cos(tt) - 1]
        )  # (3, M)

        return np.einsum("ji,jm->im", self.R, kf_ki_dir)
        # (3, M)

    def metric_G_tensor(self):
        """
        Calculate the metric tensor :math:`G`.

        Returns
        -------
        G : 2d-array
            3x3 matrix of lattice parameter info for Cartesian transforms.

        """

        alpha = np.deg2rad(self.alpha)
        beta = np.deg2rad(self.beta)
        gamma = np.deg2rad(self.gamma)

        g11 = self.a**2
        g22 = self.b**2
        g33 = self.c**2
        g12 = self.a * self.b * np.cos(gamma)
        g13 = self.c * self.a * np.cos(beta)
        g23 = self.b * self.c * np.cos(alpha)

        G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])

        return G

    def metric_G_star_tensor(self):
        """
        Calculate the reciprocal metric tensor :math:`G^*`.

        Returns
        -------
        Gstar : 2d-array
            3x3 matrix of reciprocal lattice info for Cartesian transforms.

        """

        return np.linalg.inv(self.metric_G_tensor())

    def reciprocal_lattice_B(self):
        """
        The reciprocal lattice :math:`B`-matrix.

        Returns
        -------
        B : 2d-array
            3x3 matrix of reciprocal lattice in Cartesian coordinates.

        """

        Gstar = self.metric_G_star_tensor()

        return scipy.linalg.cholesky(Gstar, lower=False)

    def orientation_U(self, u0, u1, u2):
        """
        The sample orientation matrix :math:`U`. (Scipy/Numpy version)

        Parameters
        ----------
        u0, u1, u2 : float
            Rotation parameters.

        Returns
        -------
        U : 2d-array
            3x3 sample orientation matrix.

        """

        theta = np.arccos(1 - 2 * u0)
        phi = 2 * np.pi * u1
        w = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        ax = self._angle(u2)

        return scipy.spatial.transform.Rotation.from_rotvec(ax * w).as_matrix()

    def indexer(self, UB, kf_ki_dir, d_min, d_max, wavelength, tol=0.1):
        """
        (Original scipy/numpy indexer for post-optimization check)
        ... (rest of method) ...
        """

        wl_min, wl_max = wavelength

        UB_inv = np.linalg.inv(UB)

        hkl_lamda = np.einsum("ij,jk", UB_inv, kf_ki_dir)

        lamda = np.linspace(wl_min, wl_max, 100)

        hkl = hkl_lamda[:, :, np.newaxis] / lamda

        s = np.einsum("ij,j...->i...", UB, hkl)
        s = np.linalg.norm(s, axis=0)

        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        dist = np.einsum("ij,j...->i...", UB, diff_hkl)
        dist = np.linalg.norm(dist, axis=0)

        dist[(s.T > 1 / d_min).T] = np.inf
        dist[(s.T < 1 / d_max).T] = np.inf

        h, k, l = int_hkl  # noqa: E741

        valid = np.full_like(l, True, dtype=bool)

        if self.centering == "A":
            valid = (k + l) % 2 == 0
        elif self.centering == "B":
            valid = (h + l) % 2 == 0
        elif self.centering == "C":
            valid = (h + k) % 2 == 0
        elif self.centering == "I":
            valid = (h + k + l) % 2 == 0
        elif self.centering == "F":
            valid = ((h + k) % 2 == 0) & ((l + h) % 2 == 0) & ((k + l) % 2 == 0)
        elif self.centering == "R_obv":
            valid = (-h + k + l) % 3 == 0
        elif self.centering == "R_obv":
            valid = (h - k + l) % 3 == 0

        dist[~valid] = np.inf

        ind = np.argmin(dist, axis=1)

        hkl = hkl[:, np.arange(hkl_lamda.shape[1]), ind]
        lamda = lamda[ind]

        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        mask = (np.abs(diff_hkl) < tol).all(axis=0)
        int_hkl[:, ~mask] = 0

        num = np.sum(mask)

        return num, int_hkl.T, lamda

    def UB_matrix(self, U, B):
        """
        Calculate :math:`UB`-matrix.
        ... (rest of method) ...
        """

        return U @ B

    def indexer_de(self, UB, kf_ki_dir, wavelength, tol=0.15):
        """
        (Original numpy-based indexer for post-optimization)
        ... (rest of method) ...
        """
        wl_min = np.full(kf_ki_dir.shape[1], wavelength[0])  # (M)
        wl_max = np.full(kf_ki_dir.shape[1], wavelength[1])  # (M)
        x = np.linspace(0, 1, 100)
        lamda = wl_min[:, None] + (wl_max - wl_min)[:, None] * x[None, :]

        UB_inv = np.linalg.inv(UB)

        hkl_lamda = np.einsum("ij,jk", UB_inv, kf_ki_dir)
        hkl = hkl_lamda[:, :, np.newaxis] / lamda[np.newaxis, :, :]
        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        dist = np.einsum("ij,j...->i...", UB, diff_hkl)
        dist = np.linalg.norm(dist, axis=0)

        ind = np.argmin(dist, axis=1)
        err = dist[np.arange(dist.shape[0]), ind]

        lamda = lamda[np.arange(lamda.shape[0]), ind]
        hkl = hkl_lamda / lamda
        int_hkl = np.round(hkl)
        diff_hkl = hkl - int_hkl

        mask = (np.abs(diff_hkl) < tol).all(axis=0)
        num = np.sum(mask)

        return np.sum(err ** 2), num, int_hkl.T, lamda

    def index_de(self):
        """
        Run indexing using the best parameters (self.x) found by a minimizer.
        Uses the original numpy-based indexer_de.
        """
        kf_ki_dir = self.uncertainty_line_segements()

        B = self.reciprocal_lattice_B()
        
        # self.x should be (3,) numpy array
        if not isinstance(self.x, np.ndarray):
            self.x = np.array(self.x) 
            
        U = self.orientation_U(*self.x)

        UB = self.UB_matrix(U, B)

        return self.indexer_de(UB, kf_ki_dir, self.wavelength)[1:]

    def minimize_evosax(
        self, 
        strategy_name: str, 
        population_size: int = 1000, 
        num_generations: int = 100, 
        n_runs: int = 1, 
        seed: int = 0
    ):
        """
        Minimize the objective function using evosax JAX-based algorithms.
        This replaces both the pyswarms (minimize) and scipy.DE (minimize_de) methods.
        
        It runs the optimization `n_runs` times with different seeds and
        selects the best solution.

        Parameters
        ----------
        strategy_name : str
            Name of the evosax strategy to use (e.g., 'DE' or 'PSO').
        population_size : int
            Population size for the strategy.
        num_generations : int
            Number of generations to run the optimization.
        n_runs : int
            Number of times to run the minimization with different seeds.
        seed : int
            Base seed for the random number generator.

        Returns
        -------
        (num, hkl, lamda)
            Tuple containing results from index_de().
        """
        
        kf_ki_dir = self.uncertainty_line_segements()

        # 1. Instantiate the JAX-compatible objective function
        objective = VectorizedObjectiveJAX(
            self.reciprocal_lattice_B(),
            kf_ki_dir,
            np.array(self.wavelength), # Pass wavelength bounds
            self._angle_cdf,           # Pass interpolation data
            self._angle_t,             # Pass interpolation data
        )

        # Define the sample solution (shape (3,)) to infer num_dims
        sample_solution = jnp.zeros(3)

        # 2. Initialize strategy
        # clip_min/clip_max removed from constructor
        if strategy_name.lower() == "de":
            strategy = DifferentialEvolution(
                solution=sample_solution,
                population_size=population_size
            )
            print("Using Differential Evolution (DE) strategy.")
        elif strategy_name.lower() == "pso":
            strategy = PSO(
                solution=sample_solution,
                population_size=population_size
            )
            print("Using Particle Swarm Optimization (PSO) strategy.")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}. Choose 'DE' or 'PSO'.")

        
        # 3. Get default strategy parameters
        params = strategy.default_params
        
        # 4. Define the JIT-compiled step function
        @jax.jit
        def es_step(rng, state, params):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)
            
            # Ask for new population
            population, state = strategy.ask(rng_ask, state, params)
            
            # --- Manually clip the population to [0, 1] ---
            population_clipped = jnp.clip(population, 0.0, 1.0)
            
            # Evaluate the *clipped* population
            fitness = objective(population_clipped)
            
            # Tell strategy the results using the *clipped* population
            state, metrics = strategy.tell(
                rng_tell, population_clipped, fitness, state, params
            )
            return rng, state, metrics

        # 5. Run the optimization loop N times
        best_overall_fitness = jnp.inf
        best_overall_member = None
        
        for i in range(n_runs):
            run_seed = seed + i
            print(f"\n--- Starting Run {i+1}/{n_runs} (Seed: {run_seed}) ---")
            
            # 6. Initialize strategy state for this run
            rng = jax.random.PRNGKey(run_seed)
            
            # Create initial population and fitness
            rng, rng_pop, rng_init = jax.random.split(rng, 3)
            # Population is (popsize, 3) random numbers [0, 1]
            population_init = jax.random.uniform(rng_pop, (population_size, 3))
            
            # Evaluate the initial population
            fitness_init = objective(population_init)
            
            # Initialize state using .init()
            state = strategy.init(rng_init, population_init, fitness_init, params) 
            
            pbar = range(num_generations)
            if trange is not None:
                pbar = trange(num_generations, desc=f"Run {i+1}/{n_runs}")

            for gen in pbar:
                # Run one step
                rng, state, metrics = es_step(rng, state, params)
                
                # Update progress bar description
                if trange is not None:
                    pbar.set_description(f"Run {i+1} Gen: {gen+1}/{num_generations} | Best Fitness: {metrics['best_fitness']:.4f}")
            
            # --- Correctly get best member and fitness ---
            # Access the attributes from the *final state*
            current_run_fitness = state.best_fitness
            current_run_member = state.best_solution
            print(f"Run {i+1} finished. Best fitness: {current_run_fitness:.4f}")
            
            # 8. Check if this run is the best so far
            if current_run_fitness < best_overall_fitness:
                best_overall_fitness = current_run_fitness
                best_overall_member = current_run_member # This is now the jax array
                print(f"!!! New best solution found in Run {i+1} !!!")

        # 9. Get the best parameters from all runs
        print(f"\n--- All {n_runs} runs complete ---")
        print(f"Best overall fitness: {best_overall_fitness:.4f}")
        print(f"Best parameters (u0, u1, u2): {best_overall_member}")
        
        # Store the best parameters (converting back to numpy)
        self.x = np.array(best_overall_member)
        
        # 10. Return results by running the numpy-based indexer on the best params
        return self.index_de()
