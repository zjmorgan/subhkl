import os
from functools import partial

import h5py
import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.interpolate

import gemmi

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jscipy_linalg

from evosax.algorithms import DifferentialEvolution, PSO, CMA_ES

# Try to import tqdm for a progress bar
try:
    from tqdm import trange
except ImportError:
    trange = None


class VectorizedObjectiveJAX:
    """
    JAX-compatible vectorized objective function for evosax.
    (Replaces the numpy-based VectorizedObjective)
    """
    def __init__(self, B, centering, kf_ki_dir, wavelength, angle_cdf, angle_t, weights=None, softness=0.15):
        """
        Parameters
        ----------
        B : array (3, 3)
            B matrix (from reciprocal_lattice_B)
        centering : stsr
            Bravais lattice centering
        kf_ki_dir : array (3, M)
            difference between incident and scattering directions for M
            reflections
        wavelength : array (2,)
            wavelength lower and upper bounds [min, max]
        angle_cdf : array
            CDF values for angle interpolation (from FindUB._angle_cdf)
        angle_t : array
            Angle values for interpolation (from FindUB._angle_t)
        softness : float
            Shape parameter for rounding hkls
        """
        self.B = jnp.array(B)
        self.kf_ki_dir = jnp.array(kf_ki_dir)
        self.softness = softness
        self.centering = centering
        self.angle_cdf = jnp.array(angle_cdf)
        self.angle_t = jnp.array(angle_t)

        # Ensure wavelength is a JAX array
        wavelength = jnp.array(wavelength)

        wl_min = jnp.full(self.kf_ki_dir.shape[1], wavelength[0])  # (M)
        wl_max = jnp.full(self.kf_ki_dir.shape[1], wavelength[1])  # (M)

        # Create 100 linearly spaced wavelengths for each reflection
        self.lamda = jnp.linspace(wl_min, wl_max, 100).T
        # (M, 100)

        # Handle weights: if None, default to 1.0 for everyone
        if weights is None:
            self.weights = jnp.ones(self.kf_ki_dir.shape[1])
        else:
            self.weights = jnp.array(weights)

        # Pre-calculate the maximum possible score (sum of all weights)
        # This is the score if every peak is perfectly indexed.
        self.max_score = jnp.sum(self.weights)

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

    def indexer_soft_jax(self, UB, softness=0.001):
        """
        Weighted soft-indexing objective.
        """

        UB_inv = jnp.linalg.inv(UB) # (S, 3, 3)

        # map to HKL space
        hkl_lamda = jnp.einsum("sij,jm->sim", UB_inv, self.kf_ki_dir)
        hkl = hkl_lamda[:, :, :, None] / self.lamda[None, None, :, :]
        # (S, 3, M, 100)

        # Smooth periodic distance: instead of `hkl - round(hkl)`, use Sine.
        diff_hkl_smooth = jnp.sin(jnp.pi * hkl) / jnp.pi

        # map error back to Cartesian q-space
        # (S, 3, 3) @ (S, 3, M, 100) -> (S, 3, M, 100)
        dist_vec = jnp.einsum("sij,sjmd->simd", UB, diff_hkl_smooth)

        # Squared Euclidean distance for every wavelength candidate
        dist_sq = jnp.sum(dist_vec**2, axis=1) # (S, M, 100)

        valid = jnp.full_like(hkl[:, 0], True, dtype=bool)

        h, k, l = jnp.round(hkl).transpose(1, 0, 2, 3)

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
        elif self.centering == "R":
            valid = (h + k + l) % 3 == 0
        elif self.centering == "R_obv":
            valid = (-h + k + l) % 3 == 0
        elif self.centering == "R_rev":
            valid = (h - k + l) % 3 == 0

        dist_sq = jnp.where(valid, dist_sq, jnp.inf)

        # dist_sq shape: (S, M, 100)

        # soft Wavelength Selection
        min_dist_sq = jnp.min(dist_sq, axis=2) # (S, M)

        # calculate probability of fit (0 to 1) for each peak
        peak_probs = jnp.exp(-min_dist_sq / (2 * softness**2)) # (S, M)

        # apply weights: Strong peaks contribute more to the score
        weighted_scores = peak_probs * self.weights[None, :] # (S, M)

        # If weights are normalized so mean(w)=1, this value is intuitively
        # "How many average-quality peaks did we index?"
        total_score = jnp.sum(weighted_scores, axis=1) # (S,)

        # for reporting
        ind = jnp.argmin(dist_sq, axis=2, keepdims=True) # (S, M, 1)
        min_lamb = jnp.take_along_axis(self.lamda[None], ind, axis=2)[:, :, 0] # (S, M)
        int_hkl = jnp.take_along_axis(jnp.round(hkl).astype(jnp.int32), ind[:, None], axis=3)[..., 0]

        return self.max_score - total_score, total_score, int_hkl.transpose((0, 2, 1)), min_lamb

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

#        error, num, hkl, lamda = self.indexer_jax(UB)
        error, _, _, _ = self.indexer_soft_jax(UB, softness=self.softness)

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

        # create a gemmi spacegroup object

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
            self.two_theta = f["peaks/two_theta"][()]
            self.az_phi = f["peaks/azimuthal"][()]
            self.intensity = f["peaks/intensity"][()]
            self.sigma_intensity = f["peaks/sigma"][()]
            self.centering = f["sample/centering"][()].decode("utf-8")

    def get_consistent_U_for_symmetry(self, U_mat, B_mat):
        """
        Return the proper rotations for this spacegroup and pick
        a consistent U matrix among all symmetry-related possibilities.

        Parameters:
        ----------
        U_mat: array, (3, 3)
            rotation matrix

        B_mat: array, (3, 3)
            B (instrument) matrix

        Returns:
            (U_unique, T)

            where U_unique is a symmetry-equivalent (3, 3) matrix, and T is used to transform
            the input U to the unique one
        """

        uc = gemmi.UnitCell(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )

        # extract the proper rotations from the point group
        gops = gemmi.find_lattice_symmetry(uc, self.centering, max_obliq=3.0)
        transforms = [ np.array(g.rot) // 24 for g in gops.sym_ops ]

        # select a rotation that maximes the trace of UB
        cost, T = -np.inf, np.eye(3)
        for M in transforms:
            UBp = U_mat @ B_mat @ np.linalg.inv(M)
            trace = np.trace(UBp)
            if trace > cost:
                cost = trace
                T = M.copy()

        # the new U matrix
        U_prime = U_mat @ B_mat @ np.linalg.inv(T) @ np.linalg.inv(B_mat)

        return U_prime, T

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

        # self.R.shape == (M, 3, 3)
        return np.einsum("mji,jm->im", self.R, kf_ki_dir)
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

    def UB_matrix(self, U, B):
        """
        Calculate :math:`UB`-matrix.
        ... (rest of method) ...
        """

        return U @ B

    def minimize_evosax(
        self,
        strategy_name: str, 
        population_size: int = 1000, 
        num_generations: int = 100, 
        n_runs: int = 1, 
        seed: int = 0,
        softness: float = 1e-3,
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
        softness : float
            Softness parameter (the smaller, the more stringent the peak finding criteria)

        Returns
        -------
        (num, hkl, lamda)
            Tuple containing results from index_de().
        """

        kf_ki_dir = self.uncertainty_line_segements()

        # 1. Use Signal-to-Noise (I / sigma)
        # Add a small epsilon to sigma to prevent division by zero
        weights = self.intensity / (self.sigma_intensity + 1e-6)

        # 2. Normalize weights so mean is 1.0
        # This keeps the loss function scale intuitive.
        weights = weights / np.mean(weights)

        # Optional: Clip extremely high weights to prevent one single peak
        # from dominating the entire solution (e.g., max weight = 10x average)
        weights = np.clip(weights, 0, 10.0)

        # 1. Instantiate the JAX-compatible objective function
        objective = VectorizedObjectiveJAX(
            self.reciprocal_lattice_B(),
            self.centering,
            kf_ki_dir,
            np.array(self.wavelength), # Pass wavelength bounds
            self._angle_cdf,           # Pass interpolation data
            self._angle_t,             # Pass interpolation data
            weights=weights,
            softness=softness,
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
            strategy_type = 'population_based'
            print("Using Differential Evolution (DE) strategy.")
        elif strategy_name.lower() == "pso":
            strategy = PSO(
                solution=sample_solution,
                population_size=population_size
            )
            strategy_type = 'population_based'
            print("Using Particle Swarm Optimization (PSO) strategy.")
        elif strategy_name.lower() == "cma_es":
            strategy = CMA_ES(
                solution=sample_solution,
                population_size=population_size
            )
            strategy_type = 'distribution_based'
            print("Using Covariance matrix adaptation evolution strategy (CMA-ES).")
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

            # Create initial population and fitness.
            rng, rng_pop, rng_init = jax.random.split(rng, 3)
            # Initialize state using .init()
            if strategy_type == 'population_based':
                # Population is (popsize, 3) random numbers [0, 1]
                population_init = jax.random.uniform(rng_pop, (population_size, 3))

                # Evaluate the initial population
                fitness_init = objective(population_init)

                state = strategy.init(rng_init, population_init, fitness_init, params)
            elif strategy_type == 'distribution_based':
                # solution is (3, ) random numbers [0, 1]
                solution_init = jax.random.uniform(rng_pop, (3, ))

                state = strategy.init(rng_init, solution_init, params)
            else:
                raise ValueError

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

        U = objective.orientation_U_jax(self.x[None])[0]
        B = self.reciprocal_lattice_B()
        U_new, _ = self.get_consistent_U_for_symmetry(U, B)
        _, score, hkl, lamb = objective.indexer_soft_jax((U_new @ B)[None], softness=softness)

        return score[0], hkl[0], lamb[0], U_new
