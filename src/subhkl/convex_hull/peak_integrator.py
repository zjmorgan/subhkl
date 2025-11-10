import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.stats import zscore
from scipy.optimize import minimize

from subhkl.convex_hull.offset_mask import OffsetMask
from subhkl.convex_hull.region_grower import RegionGrower


class PeakIntegrator:
    @staticmethod
    def build_from_dictionary(integration_params):
        region_growth_params = {
            "distance_threshold": integration_params.pop("region_growth_distance_threshold"),
            "min_intensity": integration_params.pop("region_growth_minimum_intensity"),
            "max_size": integration_params.pop("region_growth_maximum_pixel_radius")
        }
        other_params = {
            "box_size": integration_params["peak_center_box_size"],
            "smoothing_window_size": integration_params["peak_smoothing_window_size"],
            "min_peak_pixels": integration_params["peak_minimum_pixels"],
            "min_peak_snr": integration_params["peak_minimum_signal_to_noise"],
            "outlier_threshold": integration_params["peak_pixel_outlier_threshold"]
        }
        integrator = PeakIntegrator(
            RegionGrower(**region_growth_params),
            **other_params
        )

        return integrator

    def __init__(
            self,
            region_grower,
            box_size: int = 5,
            smoothing_window_size: int = 5,
            min_peak_pixels: int = 3,
            min_peak_snr: float = 1.0,
            outlier_threshold: float = 2.0,
    ):
        """
        Integrates all peaks for an entire bank

        Parameters
        ----------
        region_grower:
            Implementation of the region growing algorithm.
        box_size:
            Size of box around estimated peak center in which to search
            for adjusted peak center. Must be odd.
        smoothing_window_size:
            Size of smoothing window for smoothing
            convolution used to find adjusted peak centers.
        min_peak_pixels:
            Minimum number of pixels in grown region needed to
            count it as a peak detection
        min_peak_snr:
            Minimum peak signal-to-noise ratio needed to count it as a peak
        outlier_threshold:
            Threshold (in # of standard deviations) for culling
            outliers in the cluster to obtain the core cluster
        """
        self.region_grower = region_grower

        assert box_size % 2 == 1, "box_size must be odd"
        self.box_size = box_size
        self.smoothing_window_size = smoothing_window_size
        self.min_peak_pixels = min_peak_pixels
        self.min_peak_snr = min_peak_snr
        self.outlier_threshold = outlier_threshold

    def integrate_peaks(
            self,
            bank_id,
            intensity,
            peak_centers,
            return_hulls=False,
            return_headers=False
    ):
        """
        Integrates all peaks for the bank

        Parameters
        ----------
        bank_id:
            integer id of the bank being processed
        intensity:
            (H, W)-shaped array of intensities measured by the bank
        peak_centers:
            (N, 2)-shaped array of coordinates of estimated peak centers to
            compute intensity statistics for. First coordinate is y (row),
            second coordinate is x (column).
        return_hulls:
            Whether to return convex hulls of peak regions for visualization
        return_headers:
            Whether to return column headers for the intensity data

        Return
        ------
        outputs:
            Array of the peak statistics. Optionally also the peak region
            convex hulls for visualization.
        """

        # Get masks and hulls
        is_peak, peak_masks, bg_masks, peak_hulls = self._find_peak_regions(
            intensity, peak_centers
        )

        if return_headers:
            output_data = [[
                'bank_id', 'peak_idx', 'bg_den', 'peak_int', 'bg_int', 'sigma'
            ]]
        else:
            output_data = []

        # Use masks to compute intensity statistics
        for i_peak in range(len(peak_centers)):
            if is_peak[i_peak] and len(bg_masks[i_peak].nonzero()) > 0:
#                stats = self._calculate_statistics(
#                    intensity,
#                    peak_masks[i_peak],
#                    bg_masks[i_peak]
#                )
                stats = self._fit_gaussian_mle(
                    intensity,
                    peak_masks[i_peak],
                    bg_masks[i_peak]
                )
                bg_density, peak_intensity, peak_bg_intensity, sigma = map(float, stats)

                # Discard peak if SNR is too low
                snr = peak_intensity / sigma
                if snr < self.min_peak_snr:
                    bg_density, peak_intensity, peak_bg_intensity, sigma = None, None, None, None
                    is_peak[i_peak] = False
                    peak_masks[i_peak] = None
                    bg_masks[i_peak] = None
                    peak_hulls[i_peak] = (None,) * len(peak_hulls[i_peak])
            else:
                bg_density, peak_intensity, peak_bg_intensity, sigma = None, None, None, None

            output_data.append([
                bank_id, i_peak, bg_density, peak_intensity, peak_bg_intensity, sigma
            ])

        if return_hulls:
            return output_data, peak_hulls
        else:
            return output_data

    @staticmethod
    def visualize(bank_id, intensity, peak_hulls):
        """
        Create a Matplotlib Figure to visualize the given bank and its detected
        peak regions. Saves visualization to Bank_{bank_id}.png.

        Parameters
        ----------
        bank_id:
            The id of the bank to visualize
        intensity:
            (H, W)-shaped array of intensity values
        peak_hulls:
            list of hull tuples as returned by integrate_peaks
        """
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.imshow(
            intensity.T,
            cmap='gray_r',
            origin='lower',
            extent=(0, 255, 0, 255),
            vmin=0, vmax=10
        )
        axes.set_title("Bank" + str(bank_id))
        axes.axis('off')

        for core_hull, peak_hull, inner_hull, outer_hull in peak_hulls:
            if core_hull is None:
                continue

            for simplex in core_hull.simplices:
                axes.plot(
                    core_hull.points[simplex, 0], core_hull.points[simplex, 1],
                    'r-', label='Peak Hull', linewidth=0.5
                )
            axes.set_xlim([0, 255])
            axes.set_ylim([0, 255])

            for simplex in peak_hull.simplices:
                axes.plot(
                    peak_hull.points[simplex, 0], peak_hull.points[simplex, 1],
                    'y-', label='Inner Hull', linewidth=0.5
                )

            for simplex in inner_hull.simplices:
                axes.plot(
                    inner_hull.points[simplex, 0], inner_hull.points[simplex, 1],
                    'b-', label='Inner Hull', linewidth=0.5
                )

            for simplex in outer_hull.simplices:
                axes.plot(
                    outer_hull.points[simplex, 0], outer_hull.points[simplex, 1],
                    'g-', label='Inner Hull', linewidth=0.5
                )

        plt.tight_layout()
        plt.savefig('Bank_' + str(bank_id) + '.png')
        plt.close()

    def _find_peak_regions(self, intensity, peak_centers):
        """
        Finds peak regions based on estimated peak centers and an intensity map

        Parameters
        ----------
        intensity:
            (H, W)-shaped array of (integer) intensity values
        peak_centers:
            (n_peaks, 2)-shaped array of (integer) [row, col]
            coordinates of estimated peak centers

        Return
        ------
        outputs:
            4-tuple of the following:

            is_peak:
                (n_peaks,)-shaped array of booleans indicating whether the
                corresponding peak in peak_centers matched a peak in the intensity
                array
            peak_masks:
                list of n_peaks (H, W)-shaped arrays of booleans, which
                indicate pixels that belong to the corresponding peak from
                peak_centers
            bg_masks:
                list n_peaks (H, W)-shaped arrays of booleans, which indicate
                pixels that are nearby the corresponding peak from peak_centers
                but are definitely background (i.e., not contained in the inner
                region of *any* of the detected peaks)
            peak_hulls:
                list of n_peaks 4-tuples (core_hull, peak_hull, inner_hull, outer_hull)
                containing the hulls for each peak (mainly for visualization)
        """
        # Store some basic descriptive information about inputs
        im_shape = intensity.shape
        n_peaks = len(peak_centers)

        # Initialize outputs
        is_peak = np.zeros(n_peaks, dtype=bool)
        peak_masks = []
        inner_masks = []
        bg_masks = []
        peak_hulls = []

        # Smooth the intensity map for finding better peak centers
        smoothed_intensity = self._smooth(intensity)

        # Find regions and generate preliminary masks
        for peak_idx in range(n_peaks):
            estimated_center = tuple(map(int, peak_centers[peak_idx]))

            # Move center to local maximum *in the smoothed image*
            try:
                adjusted_center = self._local_max(smoothed_intensity, estimated_center)
            except ValueError:
                adjusted_center = None

            # Make sure center starts from a non-zero point *in the original
            # image* because we will grow the region based on the original
            # intensity values
            try:
                if adjusted_center is not None:
                    adjusted_center = self._find_nearest_nonzero_point(adjusted_center, intensity)

                    # Grow region
                    cluster_points = self.region_grower.get_region(intensity, adjusted_center)
                else:
                    cluster_points = np.zeros(0)
            except ValueError:
                cluster_points = np.zeros(0)

            # Check if the region grew enough to be considered a peak
            if cluster_points.shape[0] < self.min_peak_pixels:
                peak_masks.append(None)
                inner_masks.append(None)
                bg_masks.append(None)
                peak_hulls.append([None]*4)
                continue
            else:
                is_peak[peak_idx] = True

            # Get core points of peak by removing outliers
            core_points = self._remove_outliers(
                cluster_points,
                threshold=self.outlier_threshold
            )

            # Build masks and hulls
            masks, hulls = self._make_peak_hulls_and_masks(core_points, im_shape)
            peak_hulls.append(hulls)
            peak_mask, inner_mask, bg_mask = masks
            peak_masks.append(peak_mask)
            inner_masks.append(inner_mask)
            bg_masks.append(bg_mask)

        # Now that we have built the preliminary masks we can remove all the
        # inner region pixels from each of the background masks to avoid
        # counting any peak pixels as part of the background noise estimates
        any_inner_mask = np.zeros_like(intensity, dtype=bool)
        for inner_mask in inner_masks:
            if inner_mask is not None:
                indices = inner_mask.indices()
                any_inner_mask[indices[0], indices[1]] |= inner_mask.mask

        # points not in *any* inner region (make this an OffsetMask so we can
        # use &= below)
        not_any_inner_mask = OffsetMask(~any_inner_mask, (0, 0))

        for bg_mask in bg_masks:
            if bg_mask is not None:
                bg_mask &= not_any_inner_mask  # keep points not in any inner region

        return is_peak, peak_masks, bg_masks, peak_hulls

    @staticmethod
    def _fit_gaussian_mle(intensity, peak_mask, bg_mask):
        """
        Calculates peak intensity statistics by fitting a 2D Gaussian + constant
        background using Maximum Likelihood Estimation (assuming Poisson noise).

        Parameters
        ----------
        intensity:
            (H, W) array of intensity
        peak_mask:
            OffsetMask indicating which pixels belong to the peak
        bg_mask:
            OffsetMask indicating which pixels are background near the peak

        Return
        ------
        outputs:
            4-tuple of the following:

            bg_level:
                Fitted background level
            peak_integral:
                Total intensity of the Gaussian (Amplitude * 2 * pi * sx * sy)
            peak_bg_integral:
                Fitted background level * number of pixels in peak mask
            peak_integral_error:
                Estimated error (sigma) of the peak integral
        """

        # --- 1. Define Model and Objective Function ---

        def gaussian_2d_model(y, x, amplitude, y0, x0, sigma_y, sigma_x, background):
            """2D Gaussian model on a constant background."""
            y_term = ((y - y0) ** 2) / (2 * sigma_y ** 2)
            x_term = ((x - x0) ** 2) / (2 * sigma_x ** 2)
            model = background + amplitude * np.exp(-y_term - x_term)
            return model

        def negative_log_likelihood(params, y_coords, x_coords, data):
            """Poisson negative log-likelihood."""
            model = gaussian_2d_model(y_coords, x_coords, *params)

            # Add a small epsilon to model to avoid log(0)
            model[model <= 0] = 1e-9

            # NLL = sum(model - data * log(model))
            nll = np.sum(model - data * np.log(model))
            return nll

        # --- 2. Get Pixel Data ---
        peak_indices = peak_mask.nonzero()
        bg_indices = bg_mask.nonzero()

        if len(peak_indices[0]) == 0:  # No peak pixels
            return None, None, None, None

        y_peak, x_peak = peak_indices[0], peak_indices[1]
        I_peak = intensity[y_peak, x_peak]

        peak_vol = len(y_peak)

        # --- 3. Set Initial Guesses and Bounds ---

        # Background guess: mean of background pixels, ensure non-negative
        if len(bg_indices[0]) > 0:
            bg_est = np.mean(intensity[bg_indices[0], bg_indices[1]])
        else:
            bg_est = np.min(I_peak) # Fallback if bg_mask is empty
        bg_est = max(bg_est, 1e-6) # Ensure background is positive

        # Center guess: centroid of the peak mask
        y0_est = np.mean(y_peak)
        x0_est = np.mean(x_peak)

        # Amplitude guess: max peak intensity minus background
        amp_est = np.max(I_peak) - bg_est
        amp_est = max(amp_est, 1e-6) # Ensure amplitude is positive

        # Initial parameters
        p0 = [amp_est, y0_est, x0_est, 1.0, 1.0, bg_est]

        # Parameter bounds: (amp, y0, x0, sy, sx, bg)
        # Force amplitude, sigmas, and background to be positive
        bounds = [
            (1e-9, None),     # amplitude
            (np.min(y_peak), np.max(y_peak)), # y0
            (np.min(x_peak), np.max(x_peak)), # x0
            (0.1, 10.0),     # sigma_y (setting a reasonable range)
            (0.1, 10.0),     # sigma_x (setting a reasonable range)
            (1e-9, None)      # background
        ]

        # --- 4. Run Minimization ---
        try:
            result = minimize(
                negative_log_likelihood,
                p0,
                args=(y_peak, x_peak, I_peak),
                method='L-BFGS-B',
                bounds=bounds
            )

            if not result.success:
                return None, None, None, None

            # --- 5. Extract Results ---
            amplitude, y0, x0, sigma_y, sigma_x, bg_level = result.x

            # Calculate the total integral of the Gaussian
            # Integral = A * 2 * pi * sx * sy
            peak_integral = 2 * np.pi * amplitude * sigma_y * sigma_x

            # Background contribution under the peak mask
            peak_bg_integral = bg_level * peak_vol

            # --- 6. Estimate Error ---
            # Get covariance matrix (inverse of the Hessian)
            # This provides errors on the fitted parameters
            try:
                inv_hessian = result.hess_inv.todense()
                param_errors = np.sqrt(np.diag(inv_hessian))

                # Propagate error for the integral: I = 2*pi * A * sy * sx
                # (dI/dA)^2 * err_A^2 + (dI/dsy)^2 * err_sy^2 + (dI/dsx)^2 * err_sx^2

                err_A = param_errors[0]
                err_sy = param_errors[3]
                err_sx = param_errors[4]

                term_A = (2 * np.pi * sigma_y * sigma_x * err_A) ** 2
                term_sy = (2 * np.pi * amplitude * sigma_x * err_sy) ** 2
                term_sx = (2 * np.pi * amplitude * sigma_y * err_sx) ** 2

                peak_integral_error = np.sqrt(term_A + term_sy + term_sx)

            except Exception:
                # Fallback if Hessian inversion fails
                peak_integral_error = np.sqrt(peak_integral + peak_bg_integral)


            return bg_level, peak_integral, peak_bg_integral, peak_integral_error

        except (ValueError, np.linalg.LinAlgError):
            # Handle optimization or numerical errors
            return None, None, None, None

    @staticmethod
    def _calculate_statistics(intensity, peak_mask, bg_mask):
        """
        Calculates peak intensity statistics from a peak and background mask

        Parameters
        ----------
        intensity:
            (H, W) array of intensity
        peak_mask:
            OffsetMask indicating which pixels belong to the peak
        bg_mask:
            OffsetMask indicating which pixels are background near the peak

        Return
        ------
        outputs:
            4-tuple of the following:

            bg_density:
                density of intensity in the background
            peak_intensity:
                intensity of the peak alone
            peak_bg_intensity:
                intensity of the peak due to background noise
            sigma:
                estimated error in the peak intensity
        """

        peak_indices = peak_mask.nonzero()
        bg_indices = bg_mask.nonzero()

        peak_vol = len(peak_indices[0])  # number of pixels in peak
        bg_vol = len(bg_indices[0])  # number of pixels in background
        peak2bg = peak_vol / bg_vol

        total_peak_intensity = intensity[peak_indices[0], peak_indices[1]].sum()
        total_bg_intensity = intensity[bg_indices[0], bg_indices[1]].sum()

        peak_bg_intensity = peak2bg * total_bg_intensity
        peak_bg_variance = peak2bg ** 2 * total_bg_intensity

        peak_intensity = total_peak_intensity - peak_bg_intensity
        sigma = (total_peak_intensity + peak_bg_variance) ** .5

        bg_density = total_bg_intensity / bg_vol

        return bg_density, peak_intensity, peak_bg_intensity, sigma

    def _smooth(self, input_tensor):
        """
        Smooths a given image by using uniform averaging kernel. Uses
        zero-padding to obtain an image with the same size.

        Parameters
        ----------
        input_tensor:
            (H, W)-shaped array containing the image values

        Return
        ------
        smoothed:
            (H, W)-shaped array with smoothed values.
        """
        # Convert to float for averaging operations
        input_tensor = input_tensor.astype(float)

        w = self.smoothing_window_size
        smoothed = convolve2d(
            input_tensor,
            np.ones((w, w)) / w**2,
            mode='same', boundary='fill'
        )

        return smoothed

    def _local_max(self, input_tensor, center_point):
        """
        Finds the index of the pixel with the highest value in a box around
        the given center point

        Parameters
        ----------
        input_tensor:
            (H, W) 2D tensor
        center_point:
            [row, col] coordinates of the center of the box

        Return
        ------
        max_point:
            [row_max, col_max] coordinates of the pixel with the highest
            value in th box
        """
        h, w = input_tensor.shape
        r, c = center_point
        half = self.box_size // 2

        # Define the bounds of the window
        r_start = max(0, r - half)
        r_end = min(h, r + half + 1)
        c_start = max(0, c - half)
        c_end = min(w, c + half + 1)

        # Extract the local window
        window = input_tensor[r_start:r_end, c_start:c_end]

        # Find the index of the max value in the window
        max_idx_flat = np.argmax(window)
        max_idx_2d = (
            max_idx_flat // window.shape[1],
            max_idx_flat % window.shape[1]
        )

        # Map local index back to global coordinates
        global_max_idx = (r_start + max_idx_2d[0].item(), c_start + max_idx_2d[1].item())
        return global_max_idx

    @staticmethod
    def _remove_outliers(data, threshold=3.0):
        """
        Remove outliers from the dataset based on z-score.

        Parameters
        ----------
        data:
            Input point data of shape (n_samples, 2).
        threshold:
            Z-score threshold for outlier detection.

        Return
        ------
        filtered_data:
            Data with outliers removed.
        """
        z_scores = np.abs(zscore(data, axis=0))
        return data[(z_scores < threshold).all(axis=1)]

    @staticmethod
    def _find_nearest_nonzero_point(start, intensity) -> tuple[int, int]:
        """
        Searches in a spiral-like pattern around a given point to find the
        nearest point with a nonzero intensity

        Parameters
        ----------
        start: [row, col] of starting point
        intensity: (H, W)-shaped array of intensity values

        Return
        ------
        nearest: [row_near, col_near] coordinates of the nearest point to start
            that has a nonzero intensity
        """
        if np.all(intensity == 0):
            raise ValueError('Invalid intensity map--all intensities are 0!')

        if intensity[start[0], start[1]] != 0:
            return start

        h, w = intensity.shape

        for r in range(max(h, w)):
            row, col = max(0, start[0] - r), start[1]
            for col in range(max(0, start[1] - r), min(w, start[1] + r + 1)):
                if intensity[row, col] != 0:
                    return row, col

            for row in range(max(0, start[0] - r + 1), min(h, start[0] + r)):
                col = max(0, start[1] - r)
                if intensity[row, col] != 0:
                    return row, col

                col = min(w - 1, start[1] + r)
                if intensity[row, col] != 0:
                    return row, col

            row = min(h - 1, start[1] + r)
            for col in range(max(0, start[1] - r), min(w, start[1] + r + 1)):
                if intensity[row, col] != 0:
                    return row, col

        raise ValueError('Invalid intensity map--all intensities are 0!')

    def _make_peak_hulls_and_masks(self, core_points, im_shape):
        """
        Generate peak hulls and masks

        Parameters
        ----------
        core_points:
            (N, 2) array of coordinate vectors for points belonging to
            the core of the peak
        im_shape:
            [H, W] integers giving the shape of the image

        Return
        ------
        outputs:
            2-tuple of two tuples:
            ((peak_mask, inner_mask, bg_mask), (core_hull, peak_hull, inner_hull,
              outer_hull)) giving the masks and hulls for the peak
        """
        # Adjust the core hall to make sure it's not too big or too small
        core_hull = ConvexHull(core_points)

        core_scale = 0.0
        core_hull = self._expand_convex_hull(core_hull, core_scale)

        peak_scale = 0.1
        peak_hull = self._expand_convex_hull(core_hull, peak_scale)

        inner_scale = peak_scale + 0.5
        inner_hull = self._expand_convex_hull(core_hull, inner_scale)

        outer_scale = inner_scale + 1
        outer_hull = self._expand_convex_hull(core_hull, outer_scale)

        # Generate masks
        peak_mask = self._hull_mask(peak_hull, im_shape)
        inner_mask = self._hull_mask(inner_hull, im_shape)
        outer_mask = self._hull_mask(outer_hull, im_shape)

        # Remove inner mask pixels to get background pixels (for now
        # ignoring the possibility of containment in a *different* peak's
        # inner region)
        bg_mask = outer_mask - inner_mask

        return ((peak_mask, inner_mask, bg_mask),
                (core_hull, peak_hull, inner_hull, outer_hull))

    @staticmethod
    def _hull_mask(hull, shape):
        """
        Generate an OffsetMask object with a mask that is True inside
        the given convex hull and False outside

        Parameters
        ----------
        hull:
            ConvexHull object representing the 3D convex hull.
        shape:
            tuple (H, W) giving the shape of the image

        Return
        ------
        mask:
            OffsetMask describing a mask of the convex hull
        """
        hull_vertices = hull.points[hull.vertices]
        min_vert = np.maximum(np.min(hull_vertices, axis=0), 0)
        max_vert = np.max(hull_vertices, axis=0)
        max_vert[0] = min(max_vert[0], shape[0] - 1)
        max_vert[1] = min(max_vert[1], shape[1] - 1)
        w_m = int(max_vert[0] - min_vert[0] + 1)
        h_m = int(max_vert[1] - min_vert[1] + 1)

        # Generate a Cartesian grid over the domain
        x = min_vert[0] + np.arange(w_m)
        y = min_vert[1] + np.arange(h_m)
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")

        # Stack grid points into a (w, h, 2) array
        grid_points = np.stack([grid_x, grid_y], axis=-1)

        # Use Delaunay triangulation to efficiently check if points are inside the convex hull
        delaunay = Delaunay(hull.points[hull.vertices])
        mask = (delaunay.find_simplex(grid_points) >= 0).astype(bool)

        return OffsetMask(mask, min_vert)

    @staticmethod
    def _expand_convex_hull(hull, scale_factor):

        """
        Expand a convex hull along the radial direction with respect to the mean of the points.

        Parameters
        ----------
        hull:
            ConvexHull object of the original points.
        scale_factor:
            Scaling factor for radial expansion (>0 means expansion).

        Returns
        -------
        expanded_hull:
            ConvexHull of the expanded points.
        """

        # Compute the mean of the points in the convex hull
        mean_point = np.mean(hull.points[hull.vertices], axis=0)

        # Expand each vertex along the radial direction
        delta = scale_factor * (hull.points[hull.vertices] - mean_point)
        expanded_vertex = hull.points[hull.vertices] + delta

        # Compute the new convex hull
        expanded_hull = ConvexHull(expanded_vertex)

        return expanded_hull
