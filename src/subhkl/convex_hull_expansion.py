from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.stats import zscore
from scipy.signal import convolve2d


def remove_outliers(data, threshold=3.0):
    """
    Remove outliers from the dataset based on z-score.

    Parameters
    ----------
    data:
        Input 3D data of shape (n_samples, 3).
    threshold:
        Z-score threshold for outlier detection.

    Return
    ------
    filtered_data:
        Data with outliers removed.
    """
    z_scores = np.abs(zscore(data, axis=0))
    return data[(z_scores < threshold).all(axis=1)]


def find_nearest_nonzero_point(start, intensity) -> tuple[int, int]:
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


class OffsetMask:
    def __init__(self, mask, offset):
        self.mask = mask
        self.offset = np.array(offset, dtype=int)

    def indices(self):
        return (
            slice(self.offset[0], self.offset[0] + self.mask.shape[0]),
            slice(self.offset[1], self.offset[1] + self.mask.shape[1])
        )

    def __iand__(self, other):
        # Probably the fastest implementation here is to create a new mask
        # array anyway, so we just fall back to __and__
        result = self & other
        self.mask = result.mask
        self.offset = result.offset

    def __ior__(self, other):
        result = self | other
        self.mask = result.mask
        self.offset = result.offset

    def __and__(self, other):
        # Result region is intersection of two mask regions
        result_min = np.maximum(self.offset, other.offset)
        result_max = np.minimum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )
        result_shape = result_max - result_min

        if np.any(result_shape < 0):
            return OffsetMask(np.zeros((0, 0), dtype=bool), (0, 0))

        my_i = (
            slice(result_min[0] - self.offset[0], result_max[0] - self.offset[0]),
            slice(result_min[1] - self.offset[1], result_max[1] - self.offset[1])
        )
        other_i = (
            slice(result_min[0] - other.offset[0], result_max[0] - other.offset[0]),
            slice(result_min[1] - other.offset[1], result_max[1] - other.offset[1])
        )
        result_mask = self.mask[my_i[0], my_i[1]] & other.mask[other_i[0], other_i[1]]

        return OffsetMask(result_mask, result_min)

    def __rand__(self, other):
        return self & other

    def __or__(self, other):
        # Result region is union of two mask regions
        result_min = np.minimum(self.offset, other.offset)
        result_max = np.maximum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )
        result_shape = result_max - result_min

        result_mask = np.zeros(result_shape, dtype=bool)
        my_offset_rel = self.offset - result_min
        my_i = (
            slice(my_offset_rel[0], my_offset_rel[0] + self.mask.shape[0]),
            slice(my_offset_rel[1], my_offset_rel[1] + self.mask.shape[1])
        )
        result_mask[my_i[0], my_i[1]] = self.mask

        other_offset_rel = other.offset - result_min
        other_i = (
            slice(other_offset_rel[0], other_offset_rel[0] + other.mask.shape[0]),
            slice(other_offset_rel[1], other_offset_rel[1] + other.mask.shape[1])
        )
        result_mask[other_i[0], other_i[1]] &= other.mask

        return OffsetMask(result_mask, result_min)

    def __ror__(self, other):
        return self | other

    def __invert__(self):
        return OffsetMask(~self.mask, self.offset)

    def __sub__(self, other):
        # self - other = self and not other
        # Find intersection
        int_min = np.maximum(self.offset, other.offset)
        int_max = np.minimum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )

        # No need to do anything if intersection is empty
        if np.any(int_max - int_min) == 0:
            return OffsetMask(self.mask.copy(), self.offset)

        # Update mask in intersection
        my_i = (
            slice(int_min[0] - self.offset[0], int_max[0] - self.offset[0]),
            slice(int_min[1] - self.offset[1], int_max[1] - self.offset[1])
        )
        other_i = (
            slice(int_min[0] - other.offset[0], int_max[0] - other.offset[0]),
            slice(int_min[1] - other.offset[1], int_max[1] - other.offset[1])
        )
        # Copy outside of intersection because ~other is True in my region
        # (set) minus other region
        result_mask = self.mask.copy()

        # and not other in the intersection
        result_mask[my_i[0], my_i[1]] &= ~other.mask[other_i[0], other_i[1]]

        return OffsetMask(result_mask, self.offset)

    def full(self, shape):
        full_result = np.zeros(shape, dtype=bool)
        full_result[
            self.offset[0]: self.offset[0] + self.mask.shape[0],
            self.offset[1]: self.offset[1] + self.mask.shape[1]
        ] = self.mask

        return full_result

    def nonzero(self):
        x, y = np.nonzero(self.mask)
        x += self.offset[0]
        y += self.offset[1]
        return x, y


class RegionGrower:
    def __init__(self, distance_threshold, min_intensity, max_size):
        """
        Parameters
        ----------

        - distance_threshold: threshold for pixels to be considered neighbors

        - min_intensity: minimum amount of neighboring intensity to consider
            growing the cluster from any of the neighbors of the current point

        - max_size: maximum radius of the cluster
        """
        self.distance_threshold = distance_threshold
        self.min_intensity = min_intensity
        self.max_size = max_size
        self.neighbors_rel = self._get_neighbors_rel()

    def _get_neighbors_rel(self):
        threshold_int = ceil(self.distance_threshold)
        threshold_sq = self.distance_threshold ** 2

        neighbors_rel = []
        for row in range(-threshold_int, threshold_int + 1):
            for col in range(-threshold_int, threshold_int + 1):
                if row ** 2 + col ** 2 <= threshold_sq:
                    neighbors_rel.append((row, col))

        return neighbors_rel

    @staticmethod
    def _is_valid(intensity, row, col):
        return 0 <= row < intensity.shape[0] and 0 <= col < intensity.shape[1]

    def get_region(self, intensity, initial):
        """
        Gets the region by growing from the initial point

        Parameters
        ----------

        - intensity: Array of shape (H, W) of intensity values

        - initial: (row, col) coordinates of initial point from which to grow

        Return
        ------

        - cluster: (K, 2) array of coordinates of points in the cluster
        """
        initial = tuple(map(int, initial))
        visited, cluster = set(), {initial}
        grow_queue = [initial]

        while grow_queue:
            point = grow_queue.pop()
            if point in visited:
                continue

            visited.add(point)
            row, col = point

            total_neighbor_intensity = 0

            neighbor_indices = []
            for neighbor_row_rel, neighbor_col_rel in self.neighbors_rel:
                neighbor_row = row + neighbor_row_rel
                neighbor_col = col + neighbor_col_rel

                if not self._is_valid(intensity, neighbor_row, neighbor_col):
                    continue

                neighbor_intensity = int(intensity[neighbor_row, neighbor_col])

                if neighbor_intensity > 0:
                    total_neighbor_intensity += neighbor_intensity
                    neighbor_indices.append((neighbor_row, neighbor_col))

            if total_neighbor_intensity >= self.min_intensity:
                for neighbor_point in neighbor_indices:
                    if neighbor_point in visited:
                        continue

                    if neighbor_point in cluster:
                        continue

                    neighbor_row, neighbor_col = neighbor_point
                    dist_center = ((neighbor_row - initial[0]) ** 2 + (neighbor_col - initial[1]) ** 2) ** .5
                    if dist_center < self.max_size:
                        grow_queue.append(neighbor_point)
                        cluster.add(neighbor_point)

        return np.array(list(cluster), dtype=int)


class PeakIntegrator:
    def __init__(
            self,
            region_grower,
            box_size: int = 5,
            smoothing_window_size: int = 5,
            min_peak_pixels: int = 3,
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
        outlier_threshold:
            Threshold (in # of standard deviations) for culling
            outliers in the cluster to obtain the core cluster
        """
        self.region_grower = region_grower

        assert box_size % 2 == 1, "box_size must be odd"
        self.box_size = box_size
        self.smoothing_window_size = smoothing_window_size
        self.min_peak_pixels = min_peak_pixels
        self.outlier_threshold = outlier_threshold

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
            adjusted_center = self._local_max(smoothed_intensity, estimated_center)

            # Make sure center starts from a non-zero point *in the original
            # image* because we will grow the region based on the original
            # intensity values
            try:
                adjusted_center = find_nearest_nonzero_point(adjusted_center, intensity)

                # Grow region
                cluster_points = self.region_grower.get_region(intensity, adjusted_center)
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
            core_points = remove_outliers(
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
            compute intensity statistics for
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
                stats = self._calculate_statistics(
                    intensity,
                    peak_masks[i_peak],
                    bg_masks[i_peak]
                )
                bg_density, peak_intensity, peak_bg_intensity, sigma = stats
            else:
                bg_density, peak_intensity, peak_bg_intensity, sigma = 0, 0, 0, 0

            output_data.append([
                bank_id, i_peak, float(bg_density), float(peak_intensity),
                float(peak_bg_intensity), float(sigma)
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
