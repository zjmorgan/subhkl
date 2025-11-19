"""
Region growing (agglomerative clustering) algorithm
"""

from math import ceil

import numpy as np


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
        threshold_sq = self.distance_threshold**2

        neighbors_rel = []
        for row in range(-threshold_int, threshold_int + 1):
            for col in range(-threshold_int, threshold_int + 1):
                if row**2 + col**2 <= threshold_sq:
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
                    dist_center = (
                        (neighbor_row - initial[0]) ** 2
                        + (neighbor_col - initial[1]) ** 2
                    ) ** 0.5
                    if dist_center < self.max_size:
                        grow_queue.append(neighbor_point)
                        cluster.add(neighbor_point)

        return np.array(list(cluster), dtype=int)
