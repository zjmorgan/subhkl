import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import itertools

_open_kernel3 = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

_open_kernel5 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
], dtype=np.uint8)

_open_kernel7 = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0]
], dtype=np.uint8)

class ThresholdingPeakFinder:
    def __init__(
            self,
            noise_cutoff_quantile: float = 0.8,
            min_peak_dist_pixels: float = 8.0,
            mask_file: str | None = None,
            mask_rel_erosion_radius: float = 0.05,
            rel_blur_kernel_size: float = 0.08,
            open_kernel_size_pixels: int = 3,
            adaptive_normalization_rel_kernel_size: float | None = None,
            show_steps: bool = False,
            show_scale: str = "linear"
    ):
        self.noise_cutoff_quantile = noise_cutoff_quantile
        self.min_peak_dist_pixels = min_peak_dist_pixels
        self.rel_blur_kernel_size = rel_blur_kernel_size
        assert open_kernel_size_pixels in (3, 5, 7), \
            "Invalid open kernel size. Only 3, 5 and 7 are available"
        self.open_kernel_size = open_kernel_size_pixels
        self.adaptive_normalization_rel_kernel_size = adaptive_normalization_rel_kernel_size

        if mask_file is None:
            self.mask = None
        else:
            mask = np.array(Image.open(mask_file))
            radius = max(1, int(min(mask.shape) * mask_rel_erosion_radius))
            kernel = np.ones((radius, radius), dtype=np.uint8)
            self.mask = cv2.erode(mask, kernel).astype(bool)

        self.show_steps = show_steps
        self.show_scale = show_scale

    @staticmethod
    def circularity(contour):
        area = cv2.contourArea(contour)
        if area  == 0:
            return float('inf')
        else:
            diameter = np.max(np.linalg.norm(contour[None, :, 0, :] - contour, axis=-1))
            return diameter * cv2.arcLength(contour, True) / area

    def split_contour(self, points):
        # points (N, 1, 2)
        n = len(points)
        split_indices = np.array(list(itertools.combinations(range(n), r=2)))
        costs = []

        for split_pair in split_indices:
            if split_pair[1] - split_pair[0] < 3:
                costs.append(float('inf'))
                continue

            if split_pair[0] + n - split_pair[1] < 3:
                costs.append(float('inf'))
                continue

            points1 = points[split_pair[0]: split_pair[1]]
            points2 = np.concatenate([points[split_pair[1]:], points[:split_pair[0]]], axis=0)
            m1 = self.circularity(points1)
            m2 = self.circularity(points2)
            costs.append(m1 + m2)

        best_split = np.argmin(costs)
        if costs[best_split] / 2 < self.circularity(points):
            split_pair = split_indices[best_split]
            points1 = points[split_pair[0]: split_pair[1]]
            points2 = np.concatenate([points[split_pair[1]:], points[:split_pair[0]]], axis=0)

            center1 = np.mean(points1, axis=0)
            center2 = np.mean(points2, axis=0)

            if np.linalg.norm(center1 - center2) < self.min_peak_dist_pixels:
                if self.circularity(points1) < self.circularity(points2):
                    return points1,
                else:
                    return points2,
            else:
                return points1, points2
        else:
            return points,

    def find_peaks(self, im):
        im = im.astype(float)

        if self.mask is None:
            mask = np.ones(im.shape, dtype=bool)
        else:
            mask = self.mask
            assert mask.shape == im.shape, "Invalid size of mask for given image"

        if self.show_steps:
            plt.imshow(mask)
            plt.title("Mask")
            plt.show()
            plt.imshow(
                im * mask.astype(float),
                cmap="binary"
            )
            plt.title("Masked image")
            plt.show()

        sigma = self.rel_blur_kernel_size * min(im.shape)
        k_size = 2 * (max(3, int(sigma * 2.5)) // 2) + 1
        im_blur = cv2.GaussianBlur(im, (k_size, k_size), sigma)

        if self.adaptive_normalization_rel_kernel_size is not None:
            sigma = self.adaptive_normalization_rel_kernel_size * min(im.shape)
            k_size = 2 * (max(3, int(sigma * 2.5)) // 2) + 1
            norm = np.maximum(1.0, cv2.GaussianBlur(im, (k_size, k_size), sigma))
            im_blur /= norm

        if self.show_steps:
            plt.imshow(im_blur * mask.astype(float), cmap="binary")
            plt.title("Blurred")
            plt.show()

        if self.adaptive_normalization_rel_kernel_size is not None:
            noise_est = np.quantile(im_blur[mask], self.noise_cutoff_quantile)
        else:
            noise_est = np.quantile(im[mask], self.noise_cutoff_quantile)
        im_noise_sub = np.maximum(0.0, im_blur - noise_est)

        if self.show_steps:
            plt.imshow(im_noise_sub * mask.astype(float), cmap="binary", vmax=1.0)
            plt.title("Noise-subtracted")
            plt.show()

        open_kernel = _open_kernel3 if self.open_kernel_size == 3 else _open_kernel5
        im_opened = cv2.morphologyEx(((im_noise_sub > 0) * mask).astype(np.uint8), cv2.MORPH_OPEN, open_kernel)

        if self.show_steps:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(im_noise_sub * mask.astype(float), cmap="binary", vmax=1.0)
            axes[0].set_title("Noise-subtracted")
            axes[1].imshow(im_opened)
            axes[1].set_title("Opened")
            plt.show()

        contours, _ = cv2.findContours(im_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.show_steps:
            bg_im = (255 - (im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)
            bg_color = cv2.cvtColor(bg_im, cv2.COLOR_GRAY2BGR)

            im_contours = cv2.drawContours(bg_color, contours, -1, (0, 255, 0))
            plt.imshow(im_contours)
            if len(contours) > 0:
                contour_centers = np.stack([np.mean(c[:, 0, :], axis=0) for c in contours])
            else:
                contour_centers = np.empty((0, 2))
            plt.scatter(contour_centers[:, 0], contour_centers[:, 1], edgecolors='red', facecolors='none')
            plt.title("Initial contours")
            plt.show()

        did_split = True
        split_counter = 1
        while did_split:
            did_split = False
            keep_contours = []
            split_contours = []

            for c in contours:
                area = cv2.contourArea(c)
                if area <= 3 or area > .1 * im.shape[0] * im.shape[1]:
                    continue

                if self.circularity(c) > 7.0 and area > 30.0:
                    new_contours = self.split_contour(c)
                    split_contours.extend(new_contours)
                    did_split = (len(new_contours) > 1)
                else:
                    keep_contours.append(c)

            contours = keep_contours + split_contours

            if self.show_steps:
                im_contours = cv2.drawContours(bg_color, contours, -1, (0, 255, 0))
                plt.imshow(im_contours)
                if len(contours) > 0:
                    contour_centers = np.stack([np.mean(c[:, 0, :], axis=0) for c in contours])
                else:
                    contour_centers = np.empty((0, 2))
                plt.scatter(contour_centers[:, 0], contour_centers[:, 1], edgecolors='red', facecolors='none')
                plt.title(f"Split step {split_counter}")
                plt.show()

            split_counter += 1

        if self.show_steps:
            im_contours = cv2.drawContours(bg_color, contours, -1, (0, 255, 0))
            plt.imshow(im_contours)
            if len(contours) > 0:
                contour_centers = np.stack([np.mean(c[:, 0, :], axis=0) for c in contours])
            else:
                contour_centers = np.empty((0, 2))
            plt.scatter(contour_centers[:, 0], contour_centers[:, 1], edgecolors='red', facecolors='none')
            plt.title("Final contours")
            plt.show()

        if len(contours) > 0:
            contour_centers = np.stack([np.mean(c[:, 0, :], axis=0) for c in contours])
        else:
            contour_centers = np.empty((0, 2))

        if self.show_steps:
            plt.imshow(im, cmap="binary")
            plt.scatter(contour_centers[:, 0], contour_centers[:, 1], edgecolors='red', facecolors='none')
            plt.title("Peaks")
            plt.show()

        if len(contours) > 0:
            return np.stack([contour_centers[:, 1], contour_centers[:, 0]], axis=1)
        else:
            return np.empty((0, 2))
