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
            blur_kernel_sigma: int = 5,
            open_kernel_size_pixels: int = 3,
            show_steps: bool = False,
            show_scale: str = "linear"
    ):
        self.noise_cutoff_quantile = noise_cutoff_quantile
        self.min_peak_dist_pixels = min_peak_dist_pixels
        self.blur_kernel_sigma = blur_kernel_sigma
        assert open_kernel_size_pixels in (3, 5, 7), \
            "Invalid open kernel size. Only 3, 5 and 7 are available"
        self.open_kernel_size = open_kernel_size_pixels

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
                cmap="binary",
                norm=self.show_scale
            )
            plt.title("Masked image")
            plt.show()
            plt.savefig('masked_image.png')

        # 1.6 as per Marr and Hildreth, "Theory of Edge Detection"
        big_sigma = 1.6 * self.blur_kernel_sigma
        k_size_desired = max(1, int(big_sigma*3))
        k_size = 2 * (k_size_desired//2) + 1
        blur_small = cv2.GaussianBlur(im, (k_size, k_size), self.blur_kernel_sigma)
        blur_big = cv2.GaussianBlur(im, (k_size, k_size), big_sigma)

        # small - big ~ -Laplacian, so peaks occur at *large* values of im_dog
        # (red on the seismic color map)
        im_dog = blur_small - blur_big

        if self.show_steps:
            plt.imshow(im_dog * mask.astype(float), cmap="seismic", norm=self.show_scale)
            plt.title("DoG")
            plt.show()
            plt.savefig('difference_of_gaussians.png')

        noise_est = np.quantile(im_dog[mask], self.noise_cutoff_quantile)
        im_thresh = (im_dog > noise_est)

        if self.show_steps:
            plt.imshow(im_thresh * mask.astype(float), cmap="binary", vmax=1.0)
            plt.title("Threshold")
            plt.show()
            plt.savefig('threshold.png')

        open_kernel = _open_kernel3 if self.open_kernel_size == 3 else _open_kernel5
        im_opened = cv2.morphologyEx(((im_thresh > 0) * mask).astype(np.uint8), cv2.MORPH_OPEN, open_kernel)

        if self.show_steps:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(im_thresh * mask.astype(float), cmap="binary", vmax=1.0)
            axes[0].set_title("Noise-subtracted")
            axes[1].imshow(im_opened)
            axes[1].set_title("Opened")
            plt.show()
            plt.savefig('noise_subtracted.png')

        contours, _ = cv2.findContours(im_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.show_steps:
            if self.show_scale == 'linear':
                bg_im = (255 - (im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)
            else:
                log_im = np.maximum(0, np.log(im + 1e-5))
                bg_im = (255 - (log_im / log_im.max()) * 255).astype(np.uint8)
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
            plt.savefig('initial_contours.png')

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
                plt.savefig(f'split_{split_counter}.png')

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
            plt.savefig('final_contours.png')

        if len(contours) > 0:
            contour_centers = np.stack([np.mean(c[:, 0, :], axis=0) for c in contours])
        else:
            contour_centers = np.empty((0, 2))

        if self.show_steps:
            plt.imshow(im, norm=self.show_scale, cmap="binary")
            plt.scatter(contour_centers[:, 0], contour_centers[:, 1], edgecolors='red', facecolors='none')
            plt.title("Peaks")
            plt.show()
            plt.savefig('peaks.png')

        if len(contours) > 0:
            return np.stack([contour_centers[:, 1], contour_centers[:, 0]], axis=1)
        else:
            return np.empty((0, 2))
