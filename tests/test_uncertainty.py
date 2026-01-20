import numpy as np
import matplotlib.pyplot as plt
from subhkl.convex_hull.peak_integrator import PeakIntegrator
from subhkl.convex_hull.region_grower import RegionGrower

np.seterr(over="ignore")
np.seterr(invalid="ignore")

image_width = 30
image_height = 30
rate_x = 15
rate_y = 15
rate_sigma_x = 2
rate_sigma_y = 2
rate_rho = 0.6
rate_mat = np.array([
    [rate_sigma_x**2, rate_sigma_x*rate_sigma_y*rate_rho],
    [rate_sigma_x*rate_sigma_y*rate_rho, rate_sigma_y**2],
])
rate_a = 45 
rate_b = 30
true_intensity = 2 * np.pi * rate_sigma_x * rate_sigma_y * rate_a * (1 - rate_rho**2)**.5


def calc_rate(x, y):
    xy = np.stack([x - rate_x, y - rate_y], axis=-1)
    u = np.einsum("...i,ij,...j->...", xy, np.linalg.inv(rate_mat), xy)
    return rate_b + rate_a * np.exp(-u / 2)


def simulate(n, rate, rng):
    return rng.poisson(rate[None], size=(n, *rate.shape))


xp, yp = np.stack(
    np.meshgrid(np.arange(image_width), np.arange(image_height), indexing="ij"),
    axis=0
)

rate = calc_rate(xp, yp)
v_min, v_max = 0, 1.5*rate.max()
plt.imshow(rate, vmin=v_min, vmax=v_max)
plt.title("Rate function")
plt.show()

counts = simulate(1000, rate, np.random.default_rng())

plt.imshow(counts[0], vmin=v_min, vmax=v_max)
plt.title("A simulation")
plt.show()

peak_integrator = PeakIntegrator(
    RegionGrower(
        distance_threshold=2.0,
        min_intensity=35,
        max_size=8
    ),
    box_size=3,
    smoothing_window_size=3,
    min_peak_pixels=10,
    min_peak_snr=1.0
)

result, hulls = peak_integrator.integrate_peaks(
    0, counts[0], np.array([[rate_x, rate_y]]), return_hulls=True, return_headers=True
)
plt.imshow(counts[0], vmin=v_min, vmax=v_max)
for _, hull, _, _ in hulls:
    for simplex in hull.simplices:
        plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], c='r')
plt.title("Convex hull")
plt.show()
print("True intensity", true_intensity)
print("Integration result")
print(result)

peak_integrator.integrate_peaks(
    0, counts[0], np.array([[rate_x, rate_y]]),
)

integrated_i = []
integrated_sigma = []
for intensity in counts:
    result = peak_integrator.integrate_peaks(
        0, intensity, np.array([[rate_x, rate_y]])
    )
    integrated_i.append(result[0][3])
    integrated_sigma.append(result[0][5])

plt.hist(integrated_i, bins=30)
plt.title("Distribution of integrated intensities")
plt.show()

plt.hist(integrated_sigma, bins=30)
plt.title("Distribution of integrated sigmas")
plt.show()

print("Mean integrated intensity", np.mean(integrated_i))
print("Std integrated intensity", np.std(integrated_i))
print("Mean estimated std", np.mean(integrated_sigma))

new_integrated_i = []
new_integrated_sigma = []
for intensity in counts:
    result = peak_integrator.integrate_peaks(
        0, intensity, np.array([[rate_x, rate_y]]),
        integration_method="gaussian_fit"
    )
    if result[0][3] is not None:
        new_integrated_i.append(result[0][3])
        new_integrated_sigma.append(result[0][5])

plt.hist(new_integrated_i, bins=30)
plt.title("Distribution of integrated intensities (new)")
plt.show()

plt.hist(new_integrated_sigma, bins=30)
plt.title("Distribution of integrated sigmas (new)")
plt.show()

print("Mean integrated intensity (new)", np.mean(new_integrated_i))
print("Std integrated intensity (new)", np.std(new_integrated_i))
print("Mean estimated std (new)", np.mean(new_integrated_sigma))

