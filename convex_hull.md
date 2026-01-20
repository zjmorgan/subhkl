# Convex Hull Peak Integration CLI

Convex hull peak integration is used in two ways in `subhkl`:

1. Preliminary integration is performed during the `finder` subcommand to filter false positive peak detections from the peak finder.
2. Actual peak integration is performed during the `integrator` subcommand.

Both subcommands use the same CLI parameters, documented below in order of likelihood that you will need to modify them. See [Algorithm](#algorithm) below to understand better how the parameters will affect the algorithm.

- `--region-growth-minimum-intensity`: number, units events/pixel
    - Minimum average intensity (# of events) of neighboring pixels required for a pixel to be included in a peak
    - Set this slightly above the average noise level of each image
    - $\downarrow$ for more and bigger peaks
    - $\uparrow$ to remove false positives and decrease peak size
    - This is the **most important** parameter
- `--peak-minimum-pixels`: number, units pixels
    - Peaks are discarded if they don't have at least this many pixels
    - Set this a bit smaller than the smallest peak area
    - $\uparrow$ to filter out small false-positive detections caused by noise
    - $\downarrow$ to avoid false negatives if small peaks are present
- `--region-growth-maximum-pixel-radius`: number, units pixels
    - Maximum radius in pixels within which pixels can be added to the peak (measured from candidate/predicted peak location)
    - Set this slightly larger than the largest peak size
    - Use this to cap the size of peak regions if you need to set `region-growth-minimum-intensity` relatively low to avoid false negatives
- `--peak-minimum-signal-to-noise`: number, unitless
    - Peaks are discarded if the signal-to-noise ratio of the integrated intensity is below this threshold
    - Setting depends on quality of data
    - $\uparrow$ to filter out false positives not filtered by `peak-minimum-pixels`
    - $\downarrow$ to avoid false negatives in noisy data
- `--integration-method`: string enum, either `"free_fit"` or `"gaussian_fit"`
    - **Only available for `integrator`**, this controls what method is used to calculate final intensities and uncertainties after the convex hull fitting is complete
    - `"free_fit"` assumes a Poisson process with an arbitrary rate function and estimates the intensity and uncertainty by simply summing intensities in the peak and outer annular regions
    - `"gaussian_fit"` assumes a Poisson process with Gaussian-plus-offset--shaped underlying rate function, resulting in higher uncertainties (and higher variance) due to introduced model uncertainty. May also fail to fit peaks, causing some peaks to be dropped.
- `--peak-center-box-size`: odd positive integer, units pixels
    - Size of box in which to search for (smoothed) local max pixel from which to start peak clustering
    - Set this around the size of a typical peak
    - Use this to snap peak candidates/predictions to the center of peaks if they are slightly off--predicted locations won't snap farther than `peak-center-box-size`
    - Warning: set this too large, and you run the risk of a predicted peak snapping to stronger, nearby peak.
- `--peak-smoothing-window-size`: odd positive integer, units pixels
    - Size of smoothing window in pixels
    - Set ths around the size of a typical peak
    - Not a sensitive parameter; slightly improves performance of peak snapping
- `--region-growth-distance-threshold`: number, units pixels
    - Radius of neighborhood around a pixel in which the average neighbor intensity is calculated (see `--region-growth-minimum-intensity`)
    - The default of 1.5 is almost always good enough, though you could increase this in the case of high resolution but noisy data to improve the chances of correctly fitting weak peaks
- `--peak-pixel-outlier-threshold`: number, unitless
    - Peak pixels will be removed before fitting hull if their intensity is more than `peak-pixel-outlier-threshold` standard deviations away from the average intensity of pixels in the peak, which guards against the effects of noise and improves the quality of the final fitted hull
    - Default is 2.0, and you will probably never need to change it


## Algorithm

Following is a description of the convex hull peak fitting algorithm, including how each CLI parameter affects the algorithm.

1. The algorithm starts with a set of candidate/predicted peak locations $(x_0, y_0)$ (in pixel coordinates) and an intensity image $I(x,y)$
2. Then an averaging box filter with size `peak-smoothing-window-size`--which has to be an odd, positive integer to make the implementation easy for me--is used to smooth $I$ to produce a smoothed image $I_\text{s}$
3. In a box of **total size** (width and height) `peak-center-box-size` centered on $(x_0,y_0)$, an adjusted center $(x_1,y_1)$ is selected by finding the maximum intensity in $I_\text{s}$, the smoothed image. This allows slightly incorrect initial locations $(x_0,y_0)$ to "snap" onto the peak. Again, this must be an odd, positive integer for simplicity of implementation (even won't cause an error, but internally the odd integer one less will be used)
4. Clustering is used to create a set of peak pixels $P$. Start by setting ${P = \lbrace(x_1, y_1)\rbrace}$--just the adjusted initial point. Then repeat the following until no points are added to $P$:
    - For each point $(x,y) \notin P$ such that $d((x,y), P) < {}$`region-growth-distance-threshold`, calculate the average intensity ${A(x,y) = \frac{1}{|N|}\sum\limits_{(x_\text{n}, y_\text{n})\in N}I(x_\text{n},y_\text{n})}$ in the neighborhood $N$ of $(x,y)$ defined by ${N = \lbrace(x_\text{n}, y_\text{n}) : d((x, y), (x_\text{n}, y_\text{n})) \le r_1,\text{ } d((x_1,y_1), (x_\text{n},y_\text{n})) < r_2\rbrace}$, where $r_1 = {}$`region-growth-distance-threshold` and $r_2={}$`region-growth-maximum-pixel-radius`. If $A(x,y) \ge {}$`region-growth-minimum-intensity`, then add $(x,y)$ to $P$
5. Calculate the mean $\mu$ and standard deviation $\sigma$ of $\lbrace I(x,y) : (x,y) \in P\rbrace$. Remove points $(x,y)$ from $P$ if $|I(x,y) - \mu|/\sigma > {}$`peak-pixel-outlier-threshold`
6. If $|P| <{}$`peak-minimum-pixels`, then discard this peak
7. Compute the convex hull $H_\text{core}$ of $P$, which we will call the core peak hull.
8. Expand the core peak hull by scaling by a factor $s$ about its centroid to produce convex hulls $H_\text{peak}$ (the true peak hull, $s=1.1$), $H_\text{inner}$ (noise estimation inner boundary, $s=1.6$) and $H_\text{outer}$ (noise estimation outer boundary, $s=2.6$)
9. Integrate over the true peak hull $H_\text{peak}$ to obtain the peak intensity $I_\text{peak}$ and uncertainty $\sigma_\text{peak}$. Background noise is subtracted by estimating the average background noise per pixel by integrating over the annulus-like region $H_\text{outer} \setminus H_\text{inner}$.
10. If $I_\text{peak} / \sigma_\text{peak} <{}$`peak-minimum-signal-to-noise`, then discard this peak

## Gaussian Fit Integration

Assume the measured intensity image $I(x,y)$ is a Poisson process with rate function
```math
\lambda(x, y) = B + I_0\exp\left[-\frac{1}{2(1-\rho^2)}\left(\frac{(x-x')^2}{\sigma_x^2} - 2\frac{\rho(x-x')(y-y')}{\sigma_x\sigma_y} + \frac{(y-y')^2}{\sigma_y^2}\right)\right].
```
Here, $B$ is the background noise average intensity per pixel, and $I_0$ is the maximum peak intensity per pixel (not equal to the integrated intensity), and $\rho, \sigma_x, \sigma_y, x', y'$ are shape parameters describing the shape of the peak. We want to know the average integrated intensity of the peak minus the background noise. Assuming the noise $N(x,y)$ is also a Poisson process with constant rate function, the peak intensity $I(x,y) - N(x,y)$ is a Poisson process with rate $\lambda(x,y) - B$, so
```math
I_\text{final} = \mathbb{E}\left[\int (I(x,y) - N(x,y))\;\text{d}x\;\text{d}y\right] = \int (\lambda(x,y) - B)\;\text{d}x\;\text{d}y = 2\pi I_0\sigma_x\sigma_y\sqrt{1-\rho^2}.
```
Hence, we can calculate the final intensity $I_\text{final}$ by fitting the shape parameters and $B$ and $I_0$ using maximum likelihood estimation. The final uncertainty in the integrated intensity can likewise be estimated using these fitted parameters.
