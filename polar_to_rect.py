from scipy.ndimage import map_coordinates
import numpy as np
from syncom_prams import POLAR_PHI_RANGE, POLAR_RHO_RANGE


def polar_to_rect(img_p, phi_range=POLAR_PHI_RANGE, rho_range=POLAR_RHO_RANGE, n=1024):
    """
    Transform a polar image into a Cartesian image (IDL POLAR_TO_RECT).

    Parameters
    ----------
    img_p : ndarray (Np, Nr)   Polar image; axis 0 = azimuth phi, axis 1 = radius rho.
    phi_range : (min, max)     Azimuth of the first and last pixel [deg].
    rho_range : (min, max)     Radius of the first and last pixel (e.g. Rsun).
    n : int                    Number of pixels in x and y of the output.

    Returns
    -------
    img_rect : ndarray (n, n)  Indexed [ix, iy]; x and y span -rho_max ... rho_max.
    """
    img_p = np.asarray(img_p, dtype=float)
    n_phi, n_rho = img_p.shape

    # Polar sampling steps
    d_phi = (phi_range[1] - phi_range[0]) / (n_phi - 1)       # [deg]
    d_rho = (rho_range[1] - rho_range[0]) / (n_rho - 1)

    # Cartesian grid spanning the polar samples (x = rho cos(phi), y = -rho sin(phi))
    phi_1d = np.radians(phi_range[0] + d_phi * np.arange(n_phi))
    x_edge = rho_range[1] * np.cos(phi_1d)
    y_edge = -rho_range[1] * np.sin(phi_1d)
    x_1d = np.linspace(x_edge.min(), x_edge.max(), n)
    y_1d = np.linspace(y_edge.min(), y_edge.max(), n)
    x_2d, y_2d = np.meshgrid(x_1d, y_1d, indexing="ij")

    # Polar coordinates of every Cartesian pixel, as fractional indices into img_p
    rho = np.hypot(x_2d, y_2d)
    phi = np.degrees(np.arctan2(-y_2d, x_2d)) % 360.0
    i_phi = (phi - phi_range[0]) / d_phi
    i_rho = (rho - rho_range[0]) / d_rho

    # Bilinear interpolation; the first azimuth column is repeated after the last to close the circle
    img_closed = np.concatenate([img_p, img_p[:1]], axis=0)
    img_rect = map_coordinates(img_closed, [i_phi, i_rho], order=1, mode="constant", cval=0.0)

    # Sun mask: zero inside the minimum radial distance (and outside the sampled annulus)
    img_rect[(rho < rho_range[0]) | (rho > rho_range[1])] = 0.0
    
    return img_rect