import numpy as np


def gaussian_profile(npixel, avr, st_dev, periodic=False):
    """
    Generates 1D Gaussian profile(s) along an axis of npixel pixels (vectorized over blobs).

    Parameters:
        npixel (int): Number of pixels along the axis
        avr (float or np.ndarray): Center(s) of the Gaussian [pixels]
        st_dev (float or np.ndarray): Standard deviation(s) [pixels], same shape as avr
        periodic (bool): If True the axis wraps around (e.g. position angle 0/360 deg),
            so a Gaussian near one edge continues on the other edge

    Returns:
        np.ndarray: shape (npixel,) for a scalar avr, or (len(avr), npixel) for an array of centers
    """
    x = np.arange(npixel)
    avr = np.asarray(avr, dtype=float)[..., None]
    st_dev = np.asarray(st_dev, dtype=float)[..., None]

    dx = x - avr
    if periodic:
        # Shortest distance on the circle: values in [-npixel/2, npixel/2)
        dx = (dx + npixel / 2.0) % npixel - npixel / 2.0

    return np.exp(-dx**2 / (2 * st_dev**2))


def gaussian_wave(npixel, avr, st_dev):
    """
    Generates a 1D or 2D Gaussian wave (propagating blob).

    Parameters:
        npixel (list or tuple): [nx] for 1D or [nx, ny] for 2D
        avr (list or tuple): Center of the Gaussian in each dimension
        st_dev (list or tuple): Standard deviation in each dimension

    Returns:
        np.ndarray: 1D or 2D array with the Gaussian blob
    """
    # Determine dimensionality
    ndim = len(npixel)

    if ndim == 1:
        return gaussian_profile(npixel[0], avr[0], st_dev[0])

    elif ndim == 2:
        nx, ny = npixel

        # Position angle (x) is periodic, radial distance (y) is not
        Gx = gaussian_profile(nx, avr[0], st_dev[0], periodic=True)
        Gy = gaussian_profile(ny, avr[1], st_dev[1])

        return np.outer(Gx, Gy)

    else:
        raise ValueError("npixel must be a 1D or 2D list/tuple.")
