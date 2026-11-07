import numpy as np


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
        nx = npixel[0]
        x = np.arange(nx)
        G = np.exp(-((x - avr[0])**2) / (2 * st_dev[0]**2))
        return G

    elif ndim == 2:
        nx, ny = npixel
        x = np.arange(nx)
        y = np.arange(ny)

        Gx = np.exp(-((x - avr[0])**2) / (2 * st_dev[0]**2))
        Gy = np.exp(-((y - avr[1])**2) / (2 * st_dev[1]**2))

        G = np.outer(Gx, Gy)

        # Shift the blob along x-axis
        shift_x = int(np.floor(avr[0] - nx / 2.0))
        G = np.roll(G, shift=shift_x, axis=0)

        return G

    else:
        raise ValueError("npixel must be a 1D or 2D list/tuple.")
