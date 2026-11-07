import numpy as np
from astropy.io import fits

from gaussian_function import gaussian_profile
from syncom_path import syncom_run_name, syncom_output_path


def syncom_image(params, syncom_version, scalefactor, time_0, radial_i, v_array,
                 period_array, L_array, PSI, time_t, chunk_size=500):
    """
    Generates a synthetic solar wind image at a given time step (vectorized).

    Blobs sit at float (sub-pixel) radial positions radial_i + (time_t + time_0) * v.
    Each blob has 7 copies (n = 0, +-2, +-4, +-6) spaced by n * period * v + n * L,
    with the period in frames. Position angle wraps around 0/360 deg; the radial
    axis does not, so blobs leave the field. Each blob is an angular profile gx
    times a radial profile gy (gaussian_profile), so the sum over blobs is one
    matrix product per chunk: img += gx.T @ gy.

    Parameters:
        params: dict with keys 'syncom_nx', 'syncom_ny', 'syncom_n_blobs', 'syncom_cadence'
        syncom_version: string identifier for simulation version
        scalefactor: blob size scale factor (used in the run name)
        time_0: initial time offset per blob [frames]
        radial_i: array of initial radial positions [pixels]
        v_array: array of radial velocities [pixels/frame]
        period_array: array of blob periods [seconds]
        L_array: array of blob sizes [pixels]
        PSI: array of position angles [pixels of 0.1 degree]
        time_t: current time step [frames]
        chunk_size: blobs processed at once (limits memory use)

    Returns:
        img: 2D NumPy array (NX, NY) representing the synthetic image
    """
    n_blobs = params["syncom_n_blobs"]
    NX = params["syncom_nx"]
    NY = params["syncom_ny"]
    cadence = params["syncom_cadence"]

    # Per-blob inputs, first n_blobs entries
    PSI = np.asarray(PSI, dtype=float)[:n_blobs]
    L_array = np.asarray(L_array, dtype=float)[:n_blobs]
    v_array = np.asarray(v_array, dtype=float)[:n_blobs]
    period_frames = np.asarray(period_array, dtype=float)[:n_blobs] / cadence   # [frames]
    time_0 = np.broadcast_to(np.asarray(time_0, dtype=float), (n_blobs,))
    radial_t = np.asarray(radial_i, dtype=float)[:n_blobs] + (time_t + time_0) * v_array  # [pixels]

    img = np.zeros((NX, NY), dtype=float)

    for start in range(0, n_blobs, chunk_size):
        sl = slice(start, min(start + chunk_size, n_blobs))
        psi, L, P, v, r = PSI[sl], L_array[sl], period_frames[sl], v_array[sl], radial_t[sl]

        # Angular profile: Gaussian centred on psi, wrapping across 0/360 deg
        gx = gaussian_profile(NX, psi, L, periodic=True)

        # Radial profile (st_dev = 2L): the blob plus its 6 harmonic copies
        gy = np.zeros((len(psi), NY))
        for n in [0, 2, 4, 6, -2, -4, -6]:
            gy += gaussian_profile(NY, r + n * P * v + n * L, 2 * L)

        # Sum of outer(gx_i, gy_i) over all blobs in the chunk
        img += gx.T @ gy

    # Save image to SynCOM_runs/<run name>/FITS/<run name>_<time_t:04d>.fts
    # Zero-padded index -> files sort in time order; img.T -> (NY, NX) = 659 x 3600, as FlowTrack expects
    run_name = syncom_run_name(syncom_version, params["syncom_n_blobs"], scalefactor)
    filename = syncom_output_path(run_name, "fits", f"{time_t:04d}.fts")
    fits.writeto(filename, img.T, overwrite=True)

    return img
