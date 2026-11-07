import numpy as np
import time
from astropy.io import fits

from syncom_path import syncom_run_name, syncom_output_path


def _roll_rows(a, shifts):
    """
    Row-wise np.roll: rolls each row a[k] by its own shifts[k] along axis 1.
    Equivalent to np.stack([np.roll(a[k], shifts[k]) for k in range(len(a))]).
    """
    n = a.shape[1]
    idx = (np.arange(n)[None, :] - shifts[:, None]) % n
    return np.take_along_axis(a, idx, axis=1)

def syncom_image(params, syncom_version, scalefactor, time_0, radial_i, v_array,
                 period_array, L_array, PSI, time_t, chunk_size=500):
    """
    Generates a synthetic solar wind image at a given time step (vectorized).

    Each blob is separable: an angular profile gx (along NX) times a radial
    profile gy (along NY). The harmonic copies and the radial displacement are
    rolls along the radial axis only, so they are applied to gy alone, and the
    sum over all blobs becomes one matrix product: img = GX.T @ GY.
    Gives the same image as the original per-blob loop (up to floating-point rounding).

    Parameters:
        ModPramsStruct: dict-like object with keys 'SYNCOM_NX', 'SYNCOM_NY', 'SYNCOM_N_BLOBS'
        syncom_version: string identifier for simulation version
        time_0: initial time offset (in frames)
        radial_i: array of initial radial positions [pixels]
        v_array: array of radial velocities [pixels/frame]
        period_array: array of blob periods [seconds]
        L_array: array of blob sizes [degrees]
        PSI: array of position angles [degrees]
        time_t: current time step [frames]
        chunk_size: blobs processed at once (limits memory use)

    Returns:
        img: 2D NumPy array representing the synthetic image
    """
    # start_time = time.time()

    n_blobs = params["syncom_n_blobs"]
    NX = params["syncom_nx"]
    NY = params["syncom_ny"]
    cadence = params["syncom_cadence"]

    # Per-blob inputs, first n_blobs entries (as in the original loop)
    PSI = np.asarray(PSI, dtype=float)[:n_blobs]
    radial_t = np.asarray(radial_i, dtype=float)[:n_blobs] + (time_t + time_0) * v_array  # [pixels] at time_t
    L_array = np.asarray(L_array, dtype=float)[:n_blobs]
    period_frames = np.asarray(period_array, dtype=float)[:n_blobs]
    v_array = np.asarray(v_array, dtype=float)[:n_blobs]

    time_0 = np.broadcast_to(np.asarray(time_0, dtype=float), (n_blobs,))

    x = np.arange(NX)
    y = np.arange(NY)
    img = np.zeros((NX, NY), dtype=float)

    for start in range(0, n_blobs, chunk_size):
        sl = slice(start, min(start + chunk_size, n_blobs))
        psi, L, P, v, r = PSI[sl], L_array[sl], period_frames[sl], v_array[sl], radial_t[sl]

        # Angular profile: Gaussian centred on psi, wrapping across 0/360 deg
        dx = (x[None, :] - psi[:, None] + NX / 2.0) % NX - NX / 2.0
        gx = np.exp(-dx**2 / (2 * L[:, None]**2))

        # Radial profile (st_dev = 2L): the blob plus its 6 harmonic copies
        gy = np.zeros((len(psi), NY))

        # Add repeated/perturbed blobs at harmonic intervals
        # (astype(int) truncates toward zero, like int())
        for n in [0,2, 4, 6, -2, -4, -6]:
            shift_amount = r + n * P * v + n * L
            gy += np.exp(-(y[None, :] - shift_amount[:, None])**2 / (2 * (2 * L[:, None])**2))

        # Shift for radial displacement over time
        # gy = _roll_rows(gy, radial_t[sl].astype(int))

        # Sum of outer(gx_i, gy_i) over all blobs in the chunk
        img += gx.T @ gy

    # Save image to SynCOM_runs/<run name>/FITS/<run name>_<time_t>.fts
    run_name = syncom_run_name(syncom_version, params["syncom_n_blobs"], scalefactor)
    filename = syncom_output_path(run_name, "fits", f"{time_t}.fts")
    fits.writeto(filename, img.T, overwrite=True)

    # print(f"Run time for step {time.time() - start_time} s")

    return img

