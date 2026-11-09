import numpy as np

from gaussian_function import gaussian_profile


# Distance, in radial widths (2L), beyond which a blob is treated as outside the image
# (exp(-6**2 / 2) ~ 1.5e-8 of the peak is left out):
#   - blobs are launched this far below the bottom of the image (syncom_load.syncom_blob_table)
#   - a launch is drawn until its centre is this far above the top of the image (launch_range)
EDGE_SIGMAS = 6.0


def angular_profiles(params, blobs):
    """
    Angular profile of every blob along the position-angle axis (NX pixels).

    It does not change with time, so syncom_core computes it once per run and
    passes it to syncom_image for every frame.

    Parameters:
        params: dict with key 'syncom_nx'
        blobs: blob table from syncom_load.syncom_blob_table (uses 'psi' and 'L')

    Returns:
        np.ndarray (n_blobs, NX): Gaussian of width L centred on psi, wrapping
        around 0/360 deg (a blob at 359.9 deg continues at 0 deg).
    """
    return gaussian_profile(params["syncom_nx"], blobs["psi"], blobs["L"], periodic=True)


def launch_range(blobs, time_t, ny):
    """
    Launch numbers k of each blob that are visible at time step time_t.

    A blob begins from its centre: launch k starts at tau = k * period (tau =
    time_t + time_0 is the blob's own clock) with its centre at r0, and its centre
    is now at r0 + v * (tau - k * period).

    Expected behaviour:
        k_hi = newest launch that has already happened (k_hi * period <= tau).
        k_lo = oldest launch that has not yet left the top of the image
               (centre below NY - 1 + EDGE_SIGMAS * 2L).
        k can be negative: every blob has been launching once per period since long
        before time step 0, so the image is already full at the first time step.
        A blob has k_hi - k_lo + 1 launches to draw (0 if k_lo > k_hi).

    Parameters:
        blobs: blob table (uses 'r0', 'v', 'period', 'L', 'time_0')
        time_t: time step [frames]
        ny: number of radial pixels

    Returns:
        tau, k_lo, k_hi: np.ndarray (n_blobs,) each; k_lo and k_hi hold integer values
    """
    r0, v, period = blobs["r0"], blobs["v"], blobs["period"]
    tau = time_t + blobs["time_0"]
    top = (ny - 1) + EDGE_SIGMAS * 2 * blobs["L"]

    k_hi = np.floor(tau / period)
    k_lo = np.ceil((r0 + v * tau - top) / (v * period))
    return tau, k_lo, k_hi


def syncom_image(params, blobs, time_t, gx=None, chunk_size=500):
    """
    Synthetic image at time step time_t: the sum of every visible launch of every blob.

    Expected behaviour:
        - Every blob is launched once per period, at tau = ..., -P, 0, P, 2P, ... with
          tau = time_t + time_0: it was already launching before time step 0 and keeps
          launching until the last one, so the image is full from the first frame.
        - A blob begins from its centre, not from its edge: at launch its centre is at
          r0, 6 radial widths below the bottom of the image (behind the occulter). It
          enters the image fully formed, and the mean brightness is the same at every
          radius, including the bottom and top rows.
        - Each launch moves outward at the blob's speed v and leaves the image at
          the top. Launch centres of one blob are v * P pixels apart along the radius,
          and a new centre crosses the bottom row (5 Rsun) every P frames.
        - Each launch is an angular Gaussian (width L, wraps around 0/360 deg)
          times a radial Gaussian (width 2L), with peak 1.
        - Only launches that have already happened and are still in the image
          are drawn (see launch_range).
        - Nothing is written to disk; use syncom_fits.write_fits_frames for FITS.

    Parameters:
        params: dict with keys 'syncom_nx', 'syncom_ny', 'syncom_n_blobs'
        blobs: blob table from syncom_load.syncom_blob_table:
            psi [px of 0.1 deg], r0 [px from the bottom of the image, negative = below],
            v [px/frame], period [frames], L [px], time_0 [frames]
        time_t: time step [frames]
        gx: optional angular profiles from angular_profiles(params, blobs);
            computed here if not given (pass it when making many frames)
        chunk_size: blobs processed at once (limits memory use)

    Returns:
        img: np.ndarray (NX, NY), axis 0 = position angle, axis 1 = radius
    """
    n_blobs = params["syncom_n_blobs"]
    NX = params["syncom_nx"]
    NY = params["syncom_ny"]

    blobs = {key: np.asarray(blobs[key], dtype=float)[:n_blobs]
             for key in ("psi", "r0", "v", "period", "L", "time_0")}
    if gx is None:
        gx = angular_profiles(params, blobs)

    tau, k_lo, k_hi = launch_range(blobs, time_t, NY)

    img = np.zeros((NX, NY), dtype=float)

    for start in range(0, n_blobs, chunk_size):
        sl = slice(start, min(start + chunk_size, n_blobs))
        r0, v, period, L = blobs["r0"][sl], blobs["v"][sl], blobs["period"][sl], blobs["L"][sl]
        lo, hi, t = k_lo[sl], k_hi[sl], tau[sl]

        # Radial profile of each blob: one Gaussian (width 2L) per visible launch.
        # Blob i has hi[i] - lo[i] + 1 launches; loop up to the largest count in
        # the chunk and switch off the extra ones with `visible`.
        gy = np.zeros((len(r0), NY))
        n_launches = int(max(0.0, (hi - lo).max() + 1))
        for j in range(n_launches):
            k = lo + j                               # launch number
            visible = k <= hi                        # this blob has a j-th visible launch
            centre = r0 + v * (t - k * period)       # [pixels above the bottom of the image]
            gy += np.where(visible[:, None], gaussian_profile(NY, centre, 2 * L), 0.0)

        # Sum of outer(gx_i, gy_i) over all blobs in the chunk
        img += gx[sl].T @ gy

    return img
