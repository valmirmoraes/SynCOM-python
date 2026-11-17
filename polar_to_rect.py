import numpy as np
from syncom_prams import polar_phi_range, polar_ranges


def polar_to_rect(img_p, phi_range=None, rho_range=None, n=1024, params=None):
    """
    Transform a polar image into a Cartesian image (IDL POLAR_TO_RECT).

        rect = polar_to_rect(cube[:, :, k], params=params)     # ranges of the run

    Parameters
    ----------
    img_p : ndarray (Np, Nr)   Polar image; axis 0 = azimuth phi, axis 1 = radius rho.
    phi_range : (min, max)     Azimuth of the first and last pixel [deg].
    rho_range : (min, max)     Radius of the first and last pixel (e.g. Rsun).
    n : int                    Number of pixels in x and y of the output.
    params : dict              Run parameters; ranges not given are taken from them
                               (syncom_prams.polar_ranges). Without params, phi_range follows
                               from the image width, and without rho_range the image must have
                               the default NY, otherwise ValueError.

    Returns
    -------
    img_rect : ndarray (n, n)  Indexed [ix, iy]; x and y span -rho_max ... rho_max.
    """
    img_p = np.asarray(img_p, dtype=float)
    n_phi, n_rho = img_p.shape
    if phi_range is None and params is None:
        phi_range = polar_phi_range({"syncom_nx": n_phi})      # needs only NX = the image width
    if phi_range is None or rho_range is None:
        run_phi, run_rho = polar_ranges(params, shape=img_p.shape)
        phi_range = run_phi if phi_range is None else phi_range
        rho_range = run_rho if rho_range is None else rho_range

    return apply_rect_sampling(img_p, rect_sampling(img_p.shape, phi_range, rho_range, n))


def rect_sampling(shape, phi_range, rho_range, n=1024):
    """
    Where every Cartesian pixel of polar_to_rect samples a polar frame of this shape, computed once.

    Returns a dict for apply_rect_sampling: the four neighbouring polar pixels of every Cartesian
    pixel inside the field (azimuth wraps from the last column to the first) and their bilinear
    weights. A movie computes this once and then reads only those pixels from each frame.
    """
    n_phi, n_rho = shape[:2]

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

    # Polar coordinates of every Cartesian pixel, as fractional indices into the polar frame
    rho = np.hypot(x_2d, y_2d)
    phi = np.degrees(np.arctan2(-y_2d, x_2d)) % 360.0
    i_phi = (phi - phi_range[0]) / d_phi
    i_rho = (rho - rho_range[0]) / d_rho

    # Sun mask: only pixels between the minimum and maximum radial distance are sampled
    inside = np.flatnonzero((rho >= rho_range[0]) & (rho <= rho_range[1]))
    ip, ir = i_phi.ravel()[inside], i_rho.ravel()[inside]
    p0, r0 = np.floor(ip), np.floor(ir)
    wp, wr = ip - p0, ir - r0
    p0 = p0.astype(np.intp)
    r0 = np.clip(r0.astype(np.intp), 0, n_rho - 1)
    p1 = (p0 + 1) % n_phi                      # azimuth closes the circle: after the last column comes the first
    r1 = np.minimum(r0 + 1, n_rho - 1)
    return {"n": n, "inside": inside, "p0": p0 % n_phi, "p1": p1, "r0": r0, "r1": r1, "wp": wp, "wr": wr}


def apply_rect_sampling(img_p, s, k=None):
    """
    Cartesian image (n, n) from a polar frame img_p[phi, rho], or from frame k of a cube img_p[phi, rho, t].

    Bilinear interpolation with the indices and weights of rect_sampling; 0 inside the occulter and
    outside the field. With k, only the sampled pixels of frame k are read from the cube.
    """
    def get(p, r):
        return img_p[p, r] if k is None else img_p[p, r, k]
    wp, wr = s["wp"], s["wr"]
    val = ((1 - wp) * (1 - wr) * get(s["p0"], s["r0"]) + wp * (1 - wr) * get(s["p1"], s["r0"])
           + (1 - wp) * wr * get(s["p0"], s["r1"]) + wp * wr * get(s["p1"], s["r1"]))
    img_rect = np.zeros(s["n"] * s["n"])
    img_rect[s["inside"]] = val
    return img_rect.reshape(s["n"], s["n"])
