"""
Planets in SynCOM frames: Mercury to Neptune (PLANET_DB).

On/off switch: params["syncom_planets"], set with default_syncom_prams(SYNCOM_PLANETS=...):
    SYNCOM_PLANETS=("mercury", "jupiter")   -> those planets are drawn into every frame
    (any of PLANET_DB: mercury, venus, mars, jupiter, saturn, uranus, neptune)
    SYNCOM_PLANETS=()                       -> no planets (default); the frames are exactly the blob-only frames
The planet positions need the UTC time of time step 0: default_syncom_prams(SYNCOM_START_TIME="2026-07-26T06:40:00").
Time step t is at SYNCOM_START_TIME + t * cadence.

Flow (syncom_core): syncom_planet_table once per run -> local_sigma once (blob-only first frame)
-> add_planets on every frame.

Positions follow movie_generic.ipynb (PUNCH explore): body_direction gives the geocentric astrometric
direction; elongation and position angle (from solar north, counterclockwise through east) are taken
around the Sun; radii are elongations in units of the Sun's angular radius (R_sun), as on the PUNCH
elongation grid.

Brightness and shape: every planet's V magnitude comes from the ephemeris (planet_ephemeris, per
time step); how PUNCH shows a source of that magnitude is an instrument property, measured once on
PUNCH L3 CAM frames (PUNCH_CAL, calibrate_punch): a photometric zero point, the local fluctuation
vs radius, the saturation ceiling of the S/N vs radius, and the unsaturated shape. A planet is drawn
as sigma * min(linear S/N * shape, ceiling): above the ceiling its core is flat and grows with the
overexposure, as in PUNCH. Planets outside the frame for the whole run are not modelled.

Polar frames: axis 0 = column (NX per 360 deg), axis 1 = radius (row i at POLAR_R_MIN + i * pixel).
Column convention (solar_pa_to_column): column angle = 270 deg - solar PA, so polar_to_rect shows a
planet where a PUNCH image does (solar north up, east left).
"""
import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord, get_body_barycentric
from astropy.time import Time, TimeDelta
from scipy.special import erf

from syncom_prams import POLAR_R_MIN, polar_rho_range


FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # sigma = FWHM / 2.3548
RSUN_KM = 695700.0                                          # [km] solar radius (RSUN_REF of the PUNCH headers)
SOLAR_POLE = (286.13, 63.87)                                # [deg] RA, Dec (ICRS) of the solar rotation axis (IAU)
SIGMA_WIDTH_DEG = 2.0                                       # [deg] running-median width of local_sigma
DRAW_SIGMAS = 6.0                                           # a planet is drawn within +-6 sigma of its centre


# PUNCH L3 CAM instrument calibration, measured with calibrate_punch(fits_paths, params) on the two files in
# "source" (2026-07-18 11:28 and 2026-07-26 06:40), SynCOM grid NY 7779, pixel 0.0225 Rsun; re-run it to check
# (test notebook). Peak = brightest pixel minus the background (median of the 20-30 px ring); S/N = peak /
# local_sigma at the source's radius in its own frame.
#   zp_peak, zp_vmag   [MSB]   peak and V magnitude of the unsaturated zero-point source (Mercury 07-18):
#                              a planet of magnitude V has an unsaturated peak zp_peak * 10 ** (-0.4 * (V - zp_vmag))
#   sigma_r, sigma     [Rsun], [MSB]  local_sigma of the zero-point frame vs radius (interpolated; constant
#                              outside the measured range)
#   sat_r, sat_snr     [Rsun]  saturation ceiling of the peak S/N vs radius, from the saturated sources
#                              (Jupiter 07-18, Mercury 07-26, Venus 07-18 and 07-26); interpolated, constant outside
#   psf_unsat                  unsaturated shape of the zero-point source: core FWHM [deg], halo FWHM [deg],
#                              halo fraction of the peak
#   psf_growth, psf_r0 [1/Rsun], [Rsun]  the core FWHM grows with radius: core * (1 + psf_growth * max(r - psf_r0, 0)),
#                              fitted to the apparent sizes (half maximum, 30 sigma) of all the planets seen
# Planet image: sigma * min(lin * shape(x), sat_snr(r)), lin = zp_peak * 10 ** (-0.4 * (V - zp_vmag)) / sigma(r),
# shape of peak 1; the peak S/N is min(lin, sat_snr(r)).
PUNCH_CAL = {
    "zp_peak": 3.695793874386975e-12,
    "zp_vmag": 3.828063315030903,
    "sigma_r": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0],
    "sigma": [1.0751601468067671e-14, 8.456197634713954e-15, 7.109111635472205e-15, 6.1472900999978525e-15, 6.1514522171087425e-15, 5.495445979373567e-15, 5.1484294652795455e-15, 4.925608670113903e-15, 4.859100485790945e-15, 4.934112633644298e-15, 4.747328434670464e-15, 5.242820701926767e-15, 4.656656861899928e-15, 5.07136007493607e-15, 5.173887765217177e-15, 5.0362709306111754e-15, 5.2426245870284146e-15, 5.322848485056997e-15, 4.9477838244057655e-15, 5.00666365706254e-15, 5.082392622602809e-15, 4.9731023833421755e-15, 4.895698428273539e-15, 5.223630336162797e-15, 4.965028578718616e-15, 5.456241094397425e-15, 6.055274329070097e-15, 6.070152269199416e-15, 6.8264034763267276e-15, 7.780368870713354e-15, 8.37338611310301e-15],
    "sat_r": [30.933250837695223, 65.1895852894256, 167.01858501691706, 170.5031084902837],
    "sat_snr": [2182.124800480083, 2330.9853664131915, 5699.655434471151, 5760.608940977266],
    "psf_unsat": [0.0859184450289617, 0.27728534689515255, 0.019748312496604827],
    "psf_growth": 0.002782478976605137,
    "psf_r0": 36.41119122370384,
    "source": "PUNCH_L3_CAM_20260718112800_v0l.fits, PUNCH_L3_CAM_20260726064000_v0l.fits",
}


# ------------------------------------------------------------------ #
# Positions                                                          #
# ------------------------------------------------------------------ #
def body_direction(name, t):
    """
    Direction of a solar-system body seen from the Earth's center at ``t`` (astropy Time), in the
    convention of the PUNCH header's alternate (``A``) RA/Dec WCS.

    Copied from movie_generic.ipynb (punch-explore, R. Attie): geocentric astrometric position,
    light-time corrected (3 iterations), ICRS axes, without annual aberration.

    Returns
    -------
    (SkyCoord, Quantity)
        Direction (ICRS, no distance) and geocentric distance.
    """
    earth = get_body_barycentric('earth', t)
    delay = 0 * u.s
    for _ in range(3):  # light-time iteration
        vec = get_body_barycentric(name, t - delay) - earth
        delay = vec.norm() / const.c
    body = SkyCoord(vec, frame='icrs')  # geocentric vector: only its direction and length are used
    return SkyCoord(ra=body.ra, dec=body.dec, frame='icrs'), body.distance


def _unit(coord):
    """Unit vectors (3, N) of SkyCoord directions."""
    ra, dec = np.atleast_1d(coord.ra.rad), np.atleast_1d(coord.dec.rad)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


# V magnitudes (Mallama & Hilton 2018, Astronomy and Computing 25, 10): 5 log10(r * delta) + V(1, phase),
# r = heliocentric and delta = geocentric distance [AU], phase angle [deg]. Saturn and Uranus also use
# the sub-Earth (and sub-Sun) latitude on the planet [deg] (keywords sublat_earth, sublat_sun).
def mercury_vmag(r_au, delta_au, phase_deg, **geometry):
    """V magnitude of Mercury."""
    a = np.asarray(phase_deg, dtype=float)
    return (5.0 * np.log10(r_au * delta_au) - 0.613 + 6.3280e-02 * a - 1.6336e-03 * a**2 + 3.3644e-05 * a**3
            - 3.4265e-07 * a**4 + 1.6893e-09 * a**5 - 3.0334e-12 * a**6)


def venus_vmag(r_au, delta_au, phase_deg, **geometry):
    """V magnitude of Venus (two branches, joined at phase 163.7 deg)."""
    a = np.asarray(phase_deg, dtype=float)
    low = -4.384 - 1.044e-03 * a + 3.687e-04 * a**2 - 2.814e-06 * a**3 + 8.938e-09 * a**4
    high = 236.05828 - 2.81914 * a + 8.39034e-03 * a**2
    return 5.0 * np.log10(r_au * delta_au) + np.where(a < 163.7, low, high)


def mars_vmag(r_au, delta_au, phase_deg, **geometry):
    """V magnitude of Mars (phase <= 50 deg; the orbital and rotational terms, < 0.1 mag, are left out)."""
    a = np.asarray(phase_deg, dtype=float)
    return 5.0 * np.log10(r_au * delta_au) - 1.601 + 0.02267 * a - 0.0001302 * a**2


def jupiter_vmag(r_au, delta_au, phase_deg, **geometry):
    """V magnitude of Jupiter (phase <= 12 deg, always the case from the Earth)."""
    a = np.asarray(phase_deg, dtype=float)
    return 5.0 * np.log10(r_au * delta_au) - 9.395 - 3.7e-04 * a + 6.16e-04 * a**2


def saturn_vmag(r_au, delta_au, phase_deg, sublat_earth=0.0, **geometry):
    """V magnitude of Saturn with its rings (phase <= 6.5 deg; ring tilt beta = |sub-Earth latitude| <= 27 deg)."""
    a = np.asarray(phase_deg, dtype=float)
    sb = np.sin(np.radians(np.abs(np.asarray(sublat_earth, dtype=float))))
    return 5.0 * np.log10(r_au * delta_au) - 8.914 - 1.825 * sb + 0.026 * a - 0.378 * sb * np.exp(-2.25 * a)


def uranus_vmag(r_au, delta_au, phase_deg, sublat_earth=0.0, sublat_sun=0.0, **geometry):
    """V magnitude of Uranus (phase <= 3.1 deg); phi' = mean of |sub-Earth| and |sub-Sun| latitudes."""
    a = np.asarray(phase_deg, dtype=float)
    phi = 0.5 * (np.abs(np.asarray(sublat_earth, dtype=float)) + np.abs(np.asarray(sublat_sun, dtype=float)))
    return 5.0 * np.log10(r_au * delta_au) - 7.110 - 8.4e-04 * phi + 6.587e-03 * a + 1.045e-04 * a**2


def neptune_vmag(r_au, delta_au, phase_deg, **geometry):
    """V magnitude of Neptune (phase <= 1.9 deg; the value for dates after 2000)."""
    a = np.asarray(phase_deg, dtype=float)
    return 5.0 * np.log10(r_au * delta_au) - 7.00 + 7.944e-03 * a + 9.617e-05 * a**2


# Rotation poles (IAU, RA and Dec in ICRS [deg]) for the sub-Earth / sub-Sun latitudes
POLES = {"saturn": (40.589, 83.537), "uranus": (257.311, -15.175)}

# Planet database. "punch": "confirmed" = seen in the PUNCH L3 CAM calibration frames, where position,
# S/N and shape were checked against SynCOM (syncom_notebook_tests.ipynb, section 9); "ephemeris" =
# position and V magnitude from the ephemeris, drawn with the same PUNCH_CAL, but not seen in those frames.
PLANET_DB = {
    "mercury": {"vmag": mercury_vmag, "punch": "confirmed",
                "evidence": "2026-07-18 11:28 (36 Rsun, linear: zero point) and 2026-07-26 06:40 (65 Rsun, saturated)"},
    "venus":   {"vmag": venus_vmag, "punch": "confirmed",
                "evidence": "2026-07-18 11:28 (167 Rsun) and 2026-07-26 06:40 (170 Rsun), saturated"},
    "mars":    {"vmag": mars_vmag, "punch": "confirmed",
                "evidence": "2026-07-18 11:28 (160 Rsun) and 2026-07-26 06:40 (168 Rsun), at the ceiling / linear"},
    "jupiter": {"vmag": jupiter_vmag, "punch": "confirmed",
                "evidence": "2026-07-18 11:28 (31 Rsun), saturated; 2026-07-26 inside the area without data"},
    "saturn":  {"vmag": saturn_vmag, "punch": "ephemeris",
                "evidence": "outside the PUNCH field in the calibration frames (elongation 101-109 deg)"},
    "uranus":  {"vmag": uranus_vmag, "punch": "ephemeris",
                "evidence": "outside the PUNCH field in the calibration frames (elongation 51-59 deg)"},
    "neptune": {"vmag": neptune_vmag, "punch": "ephemeris",
                "evidence": "outside the PUNCH field in the calibration frames (elongation 112-119 deg)"},
}
VMAG = {name: entry["vmag"] for name, entry in PLANET_DB.items()}


def planet_ephemeris(name, times):
    """
    Position of a planet around the Sun, as seen from the Earth's center, at each time.

    Expected behaviour:
        - Same body direction as movie_generic.ipynb (body_direction).
        - eps: angle between the planet and the Sun's center; pa: position angle from solar north,
          counterclockwise through east, in [0, 360); r = eps / theta_sun, with theta_sun the Sun's
          angular radius (equals RSUN_ARC of the PUNCH headers).
        - Agrees with the PUNCH header path (planet_pixel_punch) to ~0.01 deg.

    Parameters:
        name (str): body name known to astropy ('mercury', ...)
        times (astropy Time): scalar or array

    Returns:
        dict of np.ndarray (one value per time):
            eps [deg], pa [deg], theta_sun [deg], r [Rsun], phase [deg], r_helio [AU], delta [AU], vmag,
            sublat_earth [deg] (latitude of the Earth seen from the planet; Saturn: ring tilt; 0 if no pole)
    """
    times = Time(np.atleast_1d(times.utc.isot), scale="utc") if times.isscalar else times
    body, delta = body_direction(name, times)
    sun, d_sun = body_direction('sun', times)
    ub, us = _unit(body), _unit(sun)
    up = _unit(SkyCoord(ra=SOLAR_POLE[0] * u.deg, dec=SOLAR_POLE[1] * u.deg, frame='icrs'))

    # Solar north and east on the sky at the Sun's direction (east = north x Sun, i.e. to the left)
    north = up - np.sum(up * us, axis=0) * us
    north /= np.linalg.norm(north, axis=0)
    east = np.cross(north, us, axis=0)

    cos_eps = np.clip(np.sum(ub * us, axis=0), -1.0, 1.0)
    d = ub - cos_eps * us
    eps = np.degrees(np.arccos(cos_eps))
    pa = np.degrees(np.arctan2(np.sum(d * east, axis=0), np.sum(d * north, axis=0))) % 360.0
    theta_sun = np.degrees(np.arcsin((RSUN_KM * u.km / d_sun).decompose().value))

    # Phase angle (Sun-planet-Earth), with the light-time corrected planet position
    earth = get_body_barycentric('earth', times)
    sun_b = get_body_barycentric('sun', times)
    delay = (delta / const.c).to(u.s)
    p = get_body_barycentric(name, times - delay)
    to_sun = (sun_b - p).xyz.to_value(u.AU)
    to_earth = (earth - p).xyz.to_value(u.AU)
    r_helio = np.linalg.norm(to_sun, axis=0)
    delta_au = np.linalg.norm(to_earth, axis=0)
    phase = np.degrees(np.arccos(np.clip(np.sum(to_sun * to_earth, axis=0) / (r_helio * delta_au), -1, 1)))
    sublat_earth = sublat_sun = np.zeros_like(eps)
    if name in POLES:                                  # latitude of the Earth and the Sun seen from the planet
        pole = _unit(SkyCoord(ra=POLES[name][0] * u.deg, dec=POLES[name][1] * u.deg, frame="icrs"))
        sublat_earth = np.degrees(np.arcsin(np.clip(np.sum(pole * to_earth, axis=0) / delta_au, -1, 1)))
        sublat_sun = np.degrees(np.arcsin(np.clip(np.sum(pole * to_sun, axis=0) / r_helio, -1, 1)))
    vmag = (VMAG[name](r_helio, delta_au, phase, sublat_earth=sublat_earth, sublat_sun=sublat_sun)
            if name in VMAG else np.full_like(eps, np.nan))

    return {"eps": eps, "pa": pa, "theta_sun": theta_sun, "r": eps / theta_sun, "phase": phase,
            "r_helio": r_helio, "delta": delta_au, "vmag": vmag, "sublat_earth": sublat_earth}


def solar_pa_to_column(pa_deg, nx=3600):
    """
    Column of a solar position angle in a SynCOM polar frame of nx columns (fractional pixels).

    Expected behaviour: column angle = (270 - PA) mod 360, so with polar_to_rect solar north
    (PA 0) is up, east (PA 90) left, south (PA 180) down and west (PA 270) right, as in PUNCH images.
    """
    return ((270.0 - np.asarray(pa_deg, dtype=float)) % 360.0) * nx / 360.0


def column_to_solar_pa(col, nx=3600):
    """Inverse of solar_pa_to_column."""
    return (270.0 - np.asarray(col, dtype=float) * 360.0 / nx) % 360.0


# ------------------------------------------------------------------ #
# Planet table (once per run) and drawing (every frame)               #
# ------------------------------------------------------------------ #
def syncom_planet_table(params, time0, time_t):
    """
    Track of every switched-on planet at time steps time0 ... time_t-1, in image units.

    Expected behaviour:
        - Returns None when params["syncom_planets"] is empty or missing (planets off), or when every
          switched-on planet is outside the frame for the whole run.
        - Time step t is at params["syncom_start_time"] + t * params["syncom_cadence"].
        - Everything that depends on the planet and the date comes from the ephemeris (planet_ephemeris):
          position, motion, V magnitude. How PUNCH shows it comes from PUNCH_CAL (punch_snr_shape):
          linear S/N, saturation ceiling and unsaturated shape at that radius.
        - A planet whose centre stays outside the frame (more than 6 halo widths beyond its radial
          edges) for the whole run is not modelled: a line is printed and it is left out of the table.
        - in_view: centre inside the radial range of the frame; calibrated: radius inside the range
          where PUNCH_CAL was measured (outside it the nearest measured values are used).
        - Planets not in PLANET_DB raise ValueError. punch_confirmed tells whether the planet was
          checked against PUNCH frames (PLANET_DB[name]["punch"] == "confirmed") or only comes from
          the ephemeris.

    Returns:
        None, or {name: dict of np.ndarray (n_frames,)}:
            t [frames], time (ISO str), r [Rsun], pa [deg], eps [deg], vmag,
            col [px], r_px [px from the bottom row], sig_col, sig_r [px] (core), sig_col_halo,
            sig_r_halo [px] (halo), halo_frac, snr_lin (linear S/N), ceiling (saturation S/N),
            snr = min(snr_lin, ceiling) (peak S/N), saturated (bool), in_view (bool),
            calibrated (bool), punch_confirmed (bool), dr_dt [px/frame], dr_dt_kms [km/s], dpa_dt [deg/frame]
    """
    names = tuple(params.get("syncom_planets", ()) or ())
    if not names:
        return None
    if params.get("syncom_start_time") is None:
        raise ValueError("planets are on but params['syncom_start_time'] is None; "
                         "set default_syncom_prams(SYNCOM_START_TIME='YYYY-MM-DDThh:mm:ss')")
    unknown = [n for n in names if n not in VMAG]
    if unknown:
        raise ValueError(f"unknown planet(s) {unknown} (PLANET_DB: {list(PLANET_DB)})")

    nx, ny = params["syncom_nx"], params["syncom_ny"]
    pixel, cadence = params["syncom_pixel"], params["syncom_cadence"]
    r_min, r_max = polar_rho_range(params)

    steps = np.arange(time0, time_t)
    times = Time(params["syncom_start_time"], scale="utc") + TimeDelta(steps * cadence, format="sec")

    table = {}
    for name in names:
        eph = planet_ephemeris(name, times)
        lin, ceiling, core, halo, frac = punch_snr_shape(eph["vmag"], eph["r"])
        # sigma on the sky [deg] -> radial px (linear in elongation) and column px (around the Sun)
        to_r = lambda fwhm: fwhm * FWHM_TO_SIGMA / eph["theta_sun"] / pixel
        to_col = lambda fwhm: np.degrees(np.radians(fwhm * FWHM_TO_SIGMA) / np.sin(np.radians(eph["eps"]))) * nx / 360.0
        margin = DRAW_SIGMAS * halo * FWHM_TO_SIGMA / eph["theta_sun"]              # [Rsun]
        if not np.any((eph["r"] >= r_min - margin) & (eph["r"] <= r_max + margin)):
            print(f"  {name.capitalize()}: r {eph['r'].min():.1f}-{eph['r'].max():.1f} Rsun, outside the frame "
                  f"({r_min:.0f}-{r_max:.0f} Rsun) for the whole run: not modelled")
            continue

        dr = np.gradient(eph["r"]) if len(steps) > 1 else np.zeros(1)              # [Rsun/frame]
        dpa = np.gradient(np.unwrap(np.radians(eph["pa"]))) if len(steps) > 1 else np.zeros(1)
        table[name] = {
            "t": steps, "time": np.array(times.isot), "r": eph["r"], "pa": eph["pa"], "eps": eph["eps"],
            "vmag": eph["vmag"], "col": solar_pa_to_column(eph["pa"], nx), "r_px": (eph["r"] - r_min) / pixel,
            "sig_col": to_col(core), "sig_r": to_r(core), "sig_col_halo": to_col(halo), "sig_r_halo": to_r(halo),
            "halo_frac": frac, "snr_lin": lin, "ceiling": ceiling, "snr": np.minimum(lin, ceiling),
            "saturated": lin > ceiling,
            "in_view": (eph["r"] >= r_min) & (eph["r"] <= r_max),
            "calibrated": (eph["r"] >= min(PUNCH_CAL["sigma_r"][0], PUNCH_CAL["sat_r"][0]))
                          & (eph["r"] <= max(PUNCH_CAL["sigma_r"][-1], PUNCH_CAL["sat_r"][-1])),
            "punch_confirmed": np.full(len(steps), PLANET_DB[name]["punch"] == "confirmed"),
            "dr_dt": dr / pixel, "dr_dt_kms": dr * RSUN_KM / cadence, "dpa_dt": np.degrees(dpa),
        }
    return table or None


def punch_snr_shape(vmag, r, cal=None):
    """
    Linear S/N, saturation ceiling and unsaturated shape of a point source of magnitude vmag at radius
    r [Rsun] in a PUNCH L3 CAM frame.

    Expected behaviour:
        - Linear S/N = zp_peak * 10 ** (-0.4 * (vmag - zp_vmag)) / sigma(r); ceiling = sat_snr(r);
          sigma(r) and sat_snr(r) are interpolated in r, constant outside the measured range.
        - Shape = psf_unsat with the core FWHM grown by (1 + psf_growth * max(r - psf_r0, 0)).
        - The source is drawn as min(lin * shape, ceiling): its peak S/N is min(lin, ceiling).

    Returns:
        snr_lin, ceiling, core_fwhm [deg], halo_fwhm [deg], halo_frac: np.ndarray like vmag
    """
    cal = PUNCH_CAL if cal is None else cal
    vmag, r = np.broadcast_arrays(np.asarray(vmag, dtype=float), np.asarray(r, dtype=float))
    lin = cal["zp_peak"] * 10.0 ** (-0.4 * (vmag - cal["zp_vmag"])) / np.interp(r, cal["sigma_r"], cal["sigma"])
    ceiling = np.interp(r, cal["sat_r"], cal["sat_snr"])
    core0, halo, frac = cal["psf_unsat"]
    core = core0 * (1.0 + cal["psf_growth"] * np.maximum(r - cal["psf_r0"], 0.0))
    return lin, ceiling, core, np.full(r.shape, halo), np.full(r.shape, frac)


def _pixel_gaussian(n, centre, sigma, periodic=False):
    """
    Pixel-averaged Gaussian of peak 1 (continuous peak) on the pixels within DRAW_SIGMAS sigma of centre.

    Returns (indices, values); indices wrap modulo n if periodic, else are clipped to [0, n).
    """
    lo = int(np.floor(centre - DRAW_SIGMAS * sigma - 1))
    hi = int(np.ceil(centre + DRAW_SIGMAS * sigma + 1))
    j = np.arange(lo, hi + 1)
    s = np.sqrt(2.0) * sigma
    vals = sigma * np.sqrt(np.pi / 2.0) * (erf((j + 0.5 - centre) / s) - erf((j - 0.5 - centre) / s))
    if periodic:
        return j % n, vals
    keep = (j >= 0) & (j < n)
    return j[keep], vals[keep]


def add_planets(img, params, table, time_t, sigma_syn):
    """
    Add every planet of the table to the polar frame img (NX, NY) of time step time_t, in place.

    Expected behaviour:
        - Shape P = (1 - halo_frac) * core + halo_frac * halo: two concentric 2D Gaussians of peak 1,
          centred on (col, r_px) of the table at time_t (widths sig_col/sig_r and sig_col_halo/sig_r_halo).
        - Added image = sigma_syn * min(snr_lin * P, ceiling): below the ceiling the planet is the
          linear profile; above it the core is flat at the ceiling and wider the brighter the planet,
          as a saturating detector shows it. Its peak S/N against the local fluctuation of the frame
          (local_sigma) is the PUNCH S/N, min(snr_lin, ceiling).
        - sigma_syn is one value for all planets or a dict {name: value} (planet_sigmas).
        - Pixel-averaged Gaussians (flux kept even when sigma is ~1 px); columns wrap around 0/360 deg.
        - Drawn when any part is inside the frame; nothing happens if table is None.

    Returns:
        img (the same array)
    """
    if table is None:
        return img
    nx, ny = img.shape
    for name, tr in table.items():
        k = int(np.searchsorted(tr["t"], time_t))
        if k >= len(tr["t"]) or tr["t"][k] != time_t:
            raise IndexError(f"time step {time_t} is not in the planet table ({tr['t'][0]}..{tr['t'][-1]})")
        # one window for both components: that of the halo
        ic, _ = _pixel_gaussian(nx, tr["col"][k], tr["sig_col_halo"][k], periodic=True)
        ir, _ = _pixel_gaussian(ny, tr["r_px"][k], tr["sig_r_halo"][k])
        if len(ir) == 0:
            continue
        lo_c = int(np.floor(tr["col"][k] - DRAW_SIGMAS * tr["sig_col_halo"][k] - 1))
        n_c = len(ic)
        shape = np.zeros((n_c, len(ir)))
        for frac, sc, sr in ((1.0 - tr["halo_frac"][k], tr["sig_col"][k], tr["sig_r"][k]),
                             (tr["halo_frac"][k], tr["sig_col_halo"][k], tr["sig_r_halo"][k])):
            gc = _pixel_gaussian_on(np.arange(lo_c, lo_c + n_c), tr["col"][k], sc)
            gr = _pixel_gaussian_on(ir, tr["r_px"][k], sr)
            shape += frac * np.outer(gc, gr)
        sig = sigma_syn[name] if isinstance(sigma_syn, dict) else sigma_syn
        np.add.at(img, (ic[:, None], ir[None, :]), sig * np.minimum(tr["snr_lin"][k] * shape, tr["ceiling"][k]))
    return img


def _pixel_gaussian_on(j, centre, sigma):
    """Pixel-averaged Gaussian of peak 1 (continuous peak) on the pixel indices j (unwrapped)."""
    s = np.sqrt(2.0) * sigma
    return sigma * np.sqrt(np.pi / 2.0) * (erf((j + 0.5 - centre) / s) - erf((j - 0.5 - centre) / s))


def planet_rows(table, ny, time_t=None):
    """Radial rows around the planets' track (at time_t, or at the first time step), for local_sigma."""
    rows = []
    for tr in table.values():
        k = 0 if time_t is None else int(np.searchsorted(tr["t"], time_t))
        half = max(5, int(np.ceil(3 * tr["sig_r"][k])))
        c = int(round(tr["r_px"][k]))
        rows.append(np.arange(c - half, c + half + 1))
    rows = np.unique(np.concatenate(rows))
    rows = rows[(rows >= 0) & (rows < ny)]
    if len(rows) == 0:                 # planets outside the frame: any rows (the blob statistics do not depend on r)
        rows = np.arange(min(ny, 11))
    return rows


def planet_sigmas(img, table):
    """local_sigma of the frame img at each planet's radius (first time step of its track): {name: value}."""
    return {name: local_sigma(img, planet_rows({name: tr}, img.shape[1])) for name, tr in table.items()}


def local_sigma(img, rows, width_deg=SIGMA_WIDTH_DEG, clip=5.0):
    """
    Local fluctuation of a polar frame on the given radial rows.

    Expected behaviour:
        - For each row: the frame minus its running median along the columns (width width_deg,
          wrapping at 0/360 deg), then the standard deviation over the columns after iterative
          clipping at `clip` sigma (point sources such as stars and planets are left out).
        - Returns the median over the rows. NaN pixels (no data) are ignored.
        - Used in the same way on SynCOM frames and on PUNCH frames unwrapped with unwrap_punch_polar.
          (A median-absolute-deviation estimate would be 0 on noise-free SynCOM frames.)
    """
    from astropy.stats import sigma_clipped_stats
    x = np.asarray(img, dtype=float)[:, np.atleast_1d(rows)]          # (NX, n_rows)
    nx = x.shape[0]
    w = max(3, int(round(width_deg * nx / 360.0)) | 1)                 # odd window [px]
    pad = np.concatenate([x[-(w // 2):], x, x[:w // 2]], axis=0)
    win = np.lib.stride_tricks.sliding_window_view(pad, w, axis=0)    # (NX, n_rows, w)
    hp = x - np.nanmedian(win, axis=-1)
    std = sigma_clipped_stats(hp, sigma=clip, maxiters=10, axis=0)[2]
    return float(np.nanmedian(std))


# ------------------------------------------------------------------ #
# PUNCH frames: location, polar unwrap, calibration                   #
# ------------------------------------------------------------------ #
def planet_pixel_punch(header, name="mercury"):
    """
    Planet in a PUNCH L3 frame, located as in movie_generic.ipynb.

    Expected behaviour:
        - body_direction at DATE-OBS mapped to pixels with the alternate RA/Dec WCS (key 'A'),
          then to helioprojective (lon, lat) with the primary WCS, then to elongation eps and
          position angle pa (from solar north, counterclockwise through east).
        - r = eps / RSUN_ARC.

    Returns:
        dict: x, y [0-based px], eps [deg], pa [deg], r [Rsun], time (Time)
    """
    from astropy.wcs import WCS
    t = Time(header["DATE-OBS"].strip(), scale="utc")
    x, y = WCS(header, key="A").world_to_pixel(body_direction(name, t)[0])
    lon, lat = WCS(header).all_pix2world(float(x), float(y), 0)
    lon, lat = np.radians(lon), np.radians(lat)
    eps = np.degrees(np.arccos(np.cos(lat) * np.cos(lon)))
    pa = np.degrees(np.arctan2(-np.cos(lat) * np.sin(lon), np.sin(lat))) % 360.0
    return {"x": float(x), "y": float(y), "eps": float(eps), "pa": float(pa),
            "r": float(eps * 3600.0 / header["RSUN_ARC"]), "time": t}


def unwrap_punch_polar(fits_path, params, rows=None):
    """
    PUNCH L3 frame resampled onto the SynCOM polar grid (columns x radial rows).

    Expected behaviour:
        - Column j is at solar PA column_to_solar_pa(j), row i at radius POLAR_R_MIN + i * pixel
          [Rsun], with the R_sun unit of the file (RSUN_ARC), so a feature lands on the same (column, row)
          as in a SynCOM frame of the same params.
        - Bilinear interpolation through the primary (helioprojective ARC) WCS; NaN where the file has
          no data (zeros, outside the mosaic).

    Parameters:
        fits_path: PUNCH L3 FITS file (image in extension 1)
        params: SynCOM params (syncom_nx, syncom_ny, syncom_pixel)
        rows: radial rows to compute (default: all NY rows)

    Returns:
        (img, header): img np.ndarray (NX, len(rows)), header of the image extension
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates

    with fits.open(fits_path) as hdul:
        header = hdul[1].header
        data = np.asarray(hdul[1].data, dtype=float)
    nx, ny, pixel = params["syncom_nx"], params["syncom_ny"], params["syncom_pixel"]
    rows = np.arange(ny) if rows is None else np.atleast_1d(rows)

    theta_sun = header["RSUN_ARC"] / 3600.0                              # [deg]
    psi = np.radians(column_to_solar_pa(np.arange(nx), nx))[:, None]
    eps = np.radians((POLAR_R_MIN + rows * pixel) * theta_sun)[None, :]
    lat = np.degrees(np.arcsin(np.sin(eps) * np.cos(psi)))
    lon = np.degrees(np.arctan2(-np.sin(eps) * np.sin(psi), np.cos(eps)))
    px, py = WCS(header).all_world2pix(lon.ravel(), lat.ravel(), 0)

    data = np.where(np.isfinite(data) & (data != 0), data, np.nan)
    img = map_coordinates(data, [py, px], order=1, mode="constant", cval=np.nan)
    return img.reshape(nx, len(rows)), header


def _fit_psf(data, x0, y0, half=8, mask_radius=None):
    """
    2D elliptical Gaussian + plane fit around (x0, y0) of a Cartesian PUNCH image.

    Pixels farther than mask_radius from the peak, and inside the window, are left out if
    mask_radius is given (to exclude a neighbouring source).

    Returns dict: amp (peak above background), x, y, sig1, sig2 [px], fwhm [px] (geometric mean)
    """
    from scipy.optimize import curve_fit
    xi, yi = int(round(x0)), int(round(y0))
    cut = data[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(float)
    norm = np.nanmax(np.abs(cut))                    # fit in units of the maximum (curve_fit is fragile at ~1e-12)
    cut = cut / norm
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    ky, kx = np.unravel_index(np.nanargmax(cut), cut.shape)
    use = np.isfinite(cut)
    if mask_radius is not None:
        use &= np.hypot(xx - xx[ky, kx], yy - yy[ky, kx]) <= mask_radius

    def model(xy, a, xc, yc, s1, s2, th, b0, bx, by):
        x, y = xy
        c, s = np.cos(th), np.sin(th)
        u_, v_ = c * (x - xc) + s * (y - yc), -s * (x - xc) + c * (y - yc)
        return a * np.exp(-0.5 * (u_**2 / s1**2 + v_**2 / s2**2)) + b0 + bx * x + by * y

    p0 = [cut[ky, kx] - np.nanmedian(cut), xx[ky, kx], yy[ky, kx], 1.5, 1.5, 0.0, np.nanmedian(cut), 0.0, 0.0]
    p, _ = curve_fit(model, (xx[use], yy[use]), cut[use], p0=p0, maxfev=40000)
    s1, s2 = abs(p[3]), abs(p[4])
    return {"amp": p[0] * norm, "x": xi + p[1], "y": yi + p[2], "sig1": s1, "sig2": s2,
            "fwhm": np.sqrt(s1 * s2) / FWHM_TO_SIGMA}


def _fit_profile(data, xc, yc, amp, half=30, r_out=16.0, sector_quantile=0.5):
    """
    Core + halo fit of the radial profile of a point source at (xc, yc) of a Cartesian PUNCH image.

    The profile is the sector_quantile (0.5 = median) over 8 sectors of 45 deg (in rings of 0.5 px)
    minus the background (median of the 20-30 px ring), normalized by amp; a streak through the
    source falls in a few sectors and does not change it (use a lower quantile when streaks cover
    up to half of the sectors).

    Returns dict: core_fwhm, halo_fwhm [px], halo_frac, r, profile, model (normalized to the fitted peak)
    """
    from scipy.optimize import curve_fit
    yi, xi = int(round(yc)), int(round(xc))
    cut = data[yi - half:yi + half + 1, xi - half:xi + half + 1] / amp
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    X, Y = xx + xi - xc, yy + yi - yc
    R, ang = np.hypot(X, Y), np.degrees(np.arctan2(Y, X)) % 360.0
    bkg = np.nanmedian(cut[(R > 20) & (R < 30)])
    prof, rc = [], []
    for a in np.arange(0.0, r_out, 0.5):
        ring = (R >= a) & (R < a + 0.5)
        secs = [np.nanmedian(cut[ring & (ang >= s) & (ang < s + 45)]) for s in range(0, 360, 45)
                if (ring & (ang >= s) & (ang < s + 45)).any()]
        if secs:
            prof.append(np.quantile(secs, sector_quantile) - bkg)
            rc.append(np.sqrt(np.mean(R[ring] ** 2)))
    prof, rc = np.array(prof), np.array(rc)

    def model(r, a, sc, h, sh):
        return a * ((1 - h) * np.exp(-0.5 * (r / sc) ** 2) + h * np.exp(-0.5 * (r / sh) ** 2))

    p, _ = curve_fit(model, rc, prof, p0=[1.0, 1.6, 0.03, 4.0], bounds=([0, 0.5, 0, 1], [3, 5, 0.5, 25]))
    return {"core_fwhm": p[1] / FWHM_TO_SIGMA, "halo_fwhm": p[3] / FWHM_TO_SIGMA, "halo_frac": p[2],
            "r": rc, "profile": prof / p[0], "model": model(rc, *p) / p[0]}


def _source_peak(data, x, y, box=4, ring=(20, 30)):
    """Brightest pixel within +-box px of (x, y) minus the background (median of the ring around it) [MSB]."""
    xi, yi = int(round(x)), int(round(y))
    big = data[yi - ring[1]:yi + ring[1] + 1, xi - ring[1]:xi + ring[1] + 1]
    yy, xx = np.mgrid[-ring[1]:ring[1] + 1, -ring[1]:ring[1] + 1]
    rr = np.hypot(xx, yy)
    bkg = np.nanmedian(big[(rr > ring[0]) & (rr < ring[1])])
    return float(np.nanmax(data[yi - box:yi + box + 1, xi - box:xi + box + 1]) - bkg)


def _punch_sizes(data, x, y, sigma, H=60, level=30.0):
    """
    Apparent size of a planet in a Cartesian PUNCH image [deg]: equivalent diameters (2 sqrt(area / pi)) of the
    region connected to the planet above half its peak and above `level` x sigma (background: median of the
    40-60 px ring). PUNCH pixels are 0.0225 deg.
    """
    from scipy import ndimage
    xi, yi = int(round(x)), int(round(y))
    c = np.full((2 * H + 1, 2 * H + 1), np.nan)                       # cutout, NaN beyond the image edge
    y0, y1, x0, x1 = max(yi - H, 0), min(yi + H + 1, data.shape[0]), max(xi - H, 0), min(xi + H + 1, data.shape[1])
    c[y0 - (yi - H):y1 - (yi - H), x0 - (xi - H):x1 - (xi - H)] = data[y0:y1, x0:x1]
    yy, xx = np.mgrid[-H:H + 1, -H:H + 1]
    rr = np.hypot(xx, yy)
    c = np.where(np.isfinite(c), c - np.nanmedian(c[(rr > 40) & (rr < 60)]), 0.0)
    k = np.unravel_index(np.argmax(np.where(rr <= 4, c, -np.inf)), c.shape)

    def diameter(mask):
        lab, _ = ndimage.label(mask)
        return 2.0 * np.sqrt((lab == lab[k]).sum() * 0.0225 ** 2 / np.pi) if lab[k] else 0.0
    return diameter(c > 0.5 * c[k]), diameter(c > level * sigma)


def _clipped_diameters(lin, ceiling, core, halo, frac, level=30.0):
    """Half-maximum and level-sigma diameters [deg] of min(lin * (core + halo profile), ceiling) (radial profile)."""
    x = np.linspace(0.0, 1.5, 30001)
    s = lambda f: f * FWHM_TO_SIGMA
    prof = np.minimum(lin * ((1 - frac) * np.exp(-0.5 * (x / s(core)) ** 2) + frac * np.exp(-0.5 * (x / s(halo)) ** 2)),
                      ceiling) / min(lin, ceiling)
    edge = lambda lev: 2 * x[np.argmax(prof < lev)] if (prof < lev).any() else 2 * x[-1]
    return edge(0.5), edge(level / min(lin, ceiling))


def calibrate_punch(fits_paths, params, zp=("20260718", "mercury"), names=tuple(PLANET_DB),
                    sigma_radii=np.arange(25.0, 176.0, 5.0), over_sat=2.0, verbose=True):
    """
    Measure PUNCH_CAL from PUNCH L3 CAM files that show planets.

    Expected behaviour:
        - Every planet of `names` with data at its position (as located by planet_pixel_punch, the
          movie_generic method) is measured: peak above the background (_source_peak), local_sigma at
          its radius on the frame unwrapped onto the SynCOM grid of params, S/N, V magnitude, apparent
          size (_punch_sizes: half-maximum and 30-sigma diameters), and the offset of the fitted centre
          from the predicted one. Planets outside the frames are skipped.
        - Zero point: the source zp = (text in the file name, planet); local fluctuation vs radius from
          its frame (local_sigma at sigma_radii); unsaturated shape: core + halo fit (_fit_profile) of it.
        - A source whose peak is below its linear prediction by more than over_sat is saturated; the
          saturation ceiling is the S/N of the saturated sources vs radius.
        - psf_growth: growth of the core FWHM with radius beyond the zero-point source, fitted so that the
          drawn shape min(lin * shape, ceiling) has the measured sizes (least squares in log size).

    Returns:
        (cal, sources, profile): cal in the PUNCH_CAL format; sources a list of dicts (one per measured
        planet and file); profile the core + halo fit of the zero-point source
    """
    from pathlib import Path
    from astropy.io import fits
    from scipy.optimize import minimize_scalar

    pixel = params["syncom_pixel"]
    sources, frames = [], {}
    for path in fits_paths:
        with fits.open(path) as hdul:
            header, data = hdul[1].header, np.asarray(hdul[1].data, dtype=float)
        frames[path] = data
        for name in names:
            loc = planet_pixel_punch(header, name)
            if not (np.isfinite(loc["x"]) and np.isfinite(loc["y"])):          # far outside the mosaic
                continue
            xi, yi = int(round(loc["x"])), int(round(loc["y"]))
            if not (40 <= xi < header["NAXIS1"] - 40 and 40 <= yi < header["NAXIS2"] - 40):
                continue
            if not np.isfinite(data[yi, xi]) or data[yi, xi] == 0:            # no data (occulter, outside)
                continue
            c = int(round((loc["r"] - POLAR_R_MIN) / pixel))
            rows = np.arange(c - 20, c + 21)
            polar, _ = unwrap_punch_polar(path, params, rows=rows)
            sig = local_sigma(polar, np.arange(len(rows)))
            peak = _source_peak(data, loc["x"], loc["y"])
            d_half, d_30 = _punch_sizes(data, loc["x"], loc["y"], sig)
            try:
                fit = _fit_psf(data, loc["x"], loc["y"], mask_radius=4.0)
                off = float(np.hypot(fit["x"] - loc["x"], fit["y"] - loc["y"]))
            except RuntimeError:
                off = np.nan
            eph = planet_ephemeris(name, loc["time"])
            sources.append({"file": Path(path).name, "path": path, "planet": name, "time": loc["time"].isot,
                            "r": loc["r"], "pa": loc["pa"], "vmag": float(eph["vmag"][0]),
                            "phase": float(eph["phase"][0]), "peak": peak, "sigma": sig, "snr": peak / sig,
                            "d_half": d_half, "d_30": d_30, "fit_offset": off, "x": loc["x"], "y": loc["y"]})

    zp_src = [s_ for s_ in sources if zp[0] in s_["file"] and s_["planet"] == zp[1]][0]
    for s_ in sources:
        s_["lin_peak"] = zp_src["peak"] * 10.0 ** (-0.4 * (s_["vmag"] - zp_src["vmag"]))
        s_["over"] = s_["lin_peak"] / s_["peak"]
        s_["saturated"] = s_["over"] > over_sat

    sig_prof = []
    for r in sigma_radii:
        c = int(round((r - POLAR_R_MIN) / pixel))
        polar, _ = unwrap_punch_polar(zp_src["path"], params, rows=np.arange(c - 2, c + 3))
        sig_prof.append(local_sigma(polar, np.arange(5)))
    sig_prof = np.array(sig_prof)
    ok = np.isfinite(sig_prof)

    data = frames[zp_src["path"]]
    fit = _fit_psf(data, zp_src["x"], zp_src["y"], mask_radius=4.0)
    profile = _fit_profile(data, fit["x"], fit["y"], fit["amp"])
    psf_unsat = [float(profile["core_fwhm"] * 0.0225), float(profile["halo_fwhm"] * 0.0225), float(profile["halo_frac"])]

    sat = sorted([s_ for s_ in sources if s_["saturated"]], key=lambda s_: s_["r"])
    cal = {"zp_peak": zp_src["peak"], "zp_vmag": zp_src["vmag"],
           "sigma_r": [float(v) for v in sigma_radii[ok]], "sigma": [float(v) for v in sig_prof[ok]],
           "sat_r": [float(s_["r"]) for s_ in sat], "sat_snr": [float(s_["snr"]) for s_ in sat],
           "psf_unsat": psf_unsat, "psf_growth": 0.0, "psf_r0": float(zp_src["r"]),
           "source": ", ".join(sorted({Path(p).name for p in fits_paths}))}

    def size_error(growth):
        cal["psf_growth"] = growth
        err = []
        for s_ in sources:
            lin, ceil, core, halo, frac = punch_snr_shape(s_["vmag"], s_["r"], cal)
            dh, d30 = _clipped_diameters(float(lin), float(ceil), float(core), float(halo), float(frac))
            err += [np.log(dh / s_["d_half"]), np.log(d30 / s_["d_30"])]
        return float(np.sqrt(np.mean(np.square(err))))
    best = minimize_scalar(size_error, bounds=(0.0, 0.02), method="bounded", options={"xatol": 1e-6})
    cal["psf_growth"] = float(best.x)
    for s_ in sources:
        lin, ceil, core, halo, frac = punch_snr_shape(s_["vmag"], s_["r"], cal)
        s_["model_d_half"], s_["model_d_30"] = _clipped_diameters(float(lin), float(ceil), float(core), float(halo), float(frac))
        s_["model_snr"] = float(min(lin, ceil))

    if verbose:
        print(f"zero point: {zp_src['planet']} {zp_src['time'][:16]}  peak {cal['zp_peak']:.3e} MSB at V {cal['zp_vmag']:.2f}")
        print("unsaturated shape: core %.4f deg, halo %.4f deg at %.1f%%;  core growth %.5f per Rsun beyond %.1f Rsun"
              % (*psf_unsat[:2], 100 * psf_unsat[2], cal["psf_growth"], cal["psf_r0"]))
        print(f"size model rms error {np.expm1(best.fun) * 100:.0f}% (log)")
        print("date        planet    r [Rsun]  V      over   S/N meas/model    half-max diam meas/model [deg]  30-sigma diam meas/model [deg]  fit offset")
        for s_ in sources:
            print(f"{s_['time'][:10]}  {s_['planet']:8s} {s_['r']:6.1f}  {s_['vmag']:+.2f}  {s_['over']:6.1f}  "
                  f"{s_['snr']:6.0f} / {s_['model_snr']:6.0f}    {s_['d_half']:.3f} / {s_['model_d_half']:.3f}"
                  f"                 {s_['d_30']:.3f} / {s_['model_d_30']:.3f}                 {s_['fit_offset']:.2f} px")
    return cal, sources, profile
