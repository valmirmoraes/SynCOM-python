import numpy as np
from pathlib import Path

# Shared constants (imported by polar_to_rect.py, syncom_load.py and syncom_movie.py)
SYNCOM_CADENCE = 300.0               # [s]      time between cube frames
SYNCOM_PIXEL = 0.0141875720451562    # [Rsun]   radial pixel size, as in data_sample_1 (COR2-like polar remap)
SYNCOM_NX = 3600                     # [px]     position-angle pixels of a polar frame (0.1 deg each)
SYNCOM_NY = 659                      # [px]     radial pixels of a polar frame
POLAR_R_MIN = 5.0                    # [Rsun]   radius of the bottom row of every polar frame
POLAR_PHI_RANGE = (0.0, 360.0 * (SYNCOM_NX - 1) / SYNCOM_NX)            # [deg]  = (0.0, 359.9)  (syncomnoise.pro)
POLAR_RHO_RANGE = (POLAR_R_MIN, POLAR_R_MIN + (SYNCOM_NY - 1) * SYNCOM_PIXEL)   # [Rsun] = (5.0, 14.3354)
# POLAR_PHI_RANGE / POLAR_RHO_RANGE are the ranges of the DEFAULT grid only.
# For a run, take the ranges from its params:  phi_range, rho_range = polar_ranges(params)
#   phi_range = (0, 360 (NX - 1) / NX) deg: column j is at j * 360 / NX deg
#   rho_range = (5, 5 + (NY - 1) * pixel) Rsun: row i is at 5 + i * pixel Rsun
# Plots, movies, polar_to_rect and the FITS headers all take params and call polar_ranges,
# so they always show the radii and angles the simulation used.
# POLAR_R_MIN is also where blobs are launched from: each blob centre starts 6 radial widths (12L px)
#   below it, behind the occulter (syncom_load.syncom_blob_table:
#   r0 = (SYNCOM_INITIAL_R_SUN - POLAR_R_MIN) / pixel - 12L), so every blob centre crosses row 0,
#   fully formed, once per period.


def polar_phi_range(params):
    """
    Position angle (first column, last column) [deg] of the polar frames of a run.

    Expected behaviour:
        (0, 360 (NX - 1) / NX): (0, 359.9) for NX = 3600, (0, 359.8) for NX = 1800.
    """
    nx = params["syncom_nx"]
    return (0.0, 360.0 * (nx - 1) / nx)


def polar_rho_range(params):
    """
    Radius (bottom row, top row) [Rsun] of the polar frames of a run.

    Expected behaviour:
        (5, 5 + (NY - 1) * pixel), so the plotted radii match the radii used by the simulation
        for any params: (5, 14.34) for the defaults, (5, 180) for NY 7779 and pixel 0.0225.
    """
    return (POLAR_R_MIN, POLAR_R_MIN + (params["syncom_ny"] - 1) * params["syncom_pixel"])


def polar_ranges(params=None, shape=None):
    """
    (phi_range, rho_range) of the polar frames of a run: the one place plots and movies get their axes.

    Parameters:
        params (dict): run parameters (default_syncom_prams); None -> the default grid
        shape (tuple): optional; shape of the frame or cube being plotted, (NX, NY[, NT])

    Expected behaviour:
        - Returns (polar_phi_range(params), polar_rho_range(params)).
        - If shape is given and its (NX, NY) differ from the params (or, without params, from the
          default grid), raises ValueError asking for the run's params, so a frame is never drawn
          with the radii of another grid.
    """
    if params is None:
        params = {"syncom_nx": SYNCOM_NX, "syncom_ny": SYNCOM_NY, "syncom_pixel": SYNCOM_PIXEL}
        where, fix = "the default grid has", "; pass params=<the params the run was made with>"
    else:
        where, fix = "params have", ""
    if shape is not None and tuple(shape[:2]) != (params["syncom_nx"], params["syncom_ny"]):
        raise ValueError(f"frames have (NX, NY) = {tuple(shape[:2])} but {where} "
                         f"({params['syncom_nx']}, {params['syncom_ny']}){fix}")
    return polar_phi_range(params), polar_rho_range(params)


def default_syncom_prams(
    SYNCOM_N_BLOBS=1000,
    SYNCOM_CADENCE=SYNCOM_CADENCE,
    SYNCOM_PIXEL=SYNCOM_PIXEL,         # radial pixel size [Rsun]; sets the field of view (see polar_rho_range)
    SYNCOM_NX=SYNCOM_NX,
    SYNCOM_NY=SYNCOM_NY,
    SYNCOM_ACC=0,
    SYNCOM_SCALE_FACTOR=2.0,
    SYNCOM_INITIAL_R_SUN=5.0,          # radius where blob centres enter the image [Rsun]; 5.0 = bottom of the polar image
    SYNCOM_NOISE_LEVEL=0.0,
    SYNCOM_MAX_INTENSITY=1.0e-14,
    SYNCOM_TIME=0.0,
    SYNCOM_PLANETS=(),                 # planets drawn into the frames (syncom_planets.py): ("mercury",) = on, () = off
    SYNCOM_START_TIME=None,            # UTC time of time step 0, e.g. "2026-07-26T06:40:00" (needed when planets are on)
    SYNCOMFILE=None,
    DATE=None,
    SAVEPRAMS=None,
    WORKING_DIR=None
):
    from pathlib import Path
    default_file = "syncom_20140414_densfix.sav"
    if SYNCOMFILE:
        syncomfile = SYNCOMFILE
    elif DATE:
        syncomfile = f"syncom_{DATE.replace('-', '')}_densfix.sav"
    else:
        print("No file or date entered. Using default:", default_file)
        syncomfile = default_file

    now = DATE if DATE else "2014-04-14"
    exlab = f"observer date={now}"
    label = f"SynCOM model, cube={now}\n{exlab}"
    name = "syncom"
    
    pram_dict = {
        "Name": name,
        "Label": label,
        "filename": str(Path.home() / "stereo_analysis.sav"),
        "syncom_cadence": SYNCOM_CADENCE,
        "syncom_pixel": SYNCOM_PIXEL,
        "syncom_n_blobs": SYNCOM_N_BLOBS,
        "syncom_acc": SYNCOM_ACC,
        "syncom_time": SYNCOM_TIME,
        "syncom_nx": SYNCOM_NX,
        "syncom_ny": SYNCOM_NY,
        "syncom_initial_R_sun": SYNCOM_INITIAL_R_SUN,
        "syncom_scale_factor": SYNCOM_SCALE_FACTOR,
        "syncom_noise_level": SYNCOM_NOISE_LEVEL,
        "syncom_max_intensity": SYNCOM_MAX_INTENSITY,
        "syncom_planets": tuple(SYNCOM_PLANETS),
        "syncom_start_time": None if SYNCOM_START_TIME is None else str(SYNCOM_START_TIME),
        "syncomfile": syncomfile
    }

    if SAVEPRAMS:
        save_path = Path(SAVEPRAMS)
        if WORKING_DIR:
            save_path = Path(WORKING_DIR) / SAVEPRAMS
        np.save(save_path, pram_dict)
        print(f"Saved SYNCOM parameters to: {save_path}")
    
    return pram_dict