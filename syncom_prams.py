import numpy as np
from pathlib import Path

# Shared constants (imported by polar_to_rect.py, syncom_load.py and syncom_movie.py)
SYNCOM_CADENCE = 300.0               # [s]      time between cube frames
SYNCOM_PIXEL = 0.0141875720451562    # [Rsun]   radial pixel size, as in data_sample_1 (COR2-like polar remap)
SYNCOM_NY = 659                      # [px]     radial pixels of a polar frame
POLAR_PHI_RANGE = (0.0, 359.9)       # [deg]    PA range of a polar frame      (syncomnoise.pro)
POLAR_RHO_RANGE = (5.0, 5.0 + (SYNCOM_NY - 1) * SYNCOM_PIXEL)   # [Rsun] radial range = (5.0, 14.3354)
# POLAR_RHO_RANGE[0] is the radius of the bottom row of the polar image (radial pixel 0).
#   Blob centres enter the image at SYNCOM_INITIAL_R_SUN, measured from this edge; each blob is
#   launched 6 radial widths (12L px) below that, behind the occulter (syncom_load.syncom_blob_table:
#   r0 = (SYNCOM_INITIAL_R_SUN - POLAR_RHO_RANGE[0]) / SYNCOM_PIXEL - 12L), so with both at 5.0
#   every blob centre crosses pixel 0, fully formed, once per period.
# POLAR_RHO_RANGE[1] is the radius of the top row (pixel SYNCOM_NY - 1). It is computed from
#   SYNCOM_PIXEL, so plots, movies and the simulation always use the same pixel size.
#   To change the field of view, change SYNCOM_PIXEL (or SYNCOM_NY) here. For a run with other
#   values passed to default_syncom_prams, use polar_rho_range(params) for the plots.


def polar_rho_range(params):
    """
    Radial range (r_min, r_max) [Rsun] of the polar frames of a run, for polar_to_rect and the movies.

    Expected behaviour:
        r_min = POLAR_RHO_RANGE[0] (5 Rsun, the bottom row) and r_max = r_min + (NY - 1) * pixel,
        so the plotted radii match the radii used by the simulation for any params.
    """
    r_min = POLAR_RHO_RANGE[0]
    return (r_min, r_min + (params["syncom_ny"] - 1) * params["syncom_pixel"])


def default_syncom_prams(
    SYNCOM_N_BLOBS=1000,
    SYNCOM_CADENCE=SYNCOM_CADENCE,
    SYNCOM_PIXEL=SYNCOM_PIXEL,         # radial pixel size [Rsun]; sets the field of view (see POLAR_RHO_RANGE)
    SYNCOM_NX=3600,
    SYNCOM_NY=SYNCOM_NY,
    SYNCOM_ACC=0,
    SYNCOM_SCALE_FACTOR=2.0,
    SYNCOM_INITIAL_R_SUN=5.0,          # radius where blob centres enter the image [Rsun]; 5.0 = bottom of the polar image
    SYNCOM_NOISE_LEVEL=0.0,
    SYNCOM_MAX_INTENSITY=1.0e-14,
    SYNCOM_TIME=0.0,
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
        "syncomfile": syncomfile
    }

    if SAVEPRAMS:
        save_path = Path(SAVEPRAMS)
        if WORKING_DIR:
            save_path = Path(WORKING_DIR) / SAVEPRAMS
        np.save(save_path, pram_dict)
        print(f"Saved SYNCOM parameters to: {save_path}")
    
    return pram_dict