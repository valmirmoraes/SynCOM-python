import numpy as np
from pathlib import Path

# Shared constants (imported by polar_to_rect.py and syncom_movie.py)
SYNCOM_CADENCE = 300.0             # [s]    time between cube frames
POLAR_PHI_RANGE = (0.0, 359.9)     # [deg]  PA range of a polar frame      (syncomnoise.pro)
POLAR_RHO_RANGE = (5.0, 14.226)    # [Rsun] radial range of a polar frame  (syncomnoise.pro)


def default_syncom_prams(
    SYNCOM_N_BLOBS=1000,
    SYNCOM_CADENCE=SYNCOM_CADENCE,
    SYNCOM_PIXEL=0.0141875720451562,   # radial pixel size [Rsun], as in data_sample_1
    SYNCOM_NX=3600,
    SYNCOM_NY=659,
    SYNCOM_ACC=0,
    SYNCOM_SCALE_FACTOR=2.0,
    SYNCOM_INITIAL_R_SUN=5.0,
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