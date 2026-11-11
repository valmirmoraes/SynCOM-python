"""
FITS headers for SynCOM frames.

    header = syncom_fits_header(run_name, t, params=params, start_time=START_TIME)
    fits.writeto(filename, image, header=header)

    # add/refresh headers of frames already on disk (image data is not touched)
    update_fits_headers(run_name, params=params, start_time=START_TIME)

Every frame is one 2D image, shape (NY, NX) = (radius, position angle).
The header identifies the run and time step, gives a linear WCS for both axes
(position angle in deg, radius in solar radii) and stores the model parameters,
so a file can be interpreted without the notebook.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from syncom_path import syncom_output_path, syncom_run_dir
from syncom_prams import POLAR_PHI_RANGE, polar_rho_range

CREATOR = "SynCOM_python"

# params key -> (FITS keyword, comment). Keywords are at most 8 characters.
PARAM_CARDS = {
    "syncom_cadence":       ("CADENCE",  "[s] time between frames"),
    "syncom_pixel":         ("PIXEL",    "[Rsun] radial pixel size"),
    "syncom_n_blobs":       ("NBLOBS",   "number of blobs"),
    "syncom_scale_factor":  ("SCALEFAC", "blob size scale factor"),
    "syncom_initial_R_sun": ("R_INIT",   "[Rsun] radius where blob centres enter"),
    "syncom_acc":           ("ACC",      "acceleration flag"),
    "syncom_noise_level":   ("NOISELEV", "noise level"),
    "syncom_max_intensity": ("MAXINT",   "maximum blob intensity"),
    "syncomfile":           ("SYNCFILE", "SynCOM input file"),
}


def _to_header_value(value):
    """numpy scalars -> plain Python values (astropy cannot store every numpy type)."""
    return value.item() if isinstance(value, np.generic) else value


def syncom_fits_header(run_name, t, params=None, start_time=None, nx=None, ny=None, extra=None):
    """
    Build the FITS header of SynCOM time step t.

    Parameters:
        run_name (str): run label (syncom_path.syncom_run_name)
        t (int): time step of the frame
        params (dict): optional; run parameters (syncom_prams.default_syncom_prams).
            Without them only RUNNAME, TSTEP, the axis names and the creation date are written.
        start_time (astropy.time.Time or str): optional; UTC time of time step 0.
            Needs params (for the cadence). Adds DATE-OBS = start_time + t * cadence.
        nx, ny (int): image size; default params["syncom_nx"], params["syncom_ny"]
        extra (dict): optional extra cards {KEYWORD: value} or {KEYWORD: (value, comment)}

    Returns:
        astropy.io.fits.Header

    Expected behaviour:
        - Always: RUNNAME, TSTEP, AXIS1/AXIS2 names (as in earlier files), CREATOR, DATE (UTC
          time the header was made), and a COMMENT that the intensity is in arbitrary units.
        - With params:
            * TIME = t * cadence [s]
            * linear WCS, FITS pixel 1 = array index 0:
                axis 1 (position angle): CTYPE1 'PA', CUNIT1 'deg', CRVAL1 = POLAR_PHI_RANGE[0],
                    CDELT1 = (POLAR_PHI_RANGE[1] - POLAR_PHI_RANGE[0]) / (NX - 1)
                    (0.1 deg for NX = 3600, the same grid polar_to_rect uses)
                axis 2 (radius): CTYPE2 'RADIUS', CUNIT2 'solRad', CRVAL2 = r_min of
                    polar_rho_range(params), CDELT2 = params["syncom_pixel"]
            * one card per entry of PARAM_CARDS found in params
        - With start_time and params: DATE-OBS in ISO format (UTC).
        - `extra` cards are added last and can override any card above.
        - The header does not contain SIMPLE/BITPIX/NAXIS; fits.writeto adds them from the data.
    """
    t = int(t)
    header = fits.Header()

    # Identification
    header["RUNNAME"] = run_name                       # no comment: long names fill the card
    header["TSTEP"] = (t, "time step [frames]")
    header["CREATOR"] = (CREATOR, "software that made this file")
    header["DATE"] = (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                      "UTC time the file was written")

    if params is not None:
        cadence = float(params["syncom_cadence"])
        header["TIME"] = (t * cadence, "[s] time since time step 0")
        if start_time is not None:
            date_obs = Time(start_time, scale="utc") + t * cadence / 86400.0
            header["DATE-OBS"] = (date_obs.isot, "UTC time of this frame")

    # Axes: names kept from earlier files, plus a linear WCS when params are known
    header["AXIS1"] = ("position angle", "0.1 deg per pixel")
    header["AXIS2"] = ("radius", "pixel 0 = inner edge of the field")

    if params is not None:
        nx = int(nx if nx is not None else params["syncom_nx"])
        r_min, _ = polar_rho_range(params)
        header["CTYPE1"] = ("PA", "position angle")
        header["CUNIT1"] = ("deg", "")
        header["CRPIX1"] = (1.0, "reference pixel (first column)")
        header["CRVAL1"] = (float(POLAR_PHI_RANGE[0]), "[deg] position angle of CRPIX1")
        header["CDELT1"] = ((POLAR_PHI_RANGE[1] - POLAR_PHI_RANGE[0]) / (nx - 1), "[deg] per pixel")
        header["CTYPE2"] = ("RADIUS", "distance from Sun centre")
        header["CUNIT2"] = ("solRad", "")
        header["CRPIX2"] = (1.0, "reference pixel (first row)")
        header["CRVAL2"] = (float(r_min), "[Rsun] radius of CRPIX2")
        header["CDELT2"] = (float(params["syncom_pixel"]), "[Rsun] per pixel")

        # Model parameters
        for key, (keyword, comment) in PARAM_CARDS.items():
            if key in params and params[key] is not None:
                header[keyword] = (_to_header_value(params[key]), comment)

    header["COMMENT"] = "SynCOM synthetic image; intensity in arbitrary units."
    header["COMMENT"] = "Image shape (NY, NX) = (radius, position angle)."

    if extra:
        for keyword, value in extra.items():
            header[keyword] = value

    return header


def write_syncom_header(filename, run_name, t, params=None, start_time=None, extra=None):
    """
    Replace the header of one existing SynCOM FITS file; the image data is not changed.

    Expected behaviour:
        - Opens the file in update mode, sets every card of syncom_fits_header(...) on the
          primary HDU (existing cards with the same keyword are overwritten, others are kept).
        - NX/NY are taken from the image in the file.
        - The data stays bit for bit the same (float64, uncompressed).
    """
    with fits.open(filename, mode="update") as hdul:
        hdu = hdul[0]
        ny, nx = hdu.data.shape
        new = syncom_fits_header(run_name, t, params=params, start_time=start_time,
                                 nx=nx, ny=ny, extra=extra)
        for card in new.cards:
            if card.keyword == "COMMENT":
                if str(card.value) not in [str(c) for c in hdu.header.get("COMMENT", [])]:
                    hdu.header.add_comment(card.value)
            else:
                hdu.header[card.keyword] = (card.value, card.comment)
    return Path(filename)


def update_fits_headers(run_name, params=None, start_time=None, extra=None):
    """
    Write headers into every FITS frame of a run: SynCOM_runs/<run_name>/FITS/<run_name>_<t:04d>.fts

    Parameters:
        run_name (str): run label
        params (dict): run parameters; None loads SynCOM_runs/<run_name>/<run_name>_params.npy
            if it exists (otherwise headers are written without the parameter cards)
        start_time, extra: as in syncom_fits_header

    Returns:
        list of Path: the files updated, sorted by time step

    Expected behaviour:
        - The time step of each file is read from its name (last 4 digits).
        - Image data is not touched.
    """
    if params is None:
        params_file = syncom_output_path(run_name, "run", "params.npy")
        if params_file.exists():
            params = np.load(params_file, allow_pickle=True).item()
            print(f"Loaded parameters from {params_file}")
        else:
            print(f"No params file ({params_file.name}); headers written without parameter cards")

    fits_dir = syncom_run_dir(run_name, "fits")
    files = sorted(fits_dir.glob(f"{run_name}_[0-9][0-9][0-9][0-9].fts"))
    updated = []
    for f in files:
        t = int(f.stem[-4:])
        updated.append(write_syncom_header(f, run_name, t, params=params,
                                           start_time=start_time, extra=extra))

    print(f"✅ Updated headers of {len(updated)} FITS files in {fits_dir.resolve()}")
    return updated
