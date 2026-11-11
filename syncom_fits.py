"""
FITS input/output for SynCOM cubes.

syncom_core only builds the cube in memory (and optionally saves it as .npz).
FITS frames are written separately, and only for the frames you ask for:

    write_fits_frames(cube, run_name, frames=range(0, 30))   # first 30 time steps
    cube = load_fits_cube(fits_dir, run_name, 0, 30, nx, ny)  # read them back

File layout (unchanged from earlier versions):
    SynCOM_runs/<run_name>/FITS/<run_name>_<t:04d>.fts
    one 2D image per file, shape (NY, NX) = (radius, position angle), float64,
    no compression.
"""
from pathlib import Path

import numpy as np
from astropy.io import fits

from syncom_header import syncom_fits_header
from syncom_path import syncom_output_path


def write_fits_frames(cube, run_name, frames=None, time0=0, params=None, overwrite=True,
                      start_time=None):
    """
    Write chosen frames of a SynCOM cube as one FITS file per time step.

    Parameters:
        cube (np.ndarray): cube [NX, NY, NT] from syncom_core
        run_name (str): run label (syncom_path.syncom_run_name); used for the folder
            and the file names
        frames (iterable of int): time steps to write, e.g. range(0, 30) or [0, 50, 100].
            None writes every frame of the cube.
        time0 (int): time step of cube[:, :, 0] (the `time0` given to syncom_core).
            Time step t is cube index t - time0.
        params (dict): optional; if given, the WCS and the model parameters are stored
            in each FITS header (see syncom_header.syncom_fits_header)
        overwrite (bool): replace existing files with the same name
        start_time (astropy.time.Time or str): optional; UTC time of time step 0,
            adds DATE-OBS to each header (needs params)

    Returns:
        list of Path: the files written, in the order of `frames`

    Expected behaviour:
        - File for time step t: SynCOM_runs/<run_name>/FITS/<run_name>_<t:04d>.fts
        - Image stored as cube[:, :, t - time0].T, shape (NY, NX), float64, uncompressed,
          so it reads back bit for bit (load_fits_cube, astropy, IDL readfits).
        - Raises IndexError for a time step outside the cube.
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D [NX, NY, NT], got shape {cube.shape}")
    nt = cube.shape[2]

    # Default: every time step in the cube
    steps = range(time0, time0 + nt) if frames is None else [int(t) for t in frames]

    written = []
    for t in steps:
        k = t - time0                                   # index of time step t in the cube
        if not 0 <= k < nt:
            raise IndexError(f"time step {t} is outside the cube (time steps {time0}..{time0 + nt - 1})")

        # Header: what the file is, so it can be identified without the notebook (syncom_header.py)
        header = syncom_fits_header(run_name, t, params=params, start_time=start_time,
                                    nx=cube.shape[0], ny=cube.shape[1])

        # .T -> (NY, NX) = (radius, position angle), the layout FlowTrack expects
        filename = syncom_output_path(run_name, "fits", f"{t:04d}.fts")
        fits.writeto(filename, np.asarray(cube[:, :, k], dtype=np.float64).T,
                     header=header, overwrite=overwrite)
        written.append(Path(filename))

    print(f"✅ Wrote {len(written)} FITS files to {Path(written[0]).parent.resolve()}" if written
          else "No FITS files written (empty frame list)")
    return written


def load_fits_cube(fits_dir, run_name, time0, time_t, nx, ny):
    """
    Builds the [NX, NY, NT] cube from <run_name>_<t:04d>.fts in fits_dir for t = time0 ... time_t-1.
    Returns None if any frame is missing.
    """
    files = [Path(fits_dir) / f"{run_name}_{t:04d}.fts" for t in range(time0, time_t)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"{len(missing)} of {len(files)} FITS files missing (e.g. {missing[0]}) - running the simulation instead")
        return None

    cube = np.empty((nx, ny, len(files)), dtype=float)
    for k, f in enumerate(files):
        img = fits.getdata(f)
        if img.shape == (ny, nx):          # IDL-layout file -> transpose to [NX, NY]
            img = img.T
        cube[:, :, k] = img
    print(f"✅ Built cube {cube.shape} from {len(files)} FITS files in {Path(fits_dir).resolve()}")

    return cube
