import time
from pathlib import Path
import numpy as np
from astropy.io import fits

from syncom_path import syncom_run_name, syncom_output_path
from syncom_load import syncom_load
from syncom_image import syncom_image

def syncom_core(
    N_BLOBS,
    time0,
    time_t,
    syncom_version="v1",
    scale_factor=None,
    simple=False,
    save=False,
    params=None,
    LoadStruc=None
):
    """
    Simulates the solar wind outflow using SynCOM and produces a time-dependent image cube.

    Parameters:
        N_BLOBS (int): Number of blobs to simulate
        time0 (int): Starting time step
        time_t (int): Ending time step (exclusive)
        syncom_version (str): Optional label/version for saving
        scale_factor (float): Blob size scaling factor
        simple (bool): Use sinusoidal velocity profile if True
        save (bool): Save output as compressed npz
        params (dict): Model parameters
        LoadStruc (dict): Precomputed blob data

    Returns:
        syncom_data (np.ndarray): 3D synthetic image cube [NX, NY, time_t - time0]
    """
    start_time = time.time()

    # 1. Validate and extract parameters
    if params is None:
        raise ValueError("params dictionary must be provided.")

    nx = params["syncom_nx"]
    ny = params["syncom_ny"]
    cadence = params["syncom_cadence"]
    pixel_size = params["syncom_pixel"]
    scalefactor = scale_factor if scale_factor is not None else params["syncom_scale_factor"]
    params["syncom_scale_factor"] = scalefactor

    # 2. Load or use provided blob structure
    if LoadStruc is None:
        LoadStruc = syncom_load(params, simple=simple)

    outarray = LoadStruc["outarray"]

    # 3. Unit conversions and property extraction
    Rsun2Km = 696000.0
    spatial_scale = Rsun2Km * pixel_size
    n_blobs = N_BLOBS

    rng = np.random.default_rng(22)
    time_0 = rng.uniform(1.0, 201.0, n_blobs)
    radial_i = outarray[:, 0] / pixel_size
    period_array = outarray[:, 4]
    v_array = (outarray[:, 5] * cadence) / spatial_scale  # pixels/frame
    L_array = scalefactor * outarray[:, 6] / 0.1
    PSI = outarray[:, 7]

    # 4. Prepare output cube
    nt = time_t - time0
    syncom_data = np.zeros((nx, ny, nt), dtype=float)

    # 5. Run simulation loop
    for t in range(time0, time_t):
        img = syncom_image(params, syncom_version, scalefactor, time_0, radial_i,
                   v_array, period_array, L_array, PSI, t)
        syncom_data[:, :, t - time0] = img

    # 6. Optionally save output
    if save:
        run_name = syncom_run_name(syncom_version, params["syncom_n_blobs"], scalefactor)
        output_name = syncom_output_path(run_name, "cubes", "cube.npz")
        np.savez_compressed(
            output_name,
            syncom_data=syncom_data,
            params=params,
            LoadStruc=LoadStruc,
            scale_factor=scalefactor
        )
        print(f"✅ Saved: {output_name}")

    print(f"⏱️ Total runtime: {time.time() - start_time} seconds")
    return syncom_data


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