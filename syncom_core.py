import time
import numpy as np

from syncom_path import syncom_run_name, syncom_output_path
from syncom_load import syncom_load, syncom_blob_table, syncom_planet_table, add_planets, planet_sigmas
from syncom_image import syncom_image, angular_profiles
from syncom_fits import load_fits_cube, write_fits_frames   # re-exported: `from syncom_core import load_fits_cube` still works


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

    Expected behaviour:
        - Builds the blob table once (syncom_blob_table) and the angular profiles once
          (angular_profiles), then calls syncom_image for t = time0 ... time_t-1.
        - Every blob is launched once per period, behind the occulter, before and during
          the whole run, so every frame is equally full and the mean brightness is the
          same at every radius.
        - No FITS files are written; use syncom_fits.write_fits_frames(cube, run_name,
          frames=...) for the frames you want.
        - save=True writes SynCOM_runs/<run>/cubes/<run>_cube.npz (float64, as before).
        - Progress is printed at 0%, 25%, 50% and 75% of the frames, and when done
          (frames done and elapsed seconds), each once.
        - Planets (params["syncom_planets"], e.g. ("mercury",)): their track is computed once
          (syncom_planet_table) and they are added to every frame (add_planets), with a peak of
          S/N x the local fluctuation of the first, blob-only frame at the planet's radius. With params["syncom_planets"] = ()
          (the default) nothing changes. A summary of each planet's track is printed.

    Parameters:
        N_BLOBS (int): Number of blobs to simulate (must equal params['syncom_n_blobs'])
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
    if N_BLOBS != params["syncom_n_blobs"]:
        raise ValueError(f"N_BLOBS = {N_BLOBS} but params['syncom_n_blobs'] = {params['syncom_n_blobs']}; "
                         "create params with SYNCOM_N_BLOBS=N_BLOBS")

    nx = params["syncom_nx"]
    ny = params["syncom_ny"]
    scalefactor = scale_factor if scale_factor is not None else params["syncom_scale_factor"]
    params["syncom_scale_factor"] = scalefactor

    # 2. Load or use provided blob structure
    if LoadStruc is None:
        LoadStruc = syncom_load(params, simple=simple)

    # 3. Everything that does not change with time, computed once:
    #    blob table in pixels/frames, and the angular profile of every blob
    blobs = syncom_blob_table(params, LoadStruc, scalefactor)
    gx = angular_profiles(params, blobs)
    planets = syncom_planet_table(params, time0, time_t)     # None when planets are off
    if planets is not None:
        _print_planet_summary(planets, params)
    sigma_syn = None

    # 4. Prepare output cube
    nt = time_t - time0
    syncom_data = np.zeros((nx, ny, nt), dtype=float)

    # 5. Run simulation loop, printing progress at 0, 25, 50, 75 % and when done
    print(f"  0% (0/{nt} frames)")
    quarter = 1                                  # next milestone: quarter * 25 %
    for t in range(time0, time_t):
        img = syncom_image(params, blobs, t, gx=gx)
        if planets is not None:
            if sigma_syn is None:      # local fluctuation of the first, blob-only frame at each planet's radius
                sigma_syn = planet_sigmas(img, planets)
            add_planets(img, params, planets, t, sigma_syn)
        syncom_data[:, :, t - time0] = img
        done = t - time0 + 1
        while quarter < 4 and done * 4 >= quarter * nt:
            print(f"  {quarter * 25}% ({done}/{nt} frames, {time.time() - start_time:.0f} s)")
            quarter += 1
    print(f"  done ({nt} frames, {time.time() - start_time:.0f} s)")

    # 6. Optionally save output
    if save:
        run_name = syncom_run_name(syncom_version, params["syncom_n_blobs"], scalefactor)
        output_name = syncom_output_path(run_name, "cubes", "cube.npz")
        np.savez_compressed(
            output_name,
            syncom_data=syncom_data,
            params=params,
            LoadStruc=LoadStruc,
            scale_factor=scalefactor,
            planets=planets,
            sigma_syn=sigma_syn
        )
        print(f"✅ Saved: {output_name}")

    print(f"⏱️ Total runtime: {time.time() - start_time} seconds")
    return syncom_data


def _print_planet_summary(planets, params):
    """One line per planet: radius, position angle, speed, S/N and frames in view over the run."""
    r_max = params["syncom_ny"] - 1
    for name, tr in planets.items():
        n_in = int(tr["in_view"].sum())
        print(f"  {name.capitalize()}: r {tr['r'][0]:.2f} -> {tr['r'][-1]:.2f} Rsun, "
              f"PA {tr['pa'][0]:.2f} -> {tr['pa'][-1]:.2f} deg, "
              f"dr/dt {tr['dr_dt'].mean():.3f} px/frame ({tr['dr_dt_kms'].mean():.1f} km/s), "
              f"V {tr['vmag'].mean():+.2f}, S/N {tr['snr'].mean():.0f}{' (saturated)' if tr['saturated'].any() else ''}, "
              f"in view {n_in}/{len(tr['t'])} frames")
        if not tr["punch_confirmed"][0]:
            print(f"  {name.capitalize()}: position and magnitude from the ephemeris; not checked against PUNCH frames")
        if not tr["calibrated"].all():
            print(f"  {name.capitalize()} is outside the radii where PUNCH was measured for part of the run: "
                  f"S/N and shape use the nearest measured radius")
        if n_in == 0:
            print(f"  {name.capitalize()} is outside the frame (rows 0-{r_max}) for the whole run: not drawn")
