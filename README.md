# SynCOM-python

Python implementation of **SynCOM** (Synthetic Corona Outflow Model), a simulation framework that models transient solar wind outflow in the corona. SynCOM launches radially propagating Gaussian "blobs" of plasma and renders them as synthetic coronagraph images, giving a controlled dataset with known velocities for testing and validating flow-tracking methods.

The original SynCOM is written in IDL. This repository translates it to Python and adds tools to work with STEREO/COR2 data and to measure outflow speeds.

## Notebooks

| Notebook | What it does |
|---|---|
| [`SynCOM_python_functions_v2.ipynb`](SynCOM_python_functions_v2.ipynb) | **Recommended.** SynCOM translation that reproduces the IDL code (`syncom.pro`, `syncom_load.pro`, `syncom_image.pro`). Given the same blobs it matches the IDL FITS output to floating-point precision. Includes tools to load and replay an IDL run, and MP4 movie output. |
| [`SynCOM_python_functions.ipynb`](SynCOM_python_functions.ipynb) | Version 1 of the translation, with a vectorized image generator and MP4 movie output. Kept for reference. It differs from IDL in blob placement and in how harmonic copies are summed (see the notes at the top of v2). |
| [`COR2_data_pipeline.ipynb`](COR2_data_pipeline.ipynb) | Loads STEREO/COR2 FITS files into a 3D array, fixes known bad pixels, resamples along the radial axis and saves the result as `.npy`. |
| [`TDC_method.ipynb`](TDC_method.ipynb) | Time-delay correlation (TDC) method: estimates the outflow velocity at each position angle by cross-correlating brightness time series at two radial distances. |
| [`open_file.ipynb`](open_file.ipynb) | Helpers to open SynCOM/COR2 FITS data (raw, normalized, with header) and fix bad pixels. |

## How the SynCOM model works

1. **Parameters** (`default_syncom_prams`): number of blobs, image size in position angle × radius (3600 × 659 by default, matching COR2), cadence (300 s), pixel size (9.744 Mm ≈ 0.014 R☉), start radius, blob size scale, noise and peak brightness.
2. **Profiles** (`syncom_load`): velocity, period and blob size on a 0.1° position-angle grid. The velocity is either a simple sinusoid (`simple=True`) or the STEREO-derived profile.
3. **Blobs**: random position angles. Each blob is assigned a velocity, period and size, and is given harmonic copies spaced by its period.
4. **Images** (`syncom_image`, `syncom_core`): each blob is a 2D Gaussian moved outward at its own speed. The frames are stacked into an `[NX, NY, NT]` cube and can be written to FITS files, a `.npz` cube or an MP4 movie (`save_syncom_movie`).

## Quick start

```python
params = default_syncom_prams(SYNCOM_N_BLOBS=5000, DATE="2023-04-11")

cube = syncom_core(
    N_BLOBS=5000,
    time0=0,
    time_t=20,          # frames 0-19
    syncom_version="v2",
    simple=True,        # sinusoidal velocity profile
    params=params,
    save_fits=False,    # True writes one ~19 MB FITS file per frame
)

plot_syncom_image(cube[:, :, 0])
save_syncom_movie(cube, filename="SynCOM_v2_movie.mp4", cadence=params["syncom_cadence"])
```

Run the notebook cells in order: the functions above are defined in `SynCOM_python_functions_v2.ipynb`.

> **Note:** in v2, `FREQ_FILE` and the IDL run paths (`IDL_ROUTINES_DIR`, `IDL_RUN_DIR`) point to a local external drive. Edit them in the second code cell to match your setup. `FREQ_FILE` (`velocity_stereo_0_to_1.sav`) provides the STEREO velocity and frequency profiles.

## Requirements

- Python 3.9+ (v2 was developed with Python 3.12)
- `numpy`, `scipy`, `astropy`, `matplotlib`
- `ffmpeg` for MP4 output (`conda install -c conda-forge ffmpeg`)

```bash
conda create -n syncom-punch -c conda-forge python=3.12 numpy scipy astropy matplotlib ffmpeg jupyter
```

## License

MIT. See [LICENSE](LICENSE).
