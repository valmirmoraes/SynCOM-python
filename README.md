# SynCOM-python

Python implementation of **SynCOM** (Synthetic Corona Outflow Model), a simulation framework that models transient solar wind outflow in the corona. SynCOM launches radially propagating Gaussian "blobs" of plasma and renders them as synthetic coronagraph images, giving a controlled dataset with known velocities for testing and validating flow-tracking methods.

The original SynCOM is written in IDL. This repository translates it to Python and adds tools to work with STEREO/COR2 data and to measure outflow speeds.

## Repository layout

SynCOM is split into Python modules that the notebook imports:

| File | What it does |
|---|---|
| [`SynCOM_python_notebook.ipynb`](SynCOM_python_notebook.ipynb) | Step-by-step run of the model: parameters, blob profiles, images, cube, movies and the polar to Cartesian view. |
| [`syncom_prams.py`](syncom_prams.py) | `default_syncom_prams`: simulation parameters (grid, cadence, pixel size, number of blobs, ...). |
| [`syncom_load.py`](syncom_load.py) | `syncom_load`: velocity, period and size of each blob from its position angle. |
| [`gaussian_function.py`](gaussian_function.py) | `gaussian_profile`, `gaussian_wave`: 1D and 2D Gaussian blob profiles. |
| [`syncom_image.py`](syncom_image.py) | `syncom_image`: one synthetic image (sum of all blobs at one time step). |
| [`syncom_core.py`](syncom_core.py) | `syncom_core`: runs the time loop and builds the `[NX, NY, NT]` cube; `load_fits_cube` rebuilds a cube from FITS frames. |
| [`syncom_path.py`](syncom_path.py) | Run names and output folders (`SynCOM_runs/<run name>/...`). |
| [`polar_to_rect.py`](polar_to_rect.py) | Polar (position angle x radius) to Cartesian image. |
| [`syncom_movie.py`](syncom_movie.py) | Plots and MP4 movies of the polar and Cartesian cubes. |

The earlier single-notebook versions (`SynCOM_python_functions.ipynb`, `SynCOM_python_functions_v2.ipynb`) and the COR2 / TDC notebooks (`COR2_data_pipeline.ipynb`, `TDC_method.ipynb`, `open_file.ipynb`) are deprecated on this branch. They are kept in the git history under the tag [`v1-deprecated`](https://github.com/valmirmoraes/SynCOM-python/tree/v1-deprecated).

## How the SynCOM model works

1. **Parameters** (`default_syncom_prams`): number of blobs, image size in position angle × radius (3600 × 659 by default, matching COR2), cadence (300 s), pixel size (9.744 Mm ≈ 0.014 R☉), start radius, blob size scale, noise and peak brightness.
2. **Profiles** (`syncom_load`): velocity, period and blob size on a 0.1° position-angle grid. The velocity is either a simple sinusoid (`simple=True`) or the STEREO-derived profile.
3. **Blobs**: random position angles. Each blob is assigned a velocity, period and size, and is given harmonic copies spaced by its period.
4. **Images** (`syncom_image`, `syncom_core`): each blob is a 2D Gaussian moved outward at its own speed. The frames are stacked into an `[NX, NY, NT]` cube and can be written to FITS files, a `.npz` cube or an MP4 movie (`save_syncom_movie`).

## Quick start

```python
from syncom_prams import default_syncom_prams
from syncom_core import syncom_core
from syncom_movie import save_syncom_movie

params = default_syncom_prams(SYNCOM_N_BLOBS=2000)

cube = syncom_core(
    2000,               # number of blobs
    time0=0,
    time_t=20,          # frames 0-19
    syncom_version="v2",
    scale_factor=2.0,
    simple=True,        # sinusoidal velocity profile 250 + 100 cos(2 theta) km/s
    params=params,
)

save_syncom_movie(cube, filename="SynCOM_v2_movie.mp4", cadence=params["syncom_cadence"])
```

Outputs are written to `SynCOM_runs/<run name>/` (FITS frames, cube, movies). This folder is not tracked by git.

## Requirements

- Python 3.9+ (developed with Python 3.12)
- `numpy`, `scipy`, `astropy`, `matplotlib`
- `punchbowl` (PUNCH colormap used in the plots)
- `ffmpeg` for MP4 output (`conda install -c conda-forge ffmpeg`)

```bash
conda create -n syncom-punch -c conda-forge python=3.12 numpy scipy astropy matplotlib ffmpeg jupyter
```

## License

MIT. See [LICENSE](LICENSE).
