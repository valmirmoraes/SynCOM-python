# SynCOM-python

Python implementation of **SynCOM** (Synthetic Corona Outflow Model), a simulation framework that models transient solar wind outflow in the corona. SynCOM launches radially propagating Gaussian "blobs" of plasma and renders them as synthetic coronagraph images, giving a controlled dataset with known velocities for testing and validating flow-tracking methods.

The original SynCOM is written in IDL. This repository translates it to Python and adds tools to work with STEREO/COR2 data and to measure outflow speeds.

## Repository layout

SynCOM is split into Python modules that the notebook imports:

| File | What it does |
|---|---|
| [`SynCOM_python_notebook.ipynb`](SynCOM_python_notebook.ipynb) | Step-by-step run of the model: parameters, blob profiles, images, cube, movies and the polar to Cartesian view. |
| [`syncom_prams.py`](syncom_prams.py) | `default_syncom_prams`: simulation parameters (grid, cadence, pixel size, number of blobs, ...). |
| [`syncom_load.py`](syncom_load.py) | `syncom_load`: velocity, period and size of each blob from its position angle; `syncom_blob_table`: the same in pixels and frames, computed once per run. |
| [`gaussian_function.py`](gaussian_function.py) | `gaussian_profile`, `gaussian_wave`: 1D and 2D Gaussian blob profiles. |
| [`syncom_image.py`](syncom_image.py) | `syncom_image`: one synthetic image. Every blob is launched each time its period is up, behind the occulter, before and during the whole run. |
| [`syncom_core.py`](syncom_core.py) | `syncom_core`: runs the time loop and builds the `[NX, NY, NT]` cube. |
| [`syncom_fits.py`](syncom_fits.py) | `write_fits_frames`: writes only the time steps you choose as FITS files; `load_fits_cube` reads them back. |
| [`syncom_path.py`](syncom_path.py) | Run names and output folders (`SynCOM_runs/<run name>/...`). |
| [`syncom_notebook_tests.ipynb`](syncom_notebook_tests.ipynb) | Tests and plots of the periodic blobs: v2 against v3, launch behind the occulter and equal brightness at every radius, one blob reappearing every period, radius–time diagrams, speeds recovered from the cube, the FITS sample. |
| [`polar_to_rect.py`](polar_to_rect.py) | Polar (position angle x radius) to Cartesian image. |
| [`syncom_movie.py`](syncom_movie.py) | Plots and MP4 movies of the polar and Cartesian cubes. |

The earlier single-notebook versions (`SynCOM_python_functions.ipynb`, `SynCOM_python_functions_v2.ipynb`) and the COR2 / TDC notebooks (`COR2_data_pipeline.ipynb`, `TDC_method.ipynb`, `open_file.ipynb`) are deprecated on this branch. They are kept in the git history under the tag [`v1-deprecated`](https://github.com/valmirmoraes/SynCOM-python/tree/v1-deprecated).

## How the SynCOM model works

1. **Parameters** (`default_syncom_prams`): number of blobs, image size in position angle × radius (3600 × 659 by default, matching COR2), cadence (300 s), pixel size (0.0142 R☉ ≈ 9.87 Mm), the radius where blobs enter the field (5 R☉, the bottom of the polar image), blob size scale, noise and peak brightness.
2. **Profiles** (`syncom_load`): velocity, period and blob size on a 0.1° position-angle grid. The velocity is either a simple sinusoid (`simple=True`) or the STEREO-derived profile.
3. **Blobs**: random position angles. Each blob is assigned a velocity, period and size (`syncom_blob_table` converts them to pixels and frames once per run).
4. **Images** (`syncom_image`, `syncom_core`): every blob is a source that launches a 2D Gaussian each time its period is up, from before the first frame to the last. A blob begins from its centre: at launch its centre is six radial widths (12L) below 5 R☉, behind the occulter, so it slides into the field fully formed and the mean brightness is the same at every radius. Each launch moves outward at the blob's speed, so one blob makes a train of launches `v·P` apart, and the field is equally full at every frame. The frames are stacked into an `[NX, NY, NT]` cube, which can be saved as a `.npz`, written to FITS for chosen frames (`write_fits_frames`) or turned into an MP4 movie (`save_syncom_movie`, `save_rect_movie`).

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
    syncom_version="v3",
    scale_factor=2.0,
    simple=True,        # sinusoidal velocity profile 250 + 100 cos(2 theta) km/s
    params=params,
)

save_syncom_movie(cube, filename="SynCOM_movie.mp4", cadence=params["syncom_cadence"])

# FITS files only for the frames you need (float64, uncompressed, ~19 MB each)
from syncom_fits import write_fits_frames
write_fits_frames(cube, "SynCOM_v3_N_BLOBS_2000_SCALE_FACTOR_2.0", frames=range(0, 20), params=params)
```

Outputs are written to `SynCOM_runs/<run name>/` (FITS frames, cube, movies). This folder is not tracked by git.

## Sample data

[`sample_FITS/SynCOM_v3.1-periodic_blobs_N_BLOBS_10000_SCALE_FACTOR_1.0/`](sample_FITS/SynCOM_v3.1-periodic_blobs_N_BLOBS_10000_SCALE_FACTOR_1.0/) holds time steps 0–29 (2.5 h at 300 s cadence) of the notebook run: 10000 blobs, scale factor 1, `simple=True`, 500 frames in total. Each file is float64, uncompressed, shape (659, 3600) = (radius, position angle), about 19 MB. Read them with:

```python
from syncom_fits import load_fits_cube
cube = load_fits_cube("sample_FITS/SynCOM_v3.1-periodic_blobs_N_BLOBS_10000_SCALE_FACTOR_1.0",
                      "SynCOM_v3.1-periodic_blobs_N_BLOBS_10000_SCALE_FACTOR_1.0", 0, 30, 3600, 659)
```

## Expected run times

Time to simulate 300 frames with `syncom_core` (the `syncom_image` loop), measured on an Apple M5:

| blobs \ scale factor | 1.0 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|
| 1000 | 11 s | 14 s | 14 s | 16 s |
| 2000 | 20 s | 22 s | 26 s | 28 s |
| 5000 | 47 s | 54 s | 62 s | 70 s |
| 10000 | 96 s | 109 s | 132 s | 150 s |

- **Frames:** time grows linearly with the number of frames.
- **Scale factor:** going from 1.0 to 4.0 adds 40–56%. Each blob is launched 12L below the field and drawn until it is 12L above it, so larger blobs mean more launches per frame. One frame takes about 7–8 µs per launch drawn.
- **Saving:** `save=True` adds about 2 min per 300 frames (117 s for a 5.3 GB compressed float64 `.npz`), whatever the number of blobs.
- **Movies:** the polar and Cartesian movies add about 50 s per 300 frames.
- **Memory:** the cube needs 5.7 GB per 300 frames (3600 × 659 float64 values per frame).

## Versions

| tag | what it is |
|---|---|
| `v1-deprecated` | the earlier single-notebook versions |
| `v2-fixed` | the modular package with the fixed model. Each blob is a fixed train of 7 copies, so blobs run out before the end of the simulation. |
| `v3-periodic_blobs` | every blob is launched again each time its period is up, behind the occulter, so the field is equally full at every frame and every radius (see `syncom_notebook_tests.ipynb`) |
| `v3.1-periodic_blobs` | FITS headers with a WCS and the model parameters (`syncom_header.py`), progress printing, the notebook set for the 5–180 R☉ field |
| `v4-planet_body` | planets in the frames (`syncom_planets.py`): positions from the ephemeris at every time step, brightness, size and saturation calibrated on PUNCH L3 CAM frames; plots and movies take their axes from the run's `params` |

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
