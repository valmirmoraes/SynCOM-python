from pathlib import Path


RUNS_DIR = Path("SynCOM_runs")      # every run gets its own folder in here
RUN_SUBDIRS = {"run": "", "fits": "FITS", "cubes": "cubes", "movies": "movies"}


def syncom_run_name(syncom_version, n_blobs, scalefactor):
    """Run label: prefix of every file and name of the run folder."""
    return f"SynCOM_{syncom_version}_N_BLOBS_{n_blobs}_SCALE_FACTOR_{scalefactor}"


def syncom_run_dir(run_name, kind="run"):
    """Folder of a run (kind='run') or of one type of output ('fits', 'cubes', 'movies'); created if missing."""
    folder = RUNS_DIR / run_name / RUN_SUBDIRS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def syncom_output_path(run_name, kind, suffix):
    """
    Path of an output file, always named <run_name>_<suffix>:
        SynCOM_runs/<run_name>/<run_name>_params.npy
        SynCOM_runs/<run_name>/FITS/<run_name>_<t:04d>.fts
        SynCOM_runs/<run_name>/cubes/<run_name>_cube.npz
        SynCOM_runs/<run_name>/movies/<run_name>_polar_movie.mp4, <run_name>_rect_movie.mp4
    """
    return syncom_run_dir(run_name, kind) / f"{run_name}_{suffix}"
