import shutil
import time
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time, TimeDelta
from punchbowl.data.visualize import cmap_punch
from polar_to_rect import polar_to_rect
from syncom_prams import POLAR_PHI_RANGE, POLAR_RHO_RANGE, SYNCOM_CADENCE


def plot_syncom_image(img, title="SynCOM Synthetic Image", cmap=cmap_punch):
    """
    Plots the 2D synthetic solar wind image with a tight colorbar.

    Parameters:
        img (np.ndarray): 2D image array
        title (str): Plot title
        cmap (str): Matplotlib colormap
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(img.T, origin='lower', aspect='auto', cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)
    
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Intensity')

    ax.set_title(title)
    ax.set_xlabel('Position Angle [degrees]')
    ax.set_ylabel('Radial Distance [Solar Radii]')
    plt.tight_layout()
    plt.show()


def plot_rect_image(img_rect, rho_max=POLAR_RHO_RANGE[1], title="SynCOM Cartesian image", cmap=cmap_punch,
                    label="Intensity [arbitrary]", **kwargs):
    """Plot a Cartesian image img_rect[ix, iy] from polar_to_rect."""
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    im = ax.imshow(img_rect.T, origin="lower", cmap=cmap,
                   extent=[-rho_max, rho_max, -rho_max, rho_max], **kwargs)
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label=label)
    ax.set(title=title, xlabel=r"x [R$_\odot$]", ylabel=r"y [R$_\odot$]")
    plt.show()


def save_syncom_movie(cube, filename="SynCOM_movie.mp4", title="SynCOM", start_time=None,
                      cadence=SYNCOM_CADENCE, frames=None, fps=10, cmap=cmap_punch,
                      vmin=None, vmax=None, extent=(0.0, 360.0, 5.0, 14.226),
                      label="Intensity [arbitrary]", figsize=(12, 5.2), dpi=100):
    """
    Saves the polar cube [NX, NY, NT] as an MP4, with the same colour scale for every frame.

    Parameters:
        cube (np.ndarray): 3D polar cube [NX, NY, NT] (output of syncom_core)
        filename (str): Output .mp4 path
        title (str): Run label shown before the time stamp
        start_time (astropy Time): Time of cube frame 0; frame k is at start_time + k * cadence
            (None -> the title shows elapsed seconds instead of a UTC time stamp)
        cadence (float): Seconds between frames
        frames (iterable): Cube indices to include (default: all frames)
        fps (int): Frames per second of the video
        vmin, vmax (float): Colour limits (default: 1st and 99.5th percentiles of the cube)
        extent (tuple): [PA_min, PA_max, r_min, r_max] in degrees and R_sun
            (same ranges as POLAR_PHI_RANGE / POLAR_RHO_RANGE); None -> axes in pixels
        figsize (tuple), dpi (int): Size and resolution of the video frames

    Returns:
        Path: Path to the saved MP4 file
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D [NX, NY, NT], got shape {cube.shape}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it with: conda install -c conda-forge ffmpeg")
    frames = list(range(cube.shape[2])) if frames is None else list(frames)

    # One colour scale for the whole movie (subsampled to keep it fast)
    if vmin is None or vmax is None:
        step = max(1, len(frames) // 20)
        lo, hi = np.nanpercentile(cube[::4, ::4, frames[::step]], [1, 99.5])
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    def frame_title(i, k):
        if start_time is None:
            return f"{title}   t = {k * cadence:.0f} s   frame {i + 1}/{len(frames)}"
        t = (start_time + TimeDelta(k * cadence, format="sec")).isot
        return f"{title}   {t} UTC   frame {i + 1}/{len(frames)}"

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.12, top=0.88)   # fixed layout -> no jitter
    im = ax.imshow(cube[:, :, frames[0]].T, origin="lower", aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=extent)
    cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.12)   # colour bar as tall as the image
    fig.colorbar(im, cax=cax, label=label)
    if extent is None:
        ax.set_xlabel("Position Angle [pixels]")
        ax.set_ylabel("Radial Distance [pixels]")
    else:
        ax.set_xlabel("Position Angle [deg]")
        ax.set_ylabel(r"Radial Distance [$R_\odot$]")
    ttl = fig.suptitle(frame_title(0, frames[0]), fontsize=11)              # spans the whole figure

    # yuv420p + even dimensions -> plays in QuickTime, browsers, PowerPoint, etc.
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
    t_start = time.time()
    with writer.saving(fig, str(filename), dpi=dpi):
        for i, k in enumerate(frames):
            im.set_data(cube[:, :, k].T)
            ttl.set_text(frame_title(i, k))
            writer.grab_frame()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(frames)} frames ({time.time() - t_start:.0f} s)")

    plt.close(fig)
    print(f"✅ Saved movie: {filename} ({len(frames)} frames at {fps} fps)")
    return Path(filename)


def save_rect_movie(cube, filename, title, start_time=None, cadence=SYNCOM_CADENCE,
                    frames=None, fps=10, cmap=cmap_punch, vmin=None, vmax=None, n=1024,
                    rho_max=POLAR_RHO_RANGE[1], label="Intensity [arbitrary]", dpi=150):
    """
    Converts each polar frame of cube [NX, NY, NT] to a Cartesian image with polar_to_rect
    and saves them as an MP4, with the same colour scale for every frame.

    Parameters:
        cube (np.ndarray): 3D polar cube [NX, NY, NT]
        filename (str): Output .mp4 path
        title (str): Run label shown before the time stamp
        start_time (astropy Time): Time of cube frame 0; frame k is at start_time + k * cadence
            (None -> the title shows elapsed seconds instead of a UTC time stamp)
        cadence (float): Seconds between frames
        frames (iterable): Cube indices to include (default: all frames)
        fps (int): Frames per second of the video
        vmin, vmax (float): Colour limits (default: 1st and 99.5th percentiles of the cube)
        n (int): Pixels per side of the Cartesian image
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it with: conda install -c conda-forge ffmpeg")
    frames = list(range(cube.shape[2])) if frames is None else list(frames)

    # One colour scale for the whole movie (subsampled to keep it fast)
    if vmin is None or vmax is None:
        step = max(1, len(frames) // 20)
        lo, hi = np.nanpercentile(cube[::4, ::4, frames[::step]], [1, 99.5])
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("black")                     # occulter and outside the field of view

    def rect_frame(k):
        r = polar_to_rect(cube[:, :, k], POLAR_PHI_RANGE, POLAR_RHO_RANGE, n=n)
        return np.where(r == 0, np.nan, r).T  # .T -> imshow rows = y

    def frame_title(i, k):
        if start_time is None:
            return f"{title}   t = {k * cadence:.0f} s   frame {i + 1}/{len(frames)}"
        t = (start_time + TimeDelta(k * cadence, format="sec")).isot
        return f"{title}   {t} UTC   frame {i + 1}/{len(frames)}"

    fig, ax = plt.subplots(figsize=(8, 7.4))
    fig.subplots_adjust(left=0.09, right=0.9, bottom=0.07, top=0.93)   # fixed layout -> no jitter
    im = ax.imshow(rect_frame(frames[0]), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[-rho_max, rho_max, -rho_max, rho_max])
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.12)   # colour bar as tall as the image
    fig.colorbar(im, cax=cax, label=label)
    ax.set_xlabel(r"X [$R_\odot$]")
    ax.set_ylabel(r"Y [$R_\odot$]")
    ttl = fig.suptitle(frame_title(0, frames[0]), fontsize=11)              # spans the whole figure

    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
    t_start = time.time()
    with writer.saving(fig, str(filename), dpi=dpi):
        for i, k in enumerate(frames):
            im.set_data(rect_frame(k))
            ttl.set_text(frame_title(i, k))
            writer.grab_frame()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(frames)} frames ({time.time() - t_start:.0f} s)")

    plt.close(fig)
    print(f"✅ Saved movie: {filename} ({len(frames)} frames at {fps} fps)")

    
    return Path(filename)