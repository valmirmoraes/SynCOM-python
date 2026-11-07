import numpy as np


DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def load_velocity_profile(simple=False, seed=22):
    """
    Load or generate velocity and period arrays for SynCOM simulation.
    
    Parameters:
        simple (bool): If True, use a sinusoidal profile. Otherwise use default.
        seed (int): Seed for random number generation.
    
    Returns:
        velocity (np.ndarray): Array of velocities (km/s) at 0.1 deg intervals.
        period (np.ndarray): Array of periods (s) at 0.1 deg intervals.
        blob_sizes (np.ndarray): Array of sizes (R_sun) at 0.1 deg intervals.
    """
    rng = np.random.default_rng(seed)

    if simple:
        theta = np.arange(360.0)  # Degrees
        v_theta = 250.0 + 100.0 * np.cos(2 * np.pi * theta / 180.0)
        velocity = np.interp(np.linspace(0, 359.9, 3600), theta, v_theta)
        period = rng.uniform(5400, 10800, 3600)  # 1.5 to 3 hours
    else:
        # Placeholder: realistic velocity and freq arrays would be loaded here
        frequency = np.load("frequency.npy")
        period = 1.0 / frequency
        velocity = np.load("velocity.npy")

    blob_sizes = rng.uniform(0.1, 1.1, 3600)
    
    return velocity, period, blob_sizes


def generate_blob_positions(n_blobs, initial_r_sun=1.0, seed=123):
    """
    Generate initial blob positions in polar coordinates.
    
    Returns:
        radial_i, phi_i, theta_i, psi_deg
    """
    rng = np.random.default_rng(seed)

    radial_i = np.full(n_blobs, initial_r_sun)
    phi_i = rng.uniform(0, 360, n_blobs) * DEG2RAD
    theta_i = rng.uniform(0, 180, n_blobs) * DEG2RAD

    psi_rad = np.where(np.sin(phi_i) >= 0, 2 * np.pi - theta_i, theta_i)
    psi_deg = (psi_rad * RAD2DEG * 10).astype(int)  # 0.1-degree scale
    psi_deg = np.clip(psi_deg, 0, 3599)  # Avoid out-of-bounds

    return radial_i, phi_i, theta_i, psi_deg


def assign_blob_properties(n_blobs, velocity_array, period_array, size_array, psi_deg):
    """
    Assign each blob's physical parameters based on psi_deg index.
    """
    v_arr = velocity_array[psi_deg]
    period_arr = period_array[psi_deg]
    L_arr = size_array[psi_deg]
    
    # Sort by velocity descending
    sort_idx = np.argsort(-v_arr)
    
    return v_arr[sort_idx], period_arr[sort_idx], L_arr[sort_idx], sort_idx


def build_loadstruc(n_blobs, radial_i, phi_i, theta_i, v_arr, period_arr, L_arr, psi_deg_sorted):
    """
    Construct the LoadStruc-like output: array of blob properties.
    """
    outarray = np.zeros((n_blobs, 8))
    time_0 = 0.0

    for i in range(n_blobs):
        outarray[i, 0] = radial_i[i]
        outarray[i, 1] = phi_i[i]
        outarray[i, 2] = theta_i[i]
        outarray[i, 3] = time_0
        outarray[i, 4] = period_arr[i]
        outarray[i, 5] = v_arr[i]
        outarray[i, 6] = L_arr[i]
        outarray[i, 7] = psi_deg_sorted[i]

        if i > 0 and int(outarray[i, 5]) == int(outarray[i-1, 5]):
            outarray[i, 3] = outarray[i-1, 3]
        else:
            time_0 += 1.0

    return outarray


def syncom_load(params, simple=True):
    """
    Main interface that mimics the SYNCOMLOAD procedure.
    
    Returns:
        dict with keys: 'outarray', 'velocity', 'period'
    """
    n_blobs = params['syncom_n_blobs']
    initial_r_sun = params['syncom_initial_R_sun']

    velocity, period, blob_sizes = load_velocity_profile(simple=True)
    radial_i, phi_i, theta_i, psi_deg = generate_blob_positions(n_blobs, initial_r_sun)

    v_arr, period_arr, L_arr, sort_idx = assign_blob_properties(n_blobs, velocity, period, blob_sizes, psi_deg)

    radial_i = radial_i[sort_idx]
    phi_i = phi_i[sort_idx]
    theta_i = theta_i[sort_idx]
    psi_deg_sorted = psi_deg[sort_idx]

    outarray = build_loadstruc(n_blobs, radial_i, phi_i, theta_i, v_arr, period_arr, L_arr, psi_deg_sorted)

    return {
        "outarray": outarray,
        "velocity": velocity,
        "period": period
    }