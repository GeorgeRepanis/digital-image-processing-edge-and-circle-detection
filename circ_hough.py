import numpy as np
from typing import Tuple

def circ_hough(
    in_img_array: np.ndarray,
    R_max: float,
    dim: np.ndarray,
    v_min: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Circle detection using Hough Transform.

    Args:
        in_img_array (np.ndarray): Binary edge image (values 0 or 1)
        R_max (float): Maximum allowed radius
        dim (np.ndarray): Number of bins along (x, y, r)
        v_min (int): Minimum number of votes to accept a circle

    Returns:
        centers (np.ndarray): (K x 2) array with detected circle centers (x, y)
        radii (np.ndarray): (K,) array with corresponding radii
        hough_space (np.ndarray): 3D accumulator array (votes)
    """

    rows, cols = in_img_array.shape
    nx, ny, nr = dim

    # Create the parameter spaces for radius and circle center coordinates
    r_values = np.linspace(1, R_max, nr)
    x_values = np.linspace(0, cols - 1, nx)
    y_values = np.linspace(0, rows - 1, ny)

    hough_space = np.zeros((ny, nx, nr), dtype=int)

    # Collect edge points from binary input image
    edge_points = np.argwhere(in_img_array == 1)

    # --- Speed optimization: limit number of edge points used ---
    max_points = 3000
    if edge_points.shape[0] > max_points:
        indices = np.random.choice(edge_points.shape[0], max_points, replace=False)
        edge_points = edge_points[indices]

    # Voting process: for each edge point, vote for possible circle centers
    for y0, x0 in edge_points:
        for r_idx, r in enumerate(r_values):
            for theta in np.linspace(0, 2 * np.pi, 18):  # sample every 20°
                a = x0 - r * np.cos(theta)
                b = y0 - r * np.sin(theta)

                if 0 <= a < cols and 0 <= b < rows:
                    # Convert to discrete indices in the accumulator space
                    a_idx = int(np.round((a / (cols - 1)) * (nx - 1)))
                    b_idx = int(np.round((b / (rows - 1)) * (ny - 1)))

                    # Increment vote count
                    hough_space[b_idx, a_idx, r_idx] += 1

    # Find local maxima in the accumulator space above threshold v_min
    detected = np.argwhere(hough_space >= v_min)

    centers = []
    radii = []

    # Convert from indices back to actual image coordinates and radius values
    for b_idx, a_idx, r_idx in detected:
        a = x_values[a_idx]
        b = y_values[b_idx]
        r = r_values[r_idx]

        centers.append([a, b])
        radii.append(r)

    return np.array(centers, dtype=float), np.array(radii, dtype=float), hough_space
