import numpy as np
from fir_conv import fir_conv

def log_edge(in_img_array: np.ndarray) -> np.ndarray:
    """
    Edge detection using Laplacian of Gaussian (LoG) operator 
    followed by zero-crossing detection.

    Args:
        in_img_array (np.ndarray): Grayscale image with values in [0,1], dtype=float

    Returns:
        np.ndarray: Binary edge map (0 or 1 values)
    """

    # Define the 5x5 LoG mask (as provided in the assignment)
    log_mask = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ], dtype=float)

    # Define mask and input origins (for FIR convolution compatibility)
    mask_origin = np.array([2, 2], dtype=int)  # center of 5x5 mask
    in_origin = np.array([0, 0], dtype=int)

    # Convolve input image with LoG filter using fir_conv
    convolved, _ = fir_conv(in_img_array, log_mask, in_origin, mask_origin)

    # Initialize output binary image
    out_img_array = np.zeros_like(convolved, dtype=int)

    rows, cols = convolved.shape

    # Detect zero-crossings: any sign change in 3x3 neighborhood implies an edge
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = convolved[i, j]
            neighborhood = convolved[i-1:i+2, j-1:j+2]
            if np.any((neighborhood < 0) & (center > 0)) or np.any((neighborhood > 0) & (center < 0)):
                out_img_array[i, j] = 1

    return out_img_array
