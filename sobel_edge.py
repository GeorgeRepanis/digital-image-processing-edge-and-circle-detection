import numpy as np
from fir_conv import fir_conv

def sobel_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:
    """
    Applies Sobel edge detection to a grayscale image.

    Args:
        in_img_array (np.ndarray): Grayscale image with float values in [0,1]
        thres (float): Threshold to binarize the gradient magnitude

    Returns:
        np.ndarray: Binary edge image (0 or 1 values)
    """

    # Define Sobel masks for x and y gradients
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=float)

    # Define origin for both the input and the filter
    mask_origin = np.array([1, 1], dtype=int)  # center of 3x3 mask
    in_origin = np.array([0, 0], dtype=int)

    # Compute derivatives in x and y directions using FIR convolution
    gx, _ = fir_conv(in_img_array, Gx, in_origin, mask_origin)
    gy, _ = fir_conv(in_img_array, Gy, in_origin, mask_origin)

    # Compute gradient magnitude
    g = np.sqrt(gx**2 + gy**2)

    # Apply threshold to get binary edge image
    out_img_array = (g > thres).astype(int)

    return out_img_array
