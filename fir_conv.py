import numpy as np
from scipy.signal import convolve2d

def fir_conv(
    in_img_array: np.ndarray,
    h: np.ndarray,
    in_origin: np.ndarray,
    mask_origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs 2D convolution between input image and FIR filter h.

    Args:
        in_img_array (np.ndarray): Grayscale image as a 2D array
        h (np.ndarray): FIR filter (kernel) to be applied
        in_origin (np.ndarray): Origin of input image (ignored)
        mask_origin (np.ndarray): Origin of the mask/filter (ignored)

    Returns:
        out_img_array (np.ndarray): Resulting image after convolution
        in_origin (np.ndarray): Returned unchanged for compatibility
    """

    # Perform convolution using scipy (zero-padding outside image bounds)
    out_img_array = convolve2d(in_img_array, h, mode='same', boundary='fill', fillvalue=0)

    # Return the convolved image and the original input origin (as per spec)
    return out_img_array, in_origin
