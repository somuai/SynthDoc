import numpy as np
import torch
from scipy.fft import dctn
import cv2


def compute_fft_magnitude(image_gray: np.ndarray) -> np.ndarray:
    """Compute the centered FFT magnitude spectrum of a grayscale image."""
    f_transform = np.fft.fft2(image_gray.astype(np.float32))
    f_shifted = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shifted))
    # Normalize to [0, 1]
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude


def compute_dct_map(image_gray: np.ndarray) -> np.ndarray:
    """Compute the 2D DCT of a grayscale image."""
    dct_result = dctn(image_gray.astype(np.float32), type=2, norm='ortho')
    magnitude = np.log1p(np.abs(dct_result))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude


def prepare_freq_tensor(image_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """Convert an RGB image to a 2-channel FFT+DCT tensor.

    Args:
        image_rgb: Input RGB image as numpy array (H, W, 3).
        size: Target spatial dimension.

    Returns:
        Tensor of shape (2, size, size) with FFT and DCT channels.
    """
    # Convert to grayscale
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb

    # Resize
    gray_resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)

    # Compute spectral maps
    fft_map = compute_fft_magnitude(gray_resized)
    dct_map = compute_dct_map(gray_resized)

    # Stack into 2-channel tensor
    stacked = np.stack([fft_map, dct_map], axis=0)  # (2, H, W)
    tensor = torch.from_numpy(stacked).float()

    return tensor
