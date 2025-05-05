import numpy as np

def generate_gaussian_blob(dim=(64, 64), sigma=3, amplitude=2):
    x0, y0 = dim[1] // 2, dim[0] // 2
    X, Y = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    dist = (X - x0)**2 + (Y - y0)**2
    return amplitude * np.exp(-dist / (2 * sigma**2))