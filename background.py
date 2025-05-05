import numpy as np

def generate_lumpy_background(dim=(64, 64), N_bar=15, a=3, w_b=3):
    N_b = np.random.poisson(N_bar)
    bg = np.zeros(dim)
    X, Y = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    for _ in range(N_b):
        x0, y0 = np.random.randint(0, dim[1]), np.random.randint(0, dim[0])
        dist = (X - x0)**2 + (Y - y0)**2
        lump = a * np.exp(-dist / (2 * w_b**2))
        bg += lump
    return bg