import numpy as np

def generate_lumpy_background(dim=(64, 64), N_bar=15, a=1, w_b=3):
    """
    Generate a lumpy background based on a 2D Gaussian model

    Args:
    - dim (tuple): Dimensions of the background image
    - N_bar (int): Mean number of lumps (Poisson distribution parameter)
    - a (float): Lump amplitude
    - w_b (float): Lump width (standard deviation of Gaussian)

    Returns:
    - background (2D array): The generated lumpy background
    """
    N_b = np.random.poisson(N_bar)  # Number of lumps sampled from Poisson distribution
    background = np.zeros(dim)

    # Create lumps at random positions
    for _ in range(N_b):
        # Sample random position for the lump center
        r_n = np.random.randint(0, dim[0]), np.random.randint(0, dim[1])

        # Generate a Gaussian lump
        X, Y = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]))
        r = np.stack([X, Y], axis=-1)
        dist = np.sum((r - r_n) ** 2, axis=-1)
        lump = a * np.exp(-dist / (2 * w_b ** 2))

        # Add lump to the background
        background += lump

    return background