import numpy as np
from background import generate_lumpy_background
from signal_utils import generate_gaussian_blob
from noise import add_gaussian_noise

def create_dataset(signal_type='blob', n_train=10000, n_test=1000, noise=1, N_bar=15, w_b=3, amp=3, dim=(64, 64)):
    total = n_train + n_test

    # Preallocate arrays for backgrounds
    bg1 = np.array([generate_lumpy_background(dim=dim, N_bar=N_bar, w_b=w_b, a=amp) for _ in range(total)])
    bg2 = np.array([generate_lumpy_background(dim=dim, N_bar=N_bar, w_b=w_b, a=amp) for _ in range(total)])

    # Use one shared Gaussian blob signal for all images
    signal = generate_gaussian_blob(dim=dim)

    # Add signal to bg1 for present class (broadcasted)
    present = bg1 + signal

    # Vectorized Gaussian noise addition
    noise_shape = (total,) + dim
    present_noisy = present + np.random.normal(0, noise, size=noise_shape)
    absent_noisy = bg2 + np.random.normal(0, noise, size=noise_shape)

    return {
        'train': {
            'present': list(present_noisy[:n_train]),
            'absent': list(absent_noisy[:n_train])
        },
        'test': {
            'present': list(present_noisy[n_train:]),
            'absent': list(absent_noisy[n_train:])
        }
    }