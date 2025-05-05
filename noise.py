import numpy as np

def add_gaussian_noise(image, std):
    return image + np.random.normal(0, std, image.shape)