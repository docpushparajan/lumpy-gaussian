import numpy as np

def create_laguerre_gauss_channels(dim=(64,64), sigmas=[3,6,9]):
    X, Y = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    x0, y0 = dim[1]//2, dim[0]//2
    r2 = (X - x0)**2 + (Y - y0)**2
    return [np.exp(-r2 / (2 * s**2)) / np.linalg.norm(np.exp(-r2 / (2 * s**2))) for s in sigmas]

def create_gabor_channels(dim=(64,64), freqs=[0.1, 0.2], thetas=[0, np.pi/4, np.pi/2]):
    X, Y = np.meshgrid(np.linspace(-1, 1, dim[1]), np.linspace(-1, 1, dim[0]))
    channels = []
    for f in freqs:
        for theta in thetas:
            X_theta = X * np.cos(theta) + Y * np.sin(theta)
            gb = np.exp(-0.5 * (X_theta**2)) * np.cos(2 * np.pi * f * X_theta)
            channels.append(gb / np.linalg.norm(gb))
    return channels

def channelize_images(images, channels):
    # Vectorized CHO projection using tensordot
    images = np.array(images)
    channels = np.array(channels)
    return np.tensordot(images, channels, axes=([1, 2], [1, 2]))

def cho_analysis(train_present, train_absent, test_images, channels):
    train_chan = channelize_images(train_present + train_absent, channels)
    test_chan = channelize_images(test_images, channels)
    mean1 = np.mean(train_chan[:len(train_present)], axis=0)
    mean0 = np.mean(train_chan[len(train_present):], axis=0)
    delta = mean1 - mean0
    cov = np.cov(train_chan.T)
    w = np.linalg.inv(cov + 1e-3 * np.eye(cov.shape[0])).dot(delta)
    return [np.dot(w, x) for x in test_chan]