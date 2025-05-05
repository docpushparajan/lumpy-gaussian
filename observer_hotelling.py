import numpy as np

def hotelling_observer(train_present, train_absent, test_images):
    mean1 = np.mean(train_present, axis=0).flatten()
    mean0 = np.mean(train_absent, axis=0).flatten()
    delta = mean1 - mean0
    train_data = np.array(train_present + train_absent)
    cov = np.cov(train_data.reshape(len(train_data), -1), rowvar=False)
    w = np.linalg.inv(cov + 1e-3 * np.eye(cov.shape[0])).dot(delta)
    return [np.dot(w, img.flatten()) for img in test_images]