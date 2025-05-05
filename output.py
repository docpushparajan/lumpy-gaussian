import time

from analysis import evaluate_auc_vs

if __name__ == "__main__":
    start = time.time()

    base_args = dict(signal_type='blob', n_train=10000, n_test=1000, noise=2, N_bar=15, w_b=3, amp=3)

    print("AUC vs. Number of Lumps")
    ho_nbar, cho_lg_nbar, cho_gabor_nbar = evaluate_auc_vs("N_bar", [15, 100], base_args)

    print("AUC vs. Lump Size")
    ho_nbar, cho_lg_nbar, cho_gabor_nbar = evaluate_auc_vs("w_b", [3, 5], base_args)

    print("AUC vs. Lump Amplitude")
    ho_nbar, cho_lg_nbar, cho_gabor_nbar = evaluate_auc_vs("amp", [3, 5], base_args)

    print("AUC vs. Noise Level")
    ho_nbar, cho_lg_nbar, cho_gabor_nbar = evaluate_auc_vs("noise", [1, 2, 3], base_args)

    end = time.time()
    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"Time elapsed: {int(minutes)} minutes, {seconds:.2f} seconds")