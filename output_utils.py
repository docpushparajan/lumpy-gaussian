from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def compute_aucs(data, test_imgs, test_labels, ho_scores, cho_lg_scores, cho_gabor_scores):
    ho_auc = roc_auc_score(test_labels, ho_scores)
    cho_lg_auc = roc_auc_score(test_labels, cho_lg_scores)
    cho_gabor_auc = roc_auc_score(test_labels, cho_gabor_scores)
    return ho_auc, cho_lg_auc, cho_gabor_auc

def plot_auc_vs_param(param_name, values, ho_aucs, cho_lg_aucs, cho_gabor_aucs):
    plt.figure()
    plt.plot(values, ho_aucs, marker='o', label='HO')
    plt.plot(values, cho_lg_aucs, marker='x', label='CHO (LG)')
    plt.plot(values, cho_gabor_aucs, marker='s', label='CHO (Gabor)')
    plt.title(f"AUC vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("AUC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()