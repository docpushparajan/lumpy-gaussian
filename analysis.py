from dataset import create_dataset
from observer_hotelling import hotelling_observer
from observer_cho import (
    create_laguerre_gauss_channels,
    create_gabor_channels,
    cho_analysis
)
from output_utils import compute_aucs, plot_auc_vs_param
from joblib import Parallel, delayed
import numpy as np

# Cache channels globally so they're only created once per sweep
LG_CHANNELS = create_laguerre_gauss_channels()
GABOR_CHANNELS = create_gabor_channels()

def run_single_case(param_name, val, fixed_args):
    kwargs = fixed_args.copy()
    kwargs[param_name] = val
    data = create_dataset(**kwargs)

    test_imgs = data['test']['present'] + data['test']['absent']
    test_labels = [1]*len(data['test']['present']) + [0]*len(data['test']['absent'])

    ho_scores = hotelling_observer(data['train']['present'], data['train']['absent'], test_imgs)
    cho_lg_scores = cho_analysis(data['train']['present'], data['train']['absent'], test_imgs, LG_CHANNELS)
    cho_gabor_scores = cho_analysis(data['train']['present'], data['train']['absent'], test_imgs, GABOR_CHANNELS)

    ho_auc, cho_lg_auc, cho_gabor_auc = compute_aucs(data, test_imgs, test_labels, ho_scores, cho_lg_scores, cho_gabor_scores)
    return (val, ho_auc, cho_lg_auc, cho_gabor_auc)

def evaluate_auc_vs(param_name, values, fixed_args):
    results = Parallel(n_jobs=-1)(
        delayed(run_single_case)(param_name, val, fixed_args) for val in values
    )

    # Sort results in the order of input values (preserves intent even if Parallel reorders)
    results.sort(key=lambda x: values.index(x[0]))

    ho_aucs, cho_lg_aucs, cho_gabor_aucs = [], [], []
    for val, ho, cho_lg, cho_gabor in results:
        print(f"Running: {param_name} = {val}")
        print(f"  HO AUC: {ho:.4f}")
        print(f"  CHO (Laguerre-Gauss) AUC: {cho_lg:.4f}")
        print(f"  CHO (Gabor) AUC: {cho_gabor:.4f}")
        ho_aucs.append(ho)
        cho_lg_aucs.append(cho_lg)
        cho_gabor_aucs.append(cho_gabor)

    plot_auc_vs_param(param_name, values, ho_aucs, cho_lg_aucs, cho_gabor_aucs)
    return ho_aucs, cho_lg_aucs, cho_gabor_aucs