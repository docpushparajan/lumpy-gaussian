--------------------------------------------------------------------------------
PROJECT SUMMARY
--------------------------------------------------------------------------------

Evaluate the performance of the Hotelling Observer (HO) and Channelized Hotelling Observer (CHO) 
in detecting Gaussian blob signals embedded in stochastic lumpy backgrounds with added Gaussian noise.

--------------------------------------------------------------------------------
METHODS
--------------------------------------------------------------------------------

- Generated lumpy backgrounds using Poisson-distributed 2D Gaussian lumps.
- Signal-present images contain a fixed, centered Gaussian blob (σ = 3, amplitude = 2).
- Gaussian noise added at three levels (σ = 1, 2, 3).
- Observers implemented:
  - Hotelling Observer (HO)
  - CHO with Laguerre–Gauss channels
  - CHO with Gabor channels
- Performance metric: Area Under the ROC Curve (AUC)
- Fully modular implementation with performance-optimized execution:
  - Vectorized dataset creation
  - Parallel parameter sweeps
  - Channel caching for CHO

--------------------------------------------------------------------------------
DATA
--------------------------------------------------------------------------------

- 22,000 images per parameter configuration (10,000 train + 1,000 test for both signal-present and signal-absent)
- 8 background configurations:
  - Number of lumps (15, 100)
  - Lump size (3, 5)
  - Lump amplitude (3, 5)
- 3 noise configurations (σ = 1, 2, 3)

--------------------------------------------------------------------------------
OUTPUTS
--------------------------------------------------------------------------------

- AUC vs. number of lumps
- AUC vs. lump size
- AUC vs. lump amplitude
- AUC vs. noise level
- Logically ordered output display with sequential parameter reporting
- Full reproducibility with consistent randomization and pre-defined signal characteristics

--------------------------------------------------------------------------------
FILES AND THEIR PURPOSE
--------------------------------------------------------------------------------

1. dataset.py
   - Generates signal-present and signal-absent images using a stochastic lumpy background.
   - Adds i.i.d. Gaussian noise using vectorized batch operations.
   - Uses a single centered Gaussian blob signal added to all signal-present images (per project spec).
   - Fully optimized for speed with NumPy broadcasting and preallocation.

2. background.py
   - Contains the function to generate lumpy background based on Poisson-distributed lumps.

3. signal_utils.py
   - Generates a centered Gaussian blob signal with user-defined sigma and amplitude.

4. noise.py
   - Adds i.i.d. Gaussian noise to images (used internally by dataset.py).

5. observer_hotelling.py
   - Implements the Hotelling Observer using the full image resolution (64x64).
   - Applies regularized covariance inversion to compute the decision template.
   - No dimensionality reduction is applied (maximizes fidelity).

6. observer_cho.py
   - Implements Channelized Hotelling Observer (CHO).
   - Includes both Laguerre–Gauss and Gabor channel definitions.
   - Fully vectorized projection using NumPy's tensordot for speed.

7. output_utils.py
   - Computes AUC values using sklearn.
   - Plots AUC vs parameter using matplotlib.

8. analysis.py
   - Contains evaluate_auc_vs(), which performs parameter sweeps.
   - Uses joblib.Parallel to run sweeps in parallel across CPU cores.
   - Channels (LG and Gabor) are cached and reused across sweeps for efficiency.
   - Print outputs are ordered sequentially after parallel execution completes.

9. output.py
   - Main driver script.
   - Defines base dataset parameters (e.g., signal type, background complexity, noise level).
   - Calls evaluate_auc_vs() for:
       - Number of lumps (N_bar)
       - Lump size (w_b)
       - Lump amplitude (amp)
       - Noise standard deviation
   - Prints AUCs and generates plots.

--------------------------------------------------------------------------------
HOW TO RUN THE CODE
--------------------------------------------------------------------------------

1. Ensure all `.py` files are in the same directory.
2. Run the main script:

       python output.py

3. The script will:
   - Generate a full dataset with 10,000 training and 1,000 testing images per class.
   - Add a fixed centered Gaussian blob to signal-present images.
   - Add i.i.d. Gaussian noise to all images using user-defined standard deviation.
   - Run Hotelling Observer and Channelized HO (LG + Gabor).
   - Perform AUC analysis across background parameters and noise level.
   - Print results and generate AUC vs parameter plots.

--------------------------------------------------------------------------------
OPTIMIZATIONS IMPLEMENTED
--------------------------------------------------------------------------------

- Vectorized image generation and noise application (in dataset.py)
- Preallocated NumPy arrays (in dataset.py)
- Channel caching for CHO (in analysis.py)
- Parallel execution of parameter sweeps with joblib (in analysis.py)
- Ordered printing for clarity despite parallelism
