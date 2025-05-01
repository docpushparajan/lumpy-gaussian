# lumpy-gaussian

Final project for BIOE 580, Spring 2025. \
Numerical observer analysis of Gaussian blob detection in lumpy background images.

#### Group members: Akanksha Kumar, Nishant Bhamidipati, Dhakshenan Pushparajan, and Ryan Alvin

### Project Summary
Evaluate the performance of the Hotelling Observer (HO) and Channelized Hotelling Observer (CHO) in detecting Gaussian blob signals embedded in lumpy backgrounds with added Gaussian noise.

### Methods
- Lumpy background generation using 2D Gaussian lumps.
- Signal-present images contain a centered Gaussian blob (size=3, amplitude=2).
- Gaussian noise (σ = 1, 2, 3) added to all images.
- Observers:
  - Hotelling Observer (HO)
  - CHO with Laguerre–Gauss and Gabor channels
- Metric: Area Under the ROC Curve (AUC)

### Data
- 22,000 images per configuration
- 8 background configurations (varying lump number, size, amplitude)

### Outputs
- AUC vs. lump number, lump size, lump amplitude
- AUC vs. noise level

