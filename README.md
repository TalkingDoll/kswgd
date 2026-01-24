# KSWGD: Koopman Spectral Wasserstein Gradient Descent

A research project implementing **Diffusion Map Particle System (DMPS)** and **Koopman SDMD** for particle transport and generative modeling across various domains.

## Overview

This repository provides implementations of spectral methods for sampling from target distributions via gradient flow. The core methods include:

- **DMPS (Diffusion Map Particle System)**: Uses Gaussian kernels with diffusion map normalization to transport particles toward a target distribution

- **KSWGD (Koopman Spectral Wasserstein Gradient Descent)**: Combines kernel methods with Stein discrepancy for efficient particle transport



## Dependencies

```
numpy
scipy
matplotlib
torch
cupy (optional, for GPU acceleration)
plotly (optional, for interactive plots)
scikit-learn
tqdm
joblib
torchvision
datasets (for CelebA-HQ)
```

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd kswgd
   ```

2. **Install dependencies**:
   ```bash
   pip install -r server_setup_packages.txt
   ```

3. **Run a notebook**:
   - Start with `test_1_torus_dmps.ipynb` for a quick demonstration
   - GPU recommended for `test_3`, `test_4`, and `test_5`

---


## License

This project is for research purposes.

