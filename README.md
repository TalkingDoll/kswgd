# KSWGD: Koopman Spectral Wasserstein Gradient Descent

This is a research project implementing **Koopman Spectral Wasserstein Gradient Descent (KSWGD)** for particle transport and generative modeling across various domains.

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


## References

If you use KSWGD or this code in your research, please cite the following paper:

```bibtex
@misc{xu2025generativemodelingspectralanalysis,
      title={Generative Modeling through Spectral Analysis of Koopman Operator}, 
      author={Yuanchao Xu and Fengyi Li and Masahiro Fujisawa and Youssef Marzouk and Isao Ishikawa},
      year={2025},
      eprint={2512.18837},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.18837}, 
}

```

