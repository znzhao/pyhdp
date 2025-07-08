# Sticky HDP-HMM-VAR

A Python implementation of Bayesian nonparametric models including Dirichlet Process (DP), Hierarchical Dirichlet Process (HDP), and Sticky HDP-HMM with VAR emissions. This package supports flexible modeling of time series and mixture data using advanced Bayesian inference techniques.

## Features

- **Dirichlet Process (DP) Mixture Models**
- **Hierarchical Dirichlet Process (HDP) for grouped data**
- **Sticky HDP-HMM**: Infinite-state HMM with self-transition bias
- **VAR Emissions**: Vector Autoregressive models for time series
- **Bayesian Linear and Multivariate Regression** (conjugate priors)
- **Collapsed Gibbs Sampling** and other inference algorithms
- **Examples and tests** for all models

## Installation

Clone the repository and install dependencies:

```sh
git clone https://github.com/yourusername/pyhdp.git
cd pyhdp
pip install -r requirements.txt
```

Dependencies include `numpy`, `scipy`, and `matplotlib`.

## Usage

Run all example tests:

```sh
python main.py
```

This will execute the test suites for DP, HDP, and Sticky HDP-HMM models.

### Example: Sticky HDP-HMM

You can run or modify the example in [examples/hdphmm.py](examples/hdphmm.py):

```python
from api.hdphmm import StickyHDPHMM

# See examples/hdphmm.py for test_multivariate(), test_linear(), etc.
```

### Example: Dirichlet Process

See [examples/dp.py](examples/dp.py) for DP mixture model usage.

## Project Structure

- [`main.py`](main.py): Entry point to run all tests
- [`api/`](api/): Core model implementations
  - [`bayes.py`](api/bayes.py): Bayesian regression and estimators
  - [`dp.py`](api/dp.py): Dirichlet Process models
  - [`hdp.py`](api/hdp.py): Hierarchical Dirichlet Process models
  - [`hdphmm.py`](api/hdphmm.py): Sticky HDP-HMM and HDP-HMM-VAR
  - [`utils.py`](api/utils.py): Utility functions and distributions
- [`examples/`](examples/): Example scripts and test cases

## Documentation

For detailed mathematical background and derivations, see [Sticky HDP-HMM-VAR.md](Sticky%20HDP-HMM-VAR.md).

## References

- Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011). A Sticky HDP-HMM with Application to Speaker Diarization. Annals of Applied Statistics.
- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet Processes. JASA.

## License

MIT License

---

**Author:** Zhenning Zhao