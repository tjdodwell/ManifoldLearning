# ManifoldLearning

Goal oriented dimension reduction on the Grassmann manifold with ridge surrogate models and heteroscedastic GP residual modelling.

This repository contains research code for manifold learning methods built around:

- Grassmann manifold optimisation
- Ridge surrogate modelling
- Gaussian process residual modelling (via GPyTorch)
- Torch based implementation

---

## Project Structure

This project uses a `src/` layout.

```
ManifoldLearning/
  src/manifold_learning/
  examples/
  tests/
  pyproject.toml
```

The Python package lives in:

```
src/manifold_learning
```


## Installation (Recommended)

### Install uv

If you do not already have `uv`, install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal afterwards.

### Clone the repository

```bash
git clone <REPO_URL>
cd ManifoldLearning
```

### Create the environment and install

```bash
uv venv
uv pip install -e .
```

This installs the package in editable mode so local changes are immediately reflected.

# Stochastic Grassmann Optimisation for Goal Oriented Dimension Reduction

[Link to white paper on method.](https://hackmd.io/@tjdodwell/SJ3l_3lL-l)
