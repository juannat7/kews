# KEWS: Koopman Early Warning Signals

**KEWS** (pronounced “cues,” as in cues to potential tipping) is a JAX package for Koopman-based early warning signals of bifurcation- and rate-induced tipping.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.9-000000)](https://docs.jax.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/juannat7/kews/blob/main/LICENSE.txt)
[![PyPI version](https://img.shields.io/pypi/v/kews)](https://pypi.org/project/kews/)
[![PyPI downloads](https://img.shields.io/pypi/dm/kews)](https://pypi.org/project/kews/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.14716-b31b1b.svg)](https://arxiv.org/abs/2608.14716)

</div>

## Abstract

Abrupt transitions in complex systems are often preceded by early warning signals. However, most indicators rely on the notion of critical slowing down and do not generally extend to rate-induced tipping where transitions can occur without local loss of stability. This is problematic in stochastic, nonautonomous systems where internal variability and time-varying variables interact to shape tipping onset.

We use Koopman operator theory to develop a unified early warning framework for both bifurcation and rate-induced tipping in stochastic systems. Our approach builds on residual Koopman mode decomposition that measures discrepancies between dynamics and their finite-dimensional approximation, and extends it to the control setting by augmenting the observable space with time-varying control variables.

In idealized examples, the resulting indicators recover expected signatures near bifurcation points and improve detection in rate-induced regimes where classical indicators fail. We further show that learned embeddings through deep learning outperform prescribed dictionaries, especially in a high-dimensional setting.

Applied to simulations of the Atlantic Meridional Overturning Circulation, our Koopman-based indicators distinguish tipping from non-tipping trajectories and reveal interpretable spectral signatures prior to critical transition.

## Installation

Python 3.11 or newer is required. Install the library from PyPI:

```bash
pip install kews
```

To reproduce the paper or develop the code:

```bash
git clone https://github.com/juannat7/kews.git
cd kews
pip install -e ".[paper,dev]"
```

Reproduction instructions are in [examples](https://github.com/juannat7/kews/tree/main/examples) for the model systems and [experiments](https://github.com/juannat7/kews/tree/main/experiments) for the coupled AMOC and empirical cases.

## Quickstart

Compute rolling ResKMD residuals from a scalar time series:

```python
import jax.numpy as jnp
from jax import random

import kews
from kews.observables import time_delay

time = jnp.linspace(0, 10 * jnp.pi, 100)
state = jnp.sin(time) + 0.01 * random.normal(random.key(7), time.shape)

signal = kews.resdmd(
    state,
    window=50,                           # Window length in samples (or a fraction, e.g. 0.5)
    observable=time_delay(n_delay=10),   # Time-delay dictionary with 10 coordinates
    rank=0.9,                            # Retain 90% of the SVD energy
)

# One residual per time step; the first 49 entries are NaN.
print(signal.shape)  # (100,)
```

## Control-aware residuals

Pass a control time series aligned with the state:

```python
control = jnp.linspace(-1, 1, state.size)
controlled_signal = kews.resdmd(
    state,
    window=50,
    observable=time_delay(10),
    control=control,
)
```

## Observable choices

Use fixed random Fourier features instead of delay coordinates:

```python
from kews.observables import rff

random_signal = kews.resdmd(
    state,
    window=50,
    observable=rff(n_features=100, scale=0.2, seed=7),
)
```

The random basis stays fixed across windows. An autoencoder can learn the features:

```python
from kews.observables import mlp

learned_signal = kews.resdmd(
    state,
    window=50,
    observable=mlp(hidden_layers=(16, 8), bottleneck_dim=3, seed=7),
    fit_kwargs={"epochs": 200, "batch_size": 32, "lr": 1e-3},
)
```

Training uses only the first window; the encoder is then frozen. Create a new observable for each independent fit. Paper notebooks specify their full training settings.

## Spectral diagnostics

```python
from kews import diagnostics

fit = diagnostics.fit_operator(state, observable=time_delay(10), rank=0.9)
generator = diagnostics.generator_eigenvalues(fit["eigvals"], dt=1.0)
dominant = diagnostics.dominant_mode(generator, dt=1.0)
```

`dt` is the sampling interval in the units of the input data. The mode summary includes decay rate, oscillation period, and e-folding time.

## Citation

```bibtex
@article{nathaniel2026koopman,
  title={Koopman early warning signals for bifurcation and rate-induced tipping},
  author={Nathaniel, Juan and Roesch, Carla and DeSantis, Derek and Kooloth, Parvathi and Fan, Hang and Lucarini, Valerio and Romanou, Anastasia and Gentine, Pierre},
  journal={arXiv preprint arXiv:2608.14716},
  year={2026}
}
```

Code uses the [MIT license](https://github.com/juannat7/kews/blob/main/LICENSE.txt). External datasets retain their original terms.
