"""Observable factories for :func:`kews.resdmd`."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from ._core import delay_embed, random_fourier_features


def _matrix(x: jax.Array) -> jax.Array:
    """Orient time-series input as ``(state, time)``."""
    return jnp.atleast_2d(x)


class _TimeDelay:
    """Time-delay observable returned by :func:`kews.observables.time_delay`."""

    def __init__(self, n_delay: int):
        if n_delay < 1:
            raise ValueError("n_delay must be >= 1")
        self.n_delay = n_delay

    def transform(self, x: jax.Array) -> jax.Array:
        """Return delayed state columns."""
        return delay_embed(x, self.n_delay)


class _RFF:
    """Random Fourier observable returned by :func:`kews.observables.rff`."""

    def __init__(self, n_features: int, scale: float, seed: int):
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.n_features = n_features
        self.scale = scale
        self.seed = seed

    def transform(self, x: jax.Array) -> jax.Array:
        """Return a fixed random cosine feature matrix."""
        key = jax.random.PRNGKey(self.seed)
        return random_fourier_features(x, self.n_features, self.scale, key)


def _activation(name: str) -> Callable[[jax.Array], jax.Array]:
    """Resolve a supported activation name."""
    activations = {
        "leaky_relu": nn.leaky_relu,
        "relu": nn.relu,
        "sigmoid": nn.sigmoid,
        "tanh": nn.tanh,
    }
    if name not in activations:
        raise ValueError(f"activation must be one of {tuple(activations)}")
    return activations[name]


class _AutoEncoder(nn.Module):
    """Symmetric dense autoencoder with an exposed encoder.

    Layer creation order preserves the paper's seeded parameter initialization.
    """

    input_dim: int
    hidden_layers: tuple[int, ...]
    bottleneck_dim: int
    activation: Callable[[jax.Array], jax.Array]

    @nn.compact
    def encode(self, x: jax.Array) -> jax.Array:
        """Return bottleneck features shaped ``(features, time)``."""
        hidden = x.T
        for index, width in enumerate(self.hidden_layers):
            hidden = nn.Dense(width, use_bias=False)(hidden)
            if index < len(self.hidden_layers) - 1:
                hidden = self.activation(hidden)
        return nn.Dense(self.bottleneck_dim, use_bias=False)(hidden).T

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Reconstruct input shaped ``(state, time)``."""
        hidden = self.encode(x).T
        for index, width in enumerate(reversed(self.hidden_layers)):
            hidden = nn.Dense(width, use_bias=False)(hidden)
            if index < len(self.hidden_layers) - 1:
                hidden = self.activation(hidden)
        return nn.Dense(self.input_dim, use_bias=False)(hidden).T


@lru_cache(maxsize=32)
def _train_step(module: _AutoEncoder, lr: float):
    """Reuse compilation across the paper's same-architecture ensemble fits."""
    optimizer = optax.adamw(lr)

    @jax.jit
    def step(params: Any, state: Any, batch: jax.Array):
        def loss(candidate: Any) -> jax.Array:
            reconstruction = module.apply(candidate, batch.T)
            return jnp.mean((batch.T - reconstruction) ** 2)

        _, gradients = jax.value_and_grad(loss)(params)
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state

    return step, optimizer


class _MLP:
    """Trainable observable returned by :func:`kews.observables.mlp`."""

    fit_on_first_window = True

    def __init__(
        self,
        hidden_layers: Sequence[int],
        bottleneck_dim: int,
        activation: str,
        seed: int,
    ):
        if not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        if bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be >= 1")
        self.hidden_layers = tuple(hidden_layers)
        self.bottleneck_dim = bottleneck_dim
        self.activation = _activation(activation)
        self.seed = seed
        self._input_dim: Optional[int] = None
        self._module: Optional[_AutoEncoder] = None
        self._params: Any = None

    def fit(
        self,
        x: jax.Array,
        *,
        lr: float = 1e-3,
        epochs: int = 1000,
        batch_size: int = 32,
        key: Optional[jax.Array] = None,
        shuffle: bool = True,
    ) -> "_MLP":
        """Train on time columns using AdamW and return this observable.

        The seed controls both initialization and epoch shuffles. Repeated
        calls continue from existing parameters with a fresh optimizer state.
        ``resdmd`` supplies only the first analysis window to this method.
        """
        if lr <= 0 or epochs < 1 or batch_size < 1:
            raise ValueError("lr, epochs, and batch_size must be positive")
        x = _matrix(x)
        if self._input_dim not in (None, x.shape[0]):
            raise ValueError(
                f"MLP was fitted with {self._input_dim} states, got {x.shape[0]}"
            )
        self._input_dim = x.shape[0]
        self._module = _AutoEncoder(
            self._input_dim,
            self.hidden_layers,
            self.bottleneck_dim,
            self.activation,
        )
        key = jax.random.PRNGKey(self.seed) if key is None else key
        if self._params is None:
            self._params = self._module.init(jax.random.PRNGKey(self.seed), x)
        params = self._params
        step, optimizer = _train_step(self._module, lr)
        state = optimizer.init(params)

        samples = x.T
        batch_size = min(batch_size, samples.shape[0])
        for _ in range(epochs):
            epoch_samples = samples
            if shuffle:
                key, shuffle_key = jax.random.split(key)
                # Shuffle the original sample order, preserving the seeded training path.
                epoch_samples = samples[jax.random.permutation(shuffle_key, samples.shape[0])]
            for start in range(0, samples.shape[0], batch_size):
                params, state = step(params, state, epoch_samples[start : start + batch_size])
        self._params = params
        return self

    def transform(self, x: jax.Array) -> jax.Array:
        """Encode a trajectory after fitting."""
        x = _matrix(x)
        if self._module is None or self._params is None:
            raise RuntimeError("MLP must be fitted before transform")
        if x.shape[0] != self._input_dim:
            raise ValueError(
                f"MLP was fitted with {self._input_dim} states, got {x.shape[0]}"
            )
        return self._module.apply(self._params, x, method=self._module.encode)


def time_delay(n_delay: int = 5) -> _TimeDelay:
    """Create a delay-coordinate observable.

    Args:
        n_delay: Number of delayed state copies in each feature vector.

    Returns:
        An observable accepted by :func:`kews.resdmd`.
    """
    return _TimeDelay(n_delay)


# Common abbreviation for time-delay features.
tdf = time_delay


def rff(
    n_features: int = 500, scale: float = 1.0, seed: int = 0
) -> _RFF:
    """Create a random Fourier feature observable.

    Args:
        n_features: Number of random cosine features.
        scale: Positive RBF-kernel length scale.
        seed: Random projection seed.

    Returns:
        An observable accepted by :func:`kews.resdmd`.
    """
    return _RFF(n_features, scale, seed)


def mlp(
    hidden_layers: Sequence[int] = (16, 32),
    bottleneck_dim: int = 3,
    activation: str = "sigmoid",
    seed: int = 0,
) -> _MLP:
    """Create a learned autoencoder observable.

    Args:
        hidden_layers: Encoder widths; the decoder mirrors them in reverse.
        bottleneck_dim: Number of learned observable features.
        activation: ``"relu"``, ``"tanh"``, ``"sigmoid"``, or
            ``"leaky_relu"``.
        seed: Parameter initialization and training seed.

    Returns:
        A trainable observable accepted by :func:`kews.resdmd`.
    """
    return _MLP(hidden_layers, bottleneck_dim, activation, seed)


__all__ = ["mlp", "rff", "tdf", "time_delay"]
