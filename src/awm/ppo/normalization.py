"""Training-only running normalization for 79D PPO observations."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class NormalizerState:
    count: float
    mean: tuple[float, ...]
    variance: tuple[float, ...]


class RunningObservationNormalizer:
    def __init__(self, state_dim: int = 79, *, epsilon: float = 1e-4, clip: float = 10.0) -> None:
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if clip <= 0.0:
            raise ValueError("clip must be > 0")
        self.state_dim = int(state_dim)
        self.clip = float(clip)
        self.count = float(epsilon)
        self.mean = np.zeros(self.state_dim, dtype=np.float64)
        self.var = np.ones(self.state_dim, dtype=np.float64)

    def update(self, observations: np.ndarray) -> None:
        array = np.asarray(observations, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.state_dim:
            raise ValueError(
                f"observations must have shape [batch,{self.state_dim}], got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise FloatingPointError("observations contain NaN/Inf")
        batch_count = int(array.shape[0])
        if batch_count == 0:
            return
        batch_mean = array.mean(axis=0)
        batch_var = array.var(axis=0)
        self._merge(batch_mean, batch_var, float(batch_count))

    def _merge(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = np.maximum(m2 / total, 1e-12)
        self.count = total

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float64)
        if array.shape[-1] != self.state_dim:
            raise ValueError("observation last dimension does not match state_dim")
        normalized = (array - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def normalize_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.state_dim:
            raise ValueError("observation last dimension does not match state_dim")
        mean = torch.as_tensor(self.mean, dtype=observations.dtype, device=observations.device)
        var = torch.as_tensor(self.var, dtype=observations.dtype, device=observations.device)
        return ((observations - mean) / torch.sqrt(var + 1e-8)).clamp(-self.clip, self.clip)

    def state(self) -> NormalizerState:
        return NormalizerState(
            count=self.count,
            mean=tuple(float(x) for x in self.mean),
            variance=tuple(float(x) for x in self.var),
        )

    def load_state(self, state: NormalizerState) -> None:
        if len(state.mean) != self.state_dim or len(state.variance) != self.state_dim:
            raise ValueError("normalizer state dimension mismatch")
        if state.count <= 0.0:
            raise ValueError("normalizer count must be > 0")
        variance = np.asarray(state.variance, dtype=np.float64)
        if np.any(variance <= 0.0) or not np.isfinite(variance).all():
            raise ValueError("normalizer variance must be finite and positive")
        mean = np.asarray(state.mean, dtype=np.float64)
        if not np.isfinite(mean).all():
            raise ValueError("normalizer mean must be finite")
        self.count = float(state.count)
        self.mean = mean.copy()
        self.var = variance.copy()


__all__ = ["NormalizerState", "RunningObservationNormalizer"]
