from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse


def normalize_plastic_budget(
    delta_w: np.ndarray,
    target_rms: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize one plastic update to a fixed RMS budget."""

    delta = np.asarray(delta_w, dtype=np.float32)
    if target_rms < 0:
        raise ValueError("target_rms must be >= 0")

    raw_rms = (
        float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
        if delta.size
        else 0.0
    )

    if raw_rms <= eps or target_rms == 0:
        return np.zeros_like(delta, dtype=np.float32)

    scale = np.float32(target_rms / raw_rms)
    return (delta * scale).astype(np.float32, copy=False)


def select_edges_to_postsynaptic_nodes(
    A: sparse.csr_matrix,
    post_nodes: np.ndarray,
) -> np.ndarray:
    """Return CSR data positions for edges ending at selected post neurons."""

    A = A.tocsr()
    posts = np.asarray(post_nodes, dtype=np.int64)
    if posts.size == 0:
        return np.empty(0, dtype=np.int64)

    mask = np.isin(A.indices, posts, assume_unique=False)
    return np.flatnonzero(mask).astype(np.int64)


@dataclass
class PlasticityOverlay:
    """Plastic state only for selected CSR edges, not all graph edges."""

    positions: np.ndarray
    base_weight: np.ndarray
    fast_weight: np.ndarray
    slow_weight: np.ndarray
    min_weight: float = 0.0
    max_weight: float = np.inf

    @classmethod
    def from_matrix(
        cls,
        A: sparse.csr_matrix,
        positions: np.ndarray,
        *,
        min_weight: float = 0.0,
        max_weight: float = np.inf,
    ) -> "PlasticityOverlay":
        A = A.tocsr()
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size and (pos.min() < 0 or pos.max() >= A.data.size):
            raise ValueError("positions contain invalid CSR data offsets")

        base = A.data[pos].astype(np.float32, copy=True)
        zeros = np.zeros_like(base, dtype=np.float32)
        return cls(
            positions=pos,
            base_weight=base,
            fast_weight=zeros.copy(),
            slow_weight=zeros.copy(),
            min_weight=float(min_weight),
            max_weight=float(max_weight),
        )

    @property
    def effective_weight(self) -> np.ndarray:
        w = self.base_weight + self.fast_weight + self.slow_weight
        return np.clip(
            w, self.min_weight, self.max_weight
        ).astype(np.float32, copy=False)

    def apply_to_matrix(
        self, A: sparse.csr_matrix, *, copy: bool = True
    ) -> sparse.csr_matrix:
        out = A.copy() if copy else A
        if not sparse.isspmatrix_csr(out):
            out = out.tocsr()
        out.data[self.positions] = self.effective_weight.astype(
            out.data.dtype, copy=False
        )
        return out


class EligibilityTrace:
    """Eligibility state for only the plastic edges."""

    def __init__(
        self,
        pre_index: np.ndarray,
        post_index: np.ndarray,
        decay: float = 0.95,
    ) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0, 1]")

        self.pre_index = np.asarray(pre_index, dtype=np.int64)
        self.post_index = np.asarray(post_index, dtype=np.int64)

        if self.pre_index.shape != self.post_index.shape:
            raise ValueError(
                "pre_index and post_index must have the same shape"
            )

        self.decay = float(decay)
        self.value = np.zeros(
            self.pre_index.shape[0], dtype=np.float32
        )

    def reset(self) -> None:
        self.value.fill(0.0)

    def update_hebb(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32)
        coincidence = s[self.pre_index] * s[self.post_index]
        self.value *= np.float32(self.decay)
        self.value += coincidence.astype(np.float32, copy=False)
        return self.value

    def update_temporal(
        self,
        previous_state: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        pre = np.asarray(previous_state, dtype=np.float32)
        post = np.asarray(current_state, dtype=np.float32)
        coincidence = pre[self.pre_index] * post[self.post_index]
        self.value *= np.float32(self.decay)
        self.value += coincidence.astype(np.float32, copy=False)
        return self.value


class ThreeFactorRule:
    """Three-factor update: local eligibility x modulatory signal."""

    def __init__(
        self,
        *,
        target_rms_budget: float,
        post_index: np.ndarray | None = None,
        max_abs_step: float | None = None,
    ) -> None:
        self.target_rms_budget = float(target_rms_budget)
        self.post_index = (
            None
            if post_index is None
            else np.asarray(post_index, dtype=np.int64)
        )
        self.max_abs_step = max_abs_step

    def compute(
        self,
        eligibility: np.ndarray,
        modulation,
    ) -> np.ndarray:
        e = np.asarray(eligibility, dtype=np.float32)
        m = np.asarray(modulation, dtype=np.float32)

        if m.ndim == 0:
            raw = e * m
        elif m.shape == e.shape:
            raw = e * m
        elif self.post_index is not None and m.ndim == 1:
            raw = e * m[self.post_index]
        else:
            raise ValueError(
                "modulation must be scalar, edge-sized, or neuron-sized "
                "when post_index is configured"
            )

        delta = normalize_plastic_budget(
            raw, self.target_rms_budget
        )

        if self.max_abs_step is not None:
            delta = np.clip(
                delta, -self.max_abs_step, self.max_abs_step
            )

        return delta.astype(np.float32, copy=False)


class FastSlowConsolidation:
    """Two-timescale plastic state."""

    def __init__(
        self,
        overlay: PlasticityOverlay,
        *,
        fast_decay: float = 1.0,
        slow_decay: float = 1.0,
        consolidation_rate: float = 0.02,
    ) -> None:
        if not (0.0 <= fast_decay <= 1.0):
            raise ValueError("fast_decay must be in [0, 1]")
        if not (0.0 <= slow_decay <= 1.0):
            raise ValueError("slow_decay must be in [0, 1]")
        if consolidation_rate < 0:
            raise ValueError(
                "consolidation_rate must be >= 0"
            )

        self.overlay = overlay
        self.fast_decay = float(fast_decay)
        self.slow_decay = float(slow_decay)
        self.consolidation_rate = float(consolidation_rate)

    def plastic_step(self, delta_w: np.ndarray) -> None:
        delta = np.asarray(delta_w, dtype=np.float32)
        if delta.shape != self.overlay.fast_weight.shape:
            raise ValueError(
                "delta_w shape does not match plastic edge set"
            )
        self.overlay.fast_weight += delta

    def consolidate(self, gate=1.0) -> None:
        g = np.asarray(gate, dtype=np.float32)
        transfer = (
            self.overlay.fast_weight
            * np.float32(self.consolidation_rate)
            * g
        )
        self.overlay.slow_weight *= np.float32(self.slow_decay)
        self.overlay.slow_weight += transfer
        self.overlay.fast_weight *= np.float32(self.fast_decay)
