from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import sparse


@dataclass
class GraphView:
    """Minimal graph contract used by the compatibility engine.

    Convention inherited from the MaleCNS notebooks:
    CSR rows are presynaptic neurons and columns are postsynaptic neurons.
    """

    A: sparse.csr_matrix
    nodes: object | None = None

    def __post_init__(self) -> None:
        self.A = self.A.tocsr().astype(np.float32, copy=False)
        self.in_strength = np.asarray(self.A.sum(axis=0)).ravel().astype(np.float32)
        self.out_strength = np.asarray(self.A.sum(axis=1)).ravel().astype(np.float32)

    @property
    def n_nodes(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.A.nnz)


class CompatReservoir:
    """Compatibility implementation of the recurrent update used in v0.4-v0.8.

    Historical update:
        signed = state * sign
        recurrent = A.T @ signed / max(in_strength, 1)
        proposal = tanh(gain * recurrent + external)
        state += leak * (proposal - state)

    New biological mechanisms should be added around this baseline rather than
    silently changing it.
    """

    def __init__(
        self,
        graph: GraphView,
        sign_vector: np.ndarray,
        leak: float = 0.2,
        gain: float = 1.2,
    ) -> None:
        self.g = graph
        self.leak = float(leak)
        self.gain = float(gain)
        self.state = np.zeros(graph.n_nodes, dtype=np.float32)
        self.sign = np.asarray(sign_vector, dtype=np.float32).reshape(-1)

        if self.sign.shape[0] != graph.n_nodes:
            raise ValueError(
                f"sign_vector has {self.sign.shape[0]} entries; "
                f"graph has {graph.n_nodes} nodes"
            )

        self.denom = np.maximum(graph.in_strength, 1.0).astype(np.float32)

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, external: Optional[np.ndarray] = None) -> np.ndarray:
        signed = self.state * self.sign
        recurrent = (
            self.g.A.T.dot(signed).astype(np.float32, copy=False) / self.denom
        )
        z = self.gain * recurrent

        if external is not None:
            ext = np.asarray(external, dtype=np.float32)
            if ext.ndim == 0:
                z = z + ext
            else:
                if ext.shape != self.state.shape:
                    raise ValueError(
                        f"external shape {ext.shape} does not match state "
                        f"shape {self.state.shape}"
                    )
                z = z + ext

        proposal = np.tanh(z).astype(np.float32, copy=False)
        self.state += self.leak * (proposal - self.state)
        return self.state


def legacy_sign_vector(graph, sign_mode: str = "biological_fast") -> np.ndarray:
    """Recover the sign vector from the original dynamics.py when available."""

    try:
        from dynamics import ConnectomeReservoir  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import dynamics.ConnectomeReservoir. Run this adapter "
            "inside the original MaleCNS project directory, or provide an "
            "explicit sign vector."
        ) from exc

    ref = ConnectomeReservoir(
        graph,
        leak=0.2,
        gain=1.2,
        sign_mode=sign_mode,
    )
    return np.asarray(ref.sign, dtype=np.float32).copy()
