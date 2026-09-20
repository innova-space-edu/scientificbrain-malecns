"""P1 smoke test: eligibility, budget control and consolidation."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioengine import (
    EligibilityTrace,
    PlasticityOverlay,
    ThreeFactorRule,
    FastSlowConsolidation,
)


def main() -> None:
    rng = np.random.default_rng(7)

    rows = np.array(
        [0, 0, 1, 2, 3, 4], dtype=np.int32
    )
    cols = np.array(
        [1, 2, 2, 3, 4, 0], dtype=np.int32
    )
    data = np.ones(
        rows.size, dtype=np.float32
    )
    A = sparse.csr_matrix(
        (data, (rows, cols)), shape=(5, 5)
    )

    positions = np.arange(
        A.nnz, dtype=np.int64
    )

    pre = np.repeat(
        np.arange(A.shape[0]),
        np.diff(A.indptr)
    ).astype(np.int64)
    post = A.indices.astype(np.int64)

    overlay = PlasticityOverlay.from_matrix(
        A,
        positions,
        min_weight=0.25,
        max_weight=2.0,
    )
    elig = EligibilityTrace(
        pre, post, decay=0.9
    )

    state = rng.normal(
        size=5
    ).astype(np.float32)
    e = elig.update_hebb(state)

    dopamine_per_neuron = np.array(
        [0, 0, 1, 1, 0],
        dtype=np.float32
    )
    rule = ThreeFactorRule(
        target_rms_budget=0.01,
        post_index=post,
        max_abs_step=0.05,
    )
    dw = rule.compute(
        e, dopamine_per_neuron
    )

    rms = float(
        np.sqrt(
            np.mean(
                dw.astype(np.float64) ** 2
            )
        )
    )
    if not np.isclose(
        rms, 0.01, atol=1e-6
    ):
        raise AssertionError(
            f"plastic budget mismatch: {rms}"
        )

    memory = FastSlowConsolidation(
        overlay,
        fast_decay=0.98,
        slow_decay=1.0,
        consolidation_rate=0.1,
    )
    memory.plastic_step(dw)
    before = overlay.fast_weight.copy()
    memory.consolidate(gate=1.0)

    if not np.any(
        np.abs(overlay.slow_weight) > 0
    ):
        raise AssertionError(
            "no slow consolidation occurred"
        )

    if not np.allclose(
        overlay.fast_weight,
        before * 0.98
    ):
        raise AssertionError(
            "fast decay mismatch"
        )

    print("P1 PLASTICITY PRIMITIVES: PASS")
    print(f"plastic_edges={positions.size}")
    print("target_budget_rms=0.010000")
    print(f"observed_budget_rms={rms:.6f}")
    print(
        "slow_nonzero="
        f"{np.count_nonzero(overlay.slow_weight)}"
    )


if __name__ == "__main__":
    main()
