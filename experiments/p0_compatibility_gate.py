"""P0: verify BioEngine v1 reproduces the historical MaleCNS reservoir."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioengine import GraphView, CompatReservoir, legacy_sign_vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--leak", type=float, default=0.2)
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--atol", type=float, default=2e-6)
    args = ap.parse_args()

    project = args.project.resolve()
    sys.path.insert(0, str(project))

    from loader import MaleCNSGraph
    from dynamics import ConnectomeReservoir

    base = MaleCNSGraph()
    A = sparse.load_npz(
        project / "adjacency_binary_real_v02.npz"
    ).tocsr().astype(np.float32)

    graph = GraphView(A=A, nodes=base.nodes)

    class LegacyGraphProxy:
        def __init__(self, gv: GraphView):
            self.A = gv.A
            self.nodes = gv.nodes
            self.in_strength = gv.in_strength
            self.out_strength = gv.out_strength

        @property
        def n_nodes(self):
            return self.A.shape[0]

        @property
        def n_edges(self):
            return self.A.nnz

    legacy_graph = LegacyGraphProxy(graph)
    sign = legacy_sign_vector(
        legacy_graph, sign_mode="biological_fast"
    )

    old = ConnectomeReservoir(
        legacy_graph,
        leak=args.leak,
        gain=args.gain,
        sign_mode="biological_fast",
    )
    new = CompatReservoir(
        graph,
        sign,
        leak=args.leak,
        gain=args.gain,
    )

    rng = np.random.default_rng(args.seed)
    max_abs = 0.0

    for t in range(args.steps):
        external = np.zeros(
            graph.n_nodes, dtype=np.float32
        )
        chosen = rng.choice(
            graph.n_nodes, size=64, replace=False
        )
        external[chosen] = rng.normal(
            0.0, 0.25, size=chosen.size
        ).astype(np.float32)

        old_state = np.asarray(
            old.step(external), dtype=np.float32
        ).copy()
        new_state = np.asarray(
            new.step(external), dtype=np.float32
        ).copy()

        err = float(
            np.max(np.abs(old_state - new_state))
        )
        max_abs = max(max_abs, err)

        if not np.allclose(
            old_state, new_state,
            atol=args.atol, rtol=0.0
        ):
            raise AssertionError(
                f"Compatibility gate FAILED at step {t}: "
                f"max_abs={err:.3e}"
            )

    vals, counts = np.unique(
        sign, return_counts=True
    )
    sign_counts = {
        float(k): int(v)
        for k, v in zip(vals, counts)
    }

    print("P0 COMPATIBILITY GATE: PASS")
    print(f"nodes={graph.n_nodes:,}")
    print(f"edges={graph.n_edges:,}")
    print(f"steps={args.steps}")
    print(f"max_abs_error={max_abs:.3e}")
    print(f"sign_counts={sign_counts}")


if __name__ == "__main__":
    main()
