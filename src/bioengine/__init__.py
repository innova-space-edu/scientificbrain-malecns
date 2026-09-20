"""MaleCNS BioEngine v1.

Compatibility-first biological simulation primitives for the MaleCNS project.
"""

from .core import GraphView, CompatReservoir, legacy_sign_vector
from .modulation import (
    NeurotransmitterProfile,
    build_neurotransmitter_profile,
    LEGACY_FAST_SIGN_MAP,
)
from .plasticity import (
    PlasticityOverlay,
    EligibilityTrace,
    ThreeFactorRule,
    FastSlowConsolidation,
    normalize_plastic_budget,
    select_edges_to_postsynaptic_nodes,
)

__all__ = [
    "GraphView",
    "CompatReservoir",
    "legacy_sign_vector",
    "NeurotransmitterProfile",
    "build_neurotransmitter_profile",
    "LEGACY_FAST_SIGN_MAP",
    "PlasticityOverlay",
    "EligibilityTrace",
    "ThreeFactorRule",
    "FastSlowConsolidation",
    "normalize_plastic_budget",
    "select_edges_to_postsynaptic_nodes",
]
