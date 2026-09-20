from __future__ import annotations

from dataclasses import dataclass
import re
import numpy as np
import pandas as pd


# Compatibility map only. It reproduces the simplified sign convention used
# by the archived MaleCNS experiments; it is not a universal biological claim.
LEGACY_FAST_SIGN_MAP = {
    "ach": +1.0,
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "glu": -1.0,
    "histamine": -1.0,
    "his": -1.0,
    "dopamine": 0.0,
    "da": 0.0,
    "serotonin": 0.0,
    "5ht": 0.0,
    "5-ht": 0.0,
    "octopamine": 0.0,
    "oa": 0.0,
}

MODULATOR_CANONICAL = {
    "dopamine": "DA",
    "da": "DA",
    "serotonin": "5HT",
    "5ht": "5HT",
    "5-ht": "5HT",
    "octopamine": "OA",
    "oa": "OA",
}


@dataclass(frozen=True)
class NeurotransmitterProfile:
    """Per-neuron fast-transmission and neuromodulation channels."""

    fast_sign: np.ndarray
    dopamine: np.ndarray
    serotonin: np.ndarray
    octopamine: np.ndarray
    labels: np.ndarray

    @property
    def modulatory_mask(self) -> np.ndarray:
        return (
            (self.dopamine != 0)
            | (self.serotonin != 0)
            | (self.octopamine != 0)
        )


def _norm_label(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"[\s_]+", "", s)
    return s


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {re.sub(r"[\s_\-]+", "", c.lower()): c for c in df.columns}
    for c in candidates:
        key = re.sub(r"[\s_\-]+", "", c.lower())
        if key in normalized:
            return normalized[key]
    raise KeyError(
        "Could not infer required column. Tried: "
        + ", ".join(candidates)
        + f". Available columns: {list(df.columns)}"
    )


def build_neurotransmitter_profile(
    df: pd.DataFrame,
    *,
    n_nodes: int | None = None,
    index_column: str | None = None,
    transmitter_column: str | None = None,
    fast_sign_map: dict[str, float] | None = None,
) -> NeurotransmitterProfile:
    """Build separate fast-sign and modulatory channels from a neuron table.

    If index_column is supplied, values must be dense integer indices in
    [0, n_nodes). If omitted, DataFrame row order is used. For root_id-based
    files, map root_id to the graph's neuron order before calling this function.
    """

    if transmitter_column is None:
        transmitter_column = _find_column(
            df,
            (
                "neurotransmitter",
                "nt",
                "predicted_nt",
                "predicted_neurotransmitter",
                "cell_neurotransmitter",
            ),
        )

    if n_nodes is None:
        n_nodes = len(df)

    labels = np.full(n_nodes, "", dtype=object)
    fast = np.zeros(n_nodes, dtype=np.float32)
    da = np.zeros(n_nodes, dtype=np.float32)
    ht = np.zeros(n_nodes, dtype=np.float32)
    oa = np.zeros(n_nodes, dtype=np.float32)

    fmap = dict(LEGACY_FAST_SIGN_MAP if fast_sign_map is None else fast_sign_map)

    if index_column is None:
        indices = np.arange(len(df), dtype=np.int64)
    else:
        indices = pd.to_numeric(
            df[index_column], errors="raise"
        ).to_numpy(dtype=np.int64)

    if len(indices) and (indices.min() < 0 or indices.max() >= n_nodes):
        raise ValueError("index_column contains values outside [0, n_nodes)")

    for idx, raw in zip(indices, df[transmitter_column].to_numpy(), strict=True):
        label = _norm_label(raw)
        labels[idx] = label
        fast[idx] = np.float32(fmap.get(label, 0.0))

        mod = MODULATOR_CANONICAL.get(label)
        if mod == "DA":
            da[idx] = 1.0
        elif mod == "5HT":
            ht[idx] = 1.0
        elif mod == "OA":
            oa[idx] = 1.0

    return NeurotransmitterProfile(
        fast_sign=fast,
        dopamine=da,
        serotonin=ht,
        octopamine=oa,
        labels=labels,
    )
