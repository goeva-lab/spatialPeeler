"""
diagnostics-related utilties for spatialPeeler workflows
"""

import re
from collections.abc import Iterable, Sized
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import plotnine as p9
import plotnine.geoms.geom
from scipy.stats import mannwhitneyu

import apl


def mk_mask(adata: ad.AnnData, items: Iterable[tuple[apl.FetchBase, Any]]) -> np.ndarray:
    itr = iter(items)
    k, v = next(itr)
    assert not k.by_var
    mask = k.fetch(adata) == v
    for k, v in itr:
        assert not k.by_var
        mask &= k.fetch(adata) == v

    return mask


def run_deg(
    adata: ad.AnnData, g1: dict[apl.FetchBase, Any], g2: dict[apl.FetchBase, Any], layer: str | None = None
) -> pd.DataFrame:

    mtx = adata.X if layer is None else adata.layers[layer]

    lhs, rhs = mtx[mk_mask(adata, g1.items()), :], mtx[mk_mask(adata, g2.items()), :]  # ty:ignore[not-subscriptable]

    if isinstance(lhs, np.ndarray):
        lhs_mean, rhs_mean = np.mean(lhs, axis=0), np.mean(rhs, axis=0)  # ty:ignore[no-matching-overload]
    else:
        lhs_mean, rhs_mean = lhs.mean(axis=0), rhs.mean(axis=0)  # ty:ignore[unresolved-attribute]

    # do not compute values where either lhs/rhs is all zeroes
    mask = np.logical_and(lhs_mean > 0, rhs_mean > 0)

    lhs, rhs = lhs[:, mask], rhs[:, mask]  # ty:ignore[not-subscriptable, invalid-argument-type]

    if not isinstance(lhs, np.ndarray):
        lhs, rhs = lhs.todense(), rhs.todense()  # ty:ignore[unresolved-attribute]

    pvals = np.full((adata.n_vars,), np.nan)
    pvals[mask] = mannwhitneyu(lhs, rhs, axis=0).pvalue * adata.n_vars

    lfc = np.full((adata.n_vars,), np.nan)
    lfc[mask] = np.log2(rhs_mean[mask] / lhs_mean[mask])

    return pd.DataFrame({"adj_pval": pvals, "lfc": lfc}, index=adata.var.index)


CAPTURE_PAT = re.compile(r"\{(.+)\}")


def sweep_uns(adata: ad.AnnData, uns_pat: str) -> pd.DataFrame:
    return pd.concat(
        (
            pd.DataFrame(v, index=range(len(vl) if isinstance(vl := next(iter(v.values())), Sized) else 1)).assign(
                **r.groupdict()
            )
            for k, v in adata.uns.items()
            if (r := re.compile(CAPTURE_PAT.sub(r"(?P<\1>.*)", uns_pat)).search(k)) is not None
        ),
        ignore_index=True,
    )


def geom_point_sp(size_in_px: int = 1, dpi: float = 300.0, *args, **kwargs) -> Any:
    return [
        p9.geom_point(*args, **kwargs, shape="s", stroke=0, size=(size_in_px * 72.0 / dpi) ** 2),
        p9.coord_fixed(),
    ]


def comp_sparsity(mtx: np.ndarray, *args: np.ndarray | pd.Series, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame(mtx > 0)
    gbys = []
    for i, e in enumerate(args):
        if isinstance(e, pd.Series):
            k = e.name
            df[k] = e.array
        else:
            k = f"gby_{i}"
            df[k] = e
        gbys.append(k)
    for k, v in kwargs.items():
        df[k] = v.array if isinstance(v, pd.Series) else e
        gbys.append(str(k))

    for k in gbys:
        df[k] = pd.Categorical(df[k])

    df = df.melt(id_vars=gbys, var_name="column", value_name="sparsity")
    df["column"] = pd.Categorical(df["column"])

    return df.groupby(["column", *gbys]).agg(np.mean).reset_index(drop=False)
