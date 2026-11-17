"""
diagnostics-related utilties for spatialPeeler workflows
"""

import re
from collections.abc import Callable, Iterable, Sized
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import uapl
from scipy.sparse import sparray, spmatrix
from scipy.stats import mannwhitneyu

from .peel import PAT_KEY

CAPTURE_PAT = re.compile(r"\{(.+)\}")


def sweep_uns(adata: ad.AnnData, uns_pat: str) -> pd.DataFrame:
    """
    collects all .uns values matching w/ keys matching a provided pattern

    :param adata: input object w/ ``.uns`` slots matching pattern of interest
    :param uns_pat: f-string like template providing pattern to match against
    """
    return pd.concat(
        (
            pd.DataFrame(
                v,
                index=range(len(vl) if isinstance(vl := next(iter(v.values())), Sized) else 1),
            ).assign(**r.groupdict())
            for k, v in adata.uns.items()
            if (r := re.compile(CAPTURE_PAT.sub(r"(?P<\1>.*)", uns_pat)).search(k)) is not None
        ),
        ignore_index=True,
    )


def comp_spat_stats(
    adata: ad.AnnData,
    sP_key: str | Callable[[int], str],
    sample_col: str,
    spatial_key: str = "spatial",
    range_src: str | Iterable[int] = "X_nmf",
) -> pd.DataFrame:
    """
    computes spatial statistics (Moran's I and Geary's C) for a given spatial view per sample

    :param adata: input object containing a spatial FOV of interest
    :param sP_key: `.obsm` key pattern (with placeholder `k`) used for run_fact_reg output, or a function which given a factor index returns corresponding column name
    :param sample_col: `.obs` key for sample column
    :param spatial_key: `.obsm` key for spatial information
    :param obsm_src: `.obsm` key for embedding used as `run_fact_reg` input or an iterator providing factor indices of interest
    """
    if isinstance(sP_key, str):
        fmt_str = sP_key
        sP_key = lambda k: fmt_str.format(k=k)

    if isinstance(range_src, str):
        k_range = range(adata.obsm[range_src].shape[1])
    else:
        k_range = [e for e in range_src]  # copy into list to avoid weird iterable behaviours

    out = {"k": [], "stat": [], "sample": [], "val": []}
    for sample, sub in adata.obs.groupby(sample_col, observed=True).indices.items():  # ty: ignore[missing-union-attribute]
        sub = adata[sub, :].copy()
        sc.pp.neighbors(sub, use=ad.acc.A.obsm[spatial_key])  # ty: ignore[invalid-argument-type]
        for k in k_range:
            k_pat = sub.obsm[sP_key(k)][PAT_KEY]

            for stat, stat_fn in [
                ("moran's i", sc.metrics.morans_i),
                ("geary's c", sc.metrics.gearys_c),
            ]:
                out["k"].append(k)
                out["stat"].append(stat)
                out["sample"].append(sample)

                na_filt = np.asarray(k_pat.notna())  # ty: ignore[missing-union-attribute]
                out["val"].append(stat_fn(sub[na_filt, :], k_pat.loc[na_filt]))  # ty: ignore[missing-union-attribute]

    return pd.DataFrame(out)


def mk_mask(adata: ad.AnnData, items: Iterable[tuple[uapl.pl.FetchObsWise, Any]]) -> np.ndarray:
    itr = iter(items)
    k, v = next(itr)
    mask = k.fetch(adata) == v
    for k, v in itr:
        mask &= k.fetch(adata) == v

    return mask


def run_deg_mwu(
    adata: ad.AnnData,
    g1: dict[uapl.pl.FetchObsWise, Any],
    g2: dict[uapl.pl.FetchObsWise, Any] | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """
    runs mann-whitney-U-test based DEG analysis between a set of conditions

    :param adata: input anndata
    :param g1: first condition, specified as a dictionary of source to expected value bindings
    :param g2: second condition, can be unspecified to compare against all remaining cells in the dataset
    :param layer: source layer to use to extract gene-wise information (if unset adata.X is used)
    """

    mtx = adata.X if layer is None else adata.layers[layer]

    lhs_mask = mk_mask(adata, g1.items())
    lhs = mtx[lhs_mask, :]  # ty: ignore[not-subscriptable]
    rhs_mask = ~lhs_mask if g2 is None else mk_mask(adata, g2.items())
    rhs = mtx[rhs_mask, :]  # ty: ignore[not-subscriptable]

    if isinstance(lhs, np.ndarray):
        lhs_mean, rhs_mean = np.mean(lhs, axis=0), np.mean(rhs, axis=0)  # ty: ignore[no-matching-overload]
    else:
        lhs_mean, rhs_mean = lhs.mean(axis=0), rhs.mean(axis=0)  # ty: ignore[missing-union-attribute]
        if len(lhs_mean.shape) > 1:
            lhs_mean = np.ravel(lhs_mean)
        if len(rhs_mean.shape) > 1:
            rhs_mean = np.ravel(rhs_mean)

    # do not compute values where either lhs/rhs is all zeroes
    mask = np.logical_and(lhs_mean > 0, rhs_mean > 0)

    pvals = np.full((adata.n_vars,), np.nan)
    if isinstance(mtx, sparray | spmatrix):
        idx_arr = np.arange(adata.n_vars)[mask]
        batch_n = max(1, 125000000 // adata.n_obs)  # 125e6 ~= 1GB / 64bits, trying not to alloc more than a gig at a time
        for start in range(0, idx_arr.size, batch_n):
            slc = idx_arr[start : min(adata.n_vars, start + batch_n)]
            lhs_i, rhs_i = lhs[:, slc].todense(), rhs[:, slc].todense()  # ty: ignore[missing-union-attribute, not-subscriptable, invalid-argument-type]
            pvals[slc] = mannwhitneyu(lhs_i, rhs_i).pvalue * adata.n_vars
            del lhs_i, rhs_i
    else:
        pvals[mask] = mannwhitneyu(lhs[:, mask], rhs[:, mask]).pvalue * adata.n_vars  # ty: ignore[not-subscriptable, invalid-argument-type]

    lfc = np.full((adata.n_vars,), np.nan)
    lfc[mask] = np.log2(lhs_mean[mask] / rhs_mean[mask])

    return pd.DataFrame({"adj_pval": pvals, "lfc": lfc}, index=adata.var.index)
