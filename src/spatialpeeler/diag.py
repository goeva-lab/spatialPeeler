"""
diagnostics-related utilties for spatialPeeler workflows
"""

import re
from collections.abc import Iterable, Sized
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import uapl
from scipy.sparse import sparray, spmatrix
from scipy.stats import mannwhitneyu


def mk_mask(adata: ad.AnnData, items: Iterable[tuple[uapl.FetchBase, Any]]) -> np.ndarray:
    itr = iter(items)
    k, v = next(itr)
    assert not k.by_var
    mask = k.fetch(adata) == v
    for k, v in itr:
        assert not k.by_var
        mask &= k.fetch(adata) == v

    return mask


def run_deg_mwu(
    adata: ad.AnnData,
    g1: dict[uapl.FetchBase, Any],
    g2: dict[uapl.FetchBase, Any] | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """
    runs mann-whitney-U-test-based DEG tests between a set of conditions, returning the results in a dataframe

    :param adata: input anndata
    :param g1: first condition, specified as a dictionary of source -> expected value bindings
    :param g2: second condition, can be unspecified to compare against all remaining cells in the dataset
    :param layer: source layer to use to extract gene-wise information (if unset adata.X is used)
    """

    mtx = adata.X if layer is None else adata.layers[layer]

    lhs_mask = mk_mask(adata, g1.items())
    lhs = mtx[lhs_mask, :]  # ty:ignore[not-subscriptable]
    rhs_mask = ~lhs_mask if g2 is None else mk_mask(adata, g2.items())
    rhs = mtx[rhs_mask, :]  # ty:ignore[not-subscriptable]

    if isinstance(lhs, np.ndarray):
        lhs_mean, rhs_mean = np.mean(lhs, axis=0), np.mean(rhs, axis=0)  # ty:ignore[no-matching-overload]
    else:
        lhs_mean, rhs_mean = lhs.mean(axis=0), rhs.mean(axis=0)  # ty:ignore[unresolved-attribute]
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
            lhs_i, rhs_i = lhs[:, slc].todense(), rhs[:, slc].todense()  # ty:ignore[unresolved-attribute, not-subscriptable]
            pvals[slc] = mannwhitneyu(lhs_i, rhs_i).pvalue * adata.n_vars
            del lhs_i, rhs_i
    else:
        pvals[mask] = mannwhitneyu(lhs[:, mask], rhs[:, mask]).pvalue * adata.n_vars  # ty:ignore[not-subscriptable]

    lfc = np.full((adata.n_vars,), np.nan)
    lfc[mask] = np.log2(lhs_mean[mask] / rhs_mean[mask])

    return pd.DataFrame({"adj_pval": pvals, "lfc": lfc}, index=adata.var.index)


CAPTURE_PAT = re.compile(r"\{(.+)\}")


def sweep_uns(adata: ad.AnnData, uns_pat: str) -> pd.DataFrame:
    """
    collects all .uns values matching w/ keys matching a provided pattern into a single dataframe

    :param adata:
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


def comp_sparsity(mtx, *args: np.ndarray | pd.Series, **kwargs) -> pd.DataFrame:
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

    df = df.melt(id_vars=gbys, var_name="column", value_name="density")
    df["column"] = pd.Categorical(df["column"])

    return df.groupby(["column", *gbys], observed=True).agg("mean").reset_index(drop=False)
