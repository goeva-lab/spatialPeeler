from collections.abc import Callable
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import scanpy as sc
from jax import jit
from jax import numpy as jnp
from scipy import sparse as sp
from scipy.cluster.vq import kmeans2
from scipy.special import xlogy
from scipy.stats import zscore
from statsmodels import api as sm
from statsmodels.base.model import Model, Results

PAT_KEY = "sP_pat"
BIN_KEY = "sP_bin"
INP_KEY = "sP_inp"


def pearsonr_w(x: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray):
    x_min_mu = x - jnp.average(x, weights=w)
    y_min_mu = y - jnp.average(y, weights=w)
    x_var = jnp.average(jnp.square(x_min_mu), weights=w)
    y_var = jnp.average(jnp.square(y_min_mu), weights=w)

    return jnp.average(x_min_mu * y_min_mu, weights=w) / jnp.sqrt(x_var * y_var)


def pearsonr(x: jnp.ndarray, y: jnp.ndarray):
    x_min_mu = x - jnp.mean(x)
    y_min_mu = y - jnp.mean(y)
    x_var = jnp.mean(jnp.square(x_min_mu))
    y_var = jnp.mean(jnp.square(y_min_mu))

    return jnp.mean(x_min_mu * y_min_mu) / jnp.sqrt(x_var * y_var)


def corr_feats(X: sp.csc_array, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    y: jnp.ndarray = jnp.asarray(y)

    if w is None:
        f = jit(lambda x: pearsonr(x, y))
    else:
        w: jnp.ndarray = jnp.asarray(w)
        f = jit(lambda x: pearsonr_w(x, y, w))

    return np.fromiter((f(jnp.asarray(X[:, j].toarray())) for j in range(X.shape[1])), dtype=float, count=X.shape[1])


def fit_sm(X: np.ndarray, y: np.ndarray, with_intercept: bool, model: type[Model]) -> Results:
    if with_intercept:
        X = np.vstack([X, np.ones((X.shape[0],))]).T
    with catch_warnings():
        simplefilter(action="ignore", category=RuntimeWarning)
        return model(y, X).fit(disp=False)  # ty:ignore[unknown-argument]


def run_bin(x: np.ndarray, init_mask: np.ndarray) -> np.ndarray:
    cents, clusts = kmeans2(x[init_mask], 2, minit="++")
    clusts = clusts.astype(bool)
    if cents[0] > cents[1]:
        clusts = ~clusts

    out = np.copy(init_mask).astype(bool)
    out[out] = clusts
    # relabel controls according to whichever centroid they are closer
    # to according to the clustering of the case-only data
    out[~out] = np.where(x[~out] >= np.mean(cents), True, False)
    return out


def run_fact_reg(
    adata: sc.AnnData,
    y_key: str,
    ctl_val: object,
    k: int,
    out_key: str,
    X_key: str = "X_nmf",
    filt: Callable[[sc.AnnData], sc.AnnData] | None = None,
) -> None:
    """
    runs the first step of the spatialPeeler process by:
    - logistic regression on condition labels w/ an embedding subspace + intercept as model inputs
    - adding per cell outputs to input ``.obsm`` slot
    - adding model coefficients, p-values, mean standard error, and cross-entropy loss values to input ``.uns`` slot

    :param adata: input data, must not be a view, will be mutated
    :param y_key: ``.obsm`` key to fetch condition labels from
    :param ctl_val: condition label corresponding to control
    :param k: dimension of embedding to analyze
    :param out_key: key used to insert outputs into input ``adata``, will be added to both ``.uns`` and ``.obsm``
    :param X_key: adata ``.obsm`` key to fetch embeddings from, defaults to ``"X_nmf"``
    :param filt: optional callable to subset data prior to running step, defaults to ``None``

    :raises ValueError: if input data is a view
    """
    if adata.is_view:
        raise ValueError(
            "input anndata object _must not_ be a view, "
            "use `filt` argument to run analysis on subset of data"
            " (this function mutates its input directly)"
        )

    if filt is None:
        mod = fit_sm(
            adata.obsm[X_key][:, k],
            (adata.obs[y_key] != ctl_val).to_numpy(),
            True,
            sm.Logit,
        )
        y, pat_vec = mod.model.endog, mod.predict(mod.model.exog)
        adata.obsm[out_key] = pd.DataFrame(
            {
                "cell": adata.obs_names,
                PAT_KEY: pat_vec,
                BIN_KEY: run_bin(pat_vec, y),
                INP_KEY: mod.model.exog[:, 0],
            }
        ).set_index("cell")
    else:
        idx = adata.obs.index.get_indexer_for(filt(adata).obs.index)
        adata_s = adata[idx, :]
        mod = fit_sm(
            adata_s.obsm[X_key][:, k],
            (adata_s.obs[y_key] != ctl_val).to_numpy(),
            True,
            sm.Logit,
        )
        y, pat_vec = mod.model.endog, mod.predict(mod.model.exog)
        # @TODO: find a better way to construct a SparseIndex (???)
        sp_index = type(pd.arrays.SparseArray([]).sp_index)(adata.n_obs, idx)  # ty:ignore[too-many-positional-arguments]
        adata.obsm[out_key] = pd.DataFrame(
            {
                "cell": adata.obs_names,
                PAT_KEY: pd.arrays.SparseArray(pat_vec, sp_index, fill_value=np.nan),
                BIN_KEY: pd.arrays.SparseArray(run_bin(pat_vec, y), sp_index, fill_value=np.nan),
                INP_KEY: pd.arrays.SparseArray(mod.model.exog[:, 0], sp_index, fill_value=np.nan),
            }
        ).set_index("cell")

    adata.uns[out_key] = {
        "beta": mod.params[0],
        "pval": mod.pvalues[0],  # ty:ignore[unresolved-attribute]
        "mse": np.sum(np.square(y - pat_vec)),
        "loss": np.sum(-(xlogy(y, pat_vec) + xlogy(1 - y, 1 - pat_vec))),
    }


def run_gene_reg(
    adata: sc.AnnData,
    sp_step1_key: str,
    out_key: str,
    run_reg: bool = True,
    run_corr: bool = True,
    layer: str | None = None,
) -> None:
    """
    runs the second step of the spatialPeeler process by:
    - compute correlations between spatialPeeler outputs and gene expression
    - compute regression-based (via OLS) differential expression analyses between spatialPeeler outputs and gene expression
    - adding resulting outputs to the input ``.varm`` slot

    :param adata: input data, already processed by :func:`run_fact_reg`, must not be a view, will be mutated
    :param sp_step1_key: ``.obsm`` key used by :func:`run_fact_reg` to store its outputs
    :param out_key: ``.varm`` key used to store outputs
    :param layer: optional, layer to use to fetch gene counts instead of ``.X``, defaults to ``None``

    :raises ValueError: if input adata is a view
    """
    assert sum([run_reg, run_corr]) > 0, "at least one form of analysis must be enabled"
    if adata.is_view:
        raise ValueError("input anndata object _must not_ be a view (this function mutates its input directly)")

    X = adata.X if layer is None else adata.layers[layer]

    sp_out = adata.obsm[sp_step1_key]
    w_vec = sp_out[INP_KEY].array  # ty:ignore[unresolved-attribute]
    pat_vec = sp_out[PAT_KEY].array  # ty:ignore[unresolved-attribute]

    if isinstance(w_vec, pd.arrays.SparseArray):
        assert isinstance(pat_vec, pd.arrays.SparseArray)
        assert w_vec.sp_index is pat_vec.sp_index
        X = X[w_vec.sp_index.indices, :]  # ty:ignore[not-subscriptable]
        w_vec, pat_vec = w_vec.sp_values, pat_vec.sp_values
    else:
        w_vec, pat_vec = np.asarray(w_vec), np.asarray(pat_vec)

    out: dict = {"gene": adata.var_names}

    if run_corr:
        X_csc = sp.csc_array(X)
        out["corr"] = corr_feats(X_csc, pat_vec)
        out["wcorr"] = corr_feats(X_csc, pat_vec, w_vec)

    if run_reg:
        z_pat_vec = zscore(pat_vec)
        out_keys = ("beta", "pval", "rsq")
        preg_arrs = (
            np.full((adata.n_vars,), np.nan),
            np.full((adata.n_vars,), np.nan),
            np.full((adata.n_vars,), np.nan),
        )
        zpreg_arrs = (
            np.full((adata.n_vars,), np.nan),
            np.full((adata.n_vars,), np.nan),
            np.full((adata.n_vars,), np.nan),
        )
        for j in range(X.shape[1]):  # ty: ignore[unresolved-attribute]
            gvec = X[:, j]  # ty: ignore[not-subscriptable]
            if not isinstance(gvec, np.ndarray):
                gvec = np.ravel(gvec.toarray())  # ty: ignore[unresolved-attribute]

            preg = fit_sm(pat_vec, gvec, True, sm.OLS)
            preg_arrs[0][j] = preg.params[0]
            preg_arrs[1][j] = preg.pvalues[0] * adata.n_vars  # ty:ignore[unresolved-attribute]
            with catch_warnings():
                simplefilter(action="ignore", category=RuntimeWarning)
                preg_arrs[2][j] = preg.rsquared  # ty:ignore[unresolved-attribute]

            z_gvec = zscore(gvec)
            z_index = np.nonzero(np.isfinite(z_gvec))[0]
            if z_index.size > 0:
                zpreg = fit_sm(z_pat_vec, z_gvec, True, sm.OLS)
                zpreg_arrs[0][j] = zpreg.params[0]
                zpreg_arrs[1][j] = zpreg.pvalues[0] * adata.n_vars  # ty:ignore[unresolved-attribute]
                with catch_warnings():
                    simplefilter(action="ignore", category=RuntimeWarning)
                    zpreg_arrs[2][j] = zpreg.rsquared  # ty:ignore[unresolved-attribute]
        out |= dict(zip((f"reg_{e}" for e in out_keys), preg_arrs))
        out |= dict(zip((f"zreg_{e}" for e in out_keys), zpreg_arrs))

    adata.varm[out_key] = pd.DataFrame(out).set_index("gene")
