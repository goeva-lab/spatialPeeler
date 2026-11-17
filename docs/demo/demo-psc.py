# ---
# jupyter:
#   kernelspec:
#     display_name: spatialPeeler
#     language: python
#     name: spatialpeeler
# ---
# %% [markdown]
# # analysis of Primary Sclerosing Cholangitis (PSC) data
# %% [markdown]
# we first import all script dependencies and load data into an {py:class}`~anndata.AnnData` object
# %% tags=["hide-input"] mystnb={"code_prompt_show": "show preamble"}
from functools import partial
from pathlib import Path
from warnings import catch_warnings, simplefilter

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import uapl
import upsetplot as up
from matplotlib import pyplot as plt
from plotnine import *

import spatialpeeler as sP
from spatialpeeler.diag import comp_spat_stats, sweep_uns

from _shared import comp_sparsity, pl_k_sP, pl_spat_stats, psc_ds, run_deg_groups, run_nmf, run_proc_per_ds, viz_defs  # isort: skip

viz_defs(dpi=128)
RAND_SEED = 28
# define helper function for plotting
pl_help = partial(pl_k_sP, cond_col="group", sample_col="sample_id", ctl_val="CTL")
SP_FACT_PAT_STR = "sP_p_hat_k={k}"
SP_GENE_PAT_STR = "sp_gene_id_k={k}"

# load dataset + run NMF
if (pth := Path("../.cache/psc.h5ad")).exists():
    psc = ad.read_h5ad(pth)
else:
    adatas = psc_ds(Path("../../data/psc-visium"))
    for k in adatas:
        adatas[k] = adatas[k][adatas[k].obs["in_tissue"], :]

    run_proc_per_ds(adatas)

    psc = ad.concat(adatas, label="sample_id", index_unique="_")

    sc.pp.filter_genes(psc, min_cells=psc.n_obs // 500)  # ty: ignore[too-many-positional-arguments]

    psc.var["SYMBOL"] = (
        sc.queries.biomart_annotations("hsapiens", ["ensembl_gene_id", "external_gene_name"], host="grch37.ensembl.org")
        .set_index("ensembl_gene_id")
        .merge(psc.var.index.to_series(), left_index=True, right_index=True, how="right")["external_gene_name"]
        .fillna(psc.var.index.to_series())
    )

    run_nmf(psc, 30, RAND_SEED)

    pth.parent.mkdir(exist_ok=True)
    psc.write_h5ad(pth)

assert isinstance(psc.obs, pd.DataFrame)
print(psc)

# %% [markdown]
# with our processed data, we proceed with the factor-wise regression step of spatialPeeler
# %%
X_spars = psc.X.nnz / np.prod(psc.X.shape)  # ty: ignore[missing-union-attribute]
(
    comp_sparsity(psc.obsm["X_nmf"], cond=psc.obs["group"])
    >> ggplot()
    + aes(x="density", y="column", color="cond")
    + geom_point()
    + scale_y_discrete(limits=reversed)
    + labs(
        x="fraction of non-zero observations",
        y="NMF factor",
        title=f"fraction of non-zero observations in count matrix: {X_spars * 100:.02f}%",
    )
).show()

# %%
# run factor-wise regression step
for k in range(psc.obsm["X_nmf"].shape[1]):
    sP.run_fact_reg(psc, "group", "CTL", k, SP_FACT_PAT_STR.format(k=k), X_key="X_nmf")

print(psc)

# %% [markdown]
# we now assess our resulting regression outputs distributions \
# we can extract and collate the model betas and other statistics using the `sweep_uns` function,
# and compute spatial statistics using the `comp_spat_stats` function.
#
# we can then use these outputs to plot the difference in mean Moran's I statistic between case and control samples
# (with betas plotted on a symmetric log scale to improve resolution for small betas):
# %%
beta_df = sweep_uns(psc, SP_FACT_PAT_STR)
beta_df["k"] = beta_df["k"].astype(int)
sp_stat_df = comp_spat_stats(psc, SP_FACT_PAT_STR, "sample_id")

(
    pl_spat_stats(beta_df, sp_stat_df[sp_stat_df["stat"] == "moran's i"], psc.obs, "sample_id", "group", "CTL")
    + scale_y_symlog()
    + theme(aspect_ratio=2)
).show()


# %% [markdown]
# we can see that factor 4 has a strong case-specific profile (positive beta) and shows noticeable differences
# between case and control in its spatial statistics, so we plot its distributions below:
# %%
for p in pl_help(psc, 4).values():
    p.show()


# %% [markdown]
# we can see that factor 6 has a strong control-specific profile (negative beta) and shows noticeable differences
# between case and control in its spatial statistics, so we plot its distributions below:
# %%
for p in pl_help(psc, 6).values():
    p.show()


# %% [markdown]
# to explore this further, we proceed w/ a gene-wise analysis for factor 4, which has the largest beta:
# %%
ks_oi = [4]
for k in ks_oi:
    sP.run_gene_reg(psc, SP_FACT_PAT_STR.format(k=k), SP_GENE_PAT_STR.format(k=k), layer="counts", run_reg=False)
# %%
for k in ks_oi:
    (
        uapl.plotfeats(psc)
        + aes(
            x=uapl.varm("X_nmf", k),
            y=uapl.varm(SP_GENE_PAT_STR.format(k=k), "corr"),
            label=np.where(
                np.argsort(np.argsort(-np.abs(np.asarray(psc.varm[SP_GENE_PAT_STR.format(k=k)]["corr"])))) < 8,
                psc.var["SYMBOL"],
                None,
            ),
        )
        + geom_point(size=1.5, shape="o", stroke=0)
        + geom_label(adjust_text={}, na_rm=True)
        + labs(
            colour="-log(pval)",
            x=f"NMF[{k}]",
            y="pearson correlation",
            title=f"NMF loading / p\u0302-gene correlation relationship for k={k}",
        )
    ).show()
# %%
degs = {k: run_deg_groups(psc, k, "group", "CTL") for k in ks_oi}
sig_sub = {k: degs[k].loc[lambda e: (e["adj_pval"] < 0.05) & (e["lfc"].abs() > 1)] for k in ks_oi}

# %%
for k in ks_oi:
    sig_counts = sig_sub[k].groupby("cmp", observed=True)["gene"].count()
    (
        ggplot(sig_counts.reset_index(drop=False))
        + aes(x="cmp", y="gene", fill=after_stat("y"), label=after_stat("y"))
        + geom_col(color="black")
        + geom_text(nudge_y=sig_counts.max() * 0.025)
        + theme(axis_text_x=element_text(angle=45, hjust=1))
        + guides(fill="none")
        + labs(x="", y="# of DEGs (adj_pval < 0.05 & |lfc| > 1)", title=f"DEG counts for k={k}")
    ).show()

# %% [markdown]
# we can further visualize the permutations of gene set size in these comparisons via the `upsetplot` package:
# %%
for k in ks_oi:
    with catch_warnings():
        simplefilter(action="ignore", category=FutureWarning)
        up.UpSet(
            up.from_contents({k: v.array for k, v in sig_sub[k].groupby("cmp", observed=True)["gene"]}),
            subset_size="count",
            max_subset_rank=8,
        ).plot()

    plt.show()
