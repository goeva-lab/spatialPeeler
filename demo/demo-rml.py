# ruff: noqa: F405
# %%
from pathlib import Path
from pickle import dump, load

import anndata as ad
import numpy as np
import pandas as pd
from ingest import remyelin_ds, run_nmf, run_proc_per_ds
from plotnine import *  # noqa: F403

import apl
import spatialpeeler as sP
from spatialpeeler.diag import comp_sparsity, geom_point_sp, run_deg, sweep_uns

# %%
RAND_SEED = 28

# %%
# load dataset + run NMF
if (pth := Path("../.cache/rml.pkl")).exists():
    with open(pth, "rb") as f:
        rml = load(f)
else:
    adatas = remyelin_ds(
        Path("../data/remyelin-slideseq"),
        # load only subset of interest
        lambda obs: (
            (obs["GoodQuality"] == "Yes")
            & (
                ((obs["Timepoint"].isin([3])) & (obs["Condition"] == "LPC"))
                | (obs["Timepoint"].isin([12, 18]) & (obs["Condition"] == "Saline"))
            )
        ),
    )
    run_proc_per_ds(adatas)

    rml = ad.concat(adatas, label="puck_id", index_unique="_")
    rml.obs["puck_id_cond"] = rml.obs["puck_id"].str.cat(rml.obs["Condition"], sep="_")  # ty:ignore[unresolved-attribute]

    run_nmf(rml, 30, RAND_SEED)

    with open(pth, "wb") as f:
        dump(rml, f)
print(rml)

# %% [markdown]
# before proceeding with spatialPeeler, we inspect our count matrix and NMF factors to assess their sparsity
# %%
X_spars = rml.X.nnz / np.prod(rml.X.shape)  # ty:ignore[unresolved-attribute]
(
    comp_sparsity(rml.obsm["X_nmf"], cond=rml.obs["Condition"])
    >> ggplot()
    + aes(x="column", colour="cond", y="sparsity")
    + geom_point()
    + labs(x="NMF factor", title=f"count sparsity: {X_spars * 100:.02f}%")
).show()

# %% [markdown]
# we see that the dataset is overall quite sparse (only ~1% of the data is non zero)
# and this is reflected in the factors, which are also mostly quite sparse
#
# as such, we threshold our factor regression analysis to only include cells which are non-sparse for that factor
# %%
# run factor-wise regression step
for k in range(rml.obsm["X_nmf"].shape[1]):
    sP.run_fact_reg(
        rml,
        "Condition",
        "Saline",
        k,
        f"sp_p_hat_k={k:02}",
        X_key="X_nmf",
        filt=lambda adata: adata[adata.obsm["X_nmf"][:, k] > 0, :],
    )

print(rml)

# %% [markdown]
# we now assess our resulting regression outputs distributions
# we can extract and collate the model betas and other statistics using the sweep_uns function
# %%
(
    sweep_uns(rml, "sp_p_hat_k={k}")
    >> ggplot()
    + aes(x="reorder(k, -beta)", fill="-np.log(pval)", y="beta")
    + geom_col()
    + labs(fill="-log(pval)")
    + scale_fill_cmap("turbo")
).show()

# %% [markdown]
# we can see that factor 6 has particularly high beta value and significance,
# so we can plot the spatial distribution of the factor and the pattern
# %%
rml_filt = rml[rml.obsm["X_nmf"][:, 6] > 0, :]
plo = apl.PlotAdata(rml_filt)
pl1a = plo + aes(colour=apl.obsm("X_nmf", 6)) + labs(colour="NMF[6]") + scale_colour_cmap("YlGnBu")
pl1b = (
    plo + aes(colour=apl.obsm("sp_p_hat_k=06", sP.PAT_KEY)) + labs(colour="p_hat") + scale_colour_cmap("YlGnBu", limits=(0, 1))
)
pl1 = (
    (pl1a / pl1b)
    & aes(x=apl.obsm("spatial", "x"), y=apl.obsm("spatial", "y"))
    & geom_point_sp(size_in_px=2, alpha=0.5)
    & facet_wrap(apl.obs("puck_id_cond"))
    & theme(axis_text=element_blank(), axis_ticks=element_blank(), strip_text=element_blank())
)
pl2 = (
    plo
    + aes(x=apl.obsm("X_nmf", 6), y=apl.obsm("sp_p_hat_k=06", sP.PAT_KEY), colour=apl.obs("Condition"))
    + geom_point(alpha=0.5, shape=".")
    + labs(y="p_hat")
    + scale_colour_discrete()
)
(pl1 | pl2).show()

# %%[markdown]
# we can observe that the non-linearity of the logistic regression probability outputs allows
# for strong extraction of the disease effect, which is not as visible with just the NMF factor on its own

# %%[markdown]
# we can also use the run_deg function provided in the sP.diag module to assess DEGs between various labels / combinations of labels
# %%
case1_vs_case0 = run_deg(
    rml_filt,
    {apl.obsm("sp_p_hat_k=06", sP.BIN_KEY): True},
    {apl.obs("Condition"): "LPC", apl.obsm("sp_p_hat_k=06", sP.BIN_KEY): False},
)
case1_vs_ctl = run_deg(
    rml_filt,
    {apl.obsm("sp_p_hat_k=06", sP.BIN_KEY): True},
    {apl.obs("Condition"): "Saline"},
)
case_vs_ctl = run_deg(
    rml_filt,
    {apl.obs("Condition"): "LPC"},
    {apl.obs("Condition"): "Saline"},
)

print(
    pd.concat(
        {
            "case1_vs_case0": (case1_vs_case0["adj_pval"] < 0.05).value_counts(),
            "case1_vs_ctl": (case1_vs_ctl["adj_pval"] < 0.05).value_counts(),
            "case_vs_ctl": (case_vs_ctl["adj_pval"] < 0.05).value_counts(),
        }
    )
)

# %%[markdown]
# we can also assess the distributions of p_hat values across the binarized labels:
# %%
(
    plo
    + aes(
        x=apl.obs("Condition"),
        y=apl.obsm("sp_p_hat_k=06", sP.PAT_KEY),
        fill=apl.obsm("sp_p_hat_k=06", sP.BIN_KEY),
    )
    + geom_violin()
    + labs(fill="HiDDEN - affected")
    + lims(y=(0, 1))
).show()

# %%[markdown]
# given factor 6 displays interesting properties, we proceed with gene-wise analysis for factor 6
# %%
ks_oi = [6]
for k in ks_oi:
    sP.run_gene_reg(rml, f"sp_p_hat_k={k:02}", f"sp_gene_id_k={k:02}", layer="counts")
