from collections.abc import Callable, Mapping
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
from plotnine import theme_matplotlib, theme_set, theme_update
from scipy.io import mmread
from sklearn.decomposition import NMF

import apl
import spatialpeeler as sP


def run_proc_per_ds(adatas: Mapping[str, sc.AnnData]) -> None:
    # scale by sample
    for adata in adatas.values():
        adata.raw = adata
        adata.layers["counts"] = adata.X  # ty:ignore[invalid-assignment]
        sc.pp.scale(adata, zero_center=False)


def run_nmf(adata: sc.AnnData, k: int, seed: int | None = None, key: str = "X_nmf") -> None:
    nmf_mod = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=1000)
    adata.obsm[key] = nmf_mod.fit_transform(adata.X)
    adata.varm[key] = nmf_mod.components_.T


def load_remyelin_ds(ds_path: Path, **kwargs) -> ad.AnnData:
    obs = pd.read_csv(
        next(ds_path.glob("*_barcodes.tsv")),
        delimiter="\t",
        header=None,
        names=["cell"],
    ).set_index("cell")

    adata = ad.AnnData(
        mmread(next(ds_path.glob("*.mtx")), spmatrix=True).tocsc().T,
        var=pd.read_csv(
            next(ds_path.glob("*_features.tsv")),
            delimiter="\t",
            header=None,
            names=["gene", "SYMBOL"],
        ).set_index("gene"),
        obs=obs,
        obsm={
            "spatial": pd.read_csv(
                next((ds_path / "barcode_matching").glob("*_xy.txt")),
                delimiter="\t",
                header=None,
                names=["cell", "x", "y"],
            )
            .set_index("cell")
            .loc[obs.index]
        },
    )

    for k, v in kwargs.items():
        adata.obs[k] = v

    return adata


def remyelin_ds(pth: Path, filt_fn: Callable[[pd.DataFrame], pd.Series] | None = None) -> dict[str, ad.AnnData]:
    meta = pd.read_csv(pth / "batch_key.csv")
    if filt_fn is not None:
        meta = meta.loc[filt_fn]

    return {
        row["PuckID"]: load_remyelin_ds(
            pth / row["Directory name"],
            **(
                row[
                    [
                        "Sample details",
                        "Animal",
                        "Timepoint",
                        "Condition",
                        "GoodQuality",
                    ]
                ]
            ),
        )
        for _, row in meta.iterrows()
    }


def load_psc_ds(ds_path: Path, **kwargs) -> ad.AnnData:
    c_idx = (
        pd.read_csv(
            ds_path / "barcodes.tsv.gz",
            delimiter="\t",
            header=None,
            names=["cell"],
        )
        .set_index("cell")
        .index
    )

    adata = ad.AnnData(
        mmread(ds_path / "matrix.mtx.gz", spmatrix=True).tocsc().T,
        var=pd.read_csv(  # ty:ignore[no-matching-overload]
            ds_path / "features.tsv.gz",
            delimiter="\t",
            header=None,
            names=["gene", "SYMBOL"],
            usecols=[0, 1],
        ).set_index("gene"),
        obs=pd.read_csv(
            ds_path / "meta.tsv",
            delimiter="\t",
            index_col=0,
        )[["in_tissue", "percent.mt", "cell_type"]]
        .loc[c_idx]
        .assign(in_tissue=lambda e: e["in_tissue"].astype(bool)),
        obsm={
            "spatial": pd.read_csv(  # ty:ignore[no-matching-overload]
                ds_path / "tissue_positions_list.csv.gz",
                delimiter=",",
                header=None,
                usecols=[0, 4, 5],
                names=["cell", "x", "y"],
            )
            .set_index("cell")
            .loc[c_idx]
        },
    )

    for k, v in kwargs.items():
        adata.obs[k] = v

    return adata


def psc_ds(pth: Path, filt_fn: Callable[[pd.DataFrame], pd.Series] | None = None) -> dict[str, ad.AnnData]:

    meta = pd.read_csv(pth / "meta.tsv", sep="\t")
    if filt_fn is not None:
        meta = meta.loc[filt_fn]

    return {
        row["sample_id"]: load_psc_ds(
            pth / row["sample_id"],
            **(row[["group"]]),
        )
        for _, row in meta.iterrows()
    }


def viz_defs() -> None:
    mpl.rcParams["figure.dpi"] = 300
    theme_set(theme_matplotlib)
    theme_update()


def pl_k_sP(
    adata: ad.AnnData,
    k_oi: int,
    cond_lab: str,
    sample_lab: str,
    ctl_val: str,
    fmt_str: str = "sP_p_hat_k={:02}",
    spatial_obsm: str = "spatial",
) -> Mapping[str, p9.ggplot]:
    obsm_sl_oi = fmt_str.format(k_oi)
    plo = apl.PlotAdata(adata)

    p_hat_thresh = (
        adata.obsm[obsm_sl_oi]
        .groupby(  # ty:ignore[unresolved-attribute]
            sP.BIN_KEY
        )[sP.PAT_KEY]
        .agg(["min", "max"])
        .melt()["value"]
        .iloc[[1, 2]]
        .mean()
    )
    sp_pl = [
        p9.aes(x=apl.obsm(spatial_obsm, "x"), y=apl.obsm(spatial_obsm, "y")),
        p9.geom_point(shape="h", size=0.1),
        p9.coord_fixed(),
        p9.facet_wrap([apl.obs(cond_lab), apl.obs(sample_lab)], nrow=2, shrink=False, scales="free"),
        p9.theme(axis_text=p9.element_blank(), axis_ticks=p9.element_blank(), aspect_ratio=1),
    ]
    vln_pl = (
        plo
        + p9.aes(y=apl.obsm(obsm_sl_oi, sP.PAT_KEY))
        + p9.geom_violin(scale="count")
        + p9.geom_boxplot(width=0.1, outlier_shape="", position=p9.position_dodge(0.9), show_legend=False)
        + p9.labs(y="p\u0302", title=f"p\u0302 distributions for k={k_oi}", fill="")
        + p9.geom_hline(yintercept=p_hat_thresh, linetype="dashed", alpha=0.8, color="red")
        + p9.guides(color="none")
    )

    return {
        "spatial p-hat": sum(sp_pl, start=plo)
        + p9.aes(color=apl.obsm(obsm_sl_oi, sP.PAT_KEY))
        + p9.scale_color_cmap("cividis")
        + p9.labs(color="p\u0302", title=f"p\u0302 spatial distributions for k={k_oi}"),
        "spatial bin": sum(sp_pl, start=plo)
        + p9.aes(color=apl.obs(cond_lab).fetch(adata) + np.where(apl.obsm(obsm_sl_oi, sP.BIN_KEY).fetch(adata), "1", "0"))
        + p9.labs(color="refined\nlabel", title=f"refined label spatial distributions for k={k_oi}"),
        "spatial p-hat (case-1 only)": sum(sp_pl, start=apl.PlotAdata(adata[adata.obs[cond_lab] != ctl_val, :]))
        + p9.aes(color=apl.obsm(obsm_sl_oi, sP.PAT_KEY), alpha=apl.obsm(obsm_sl_oi, sP.BIN_KEY))
        + p9.scale_color_cmap("cividis")
        + p9.scale_alpha_manual({False: 0, True: 1}, guide=None)
        + p9.labs(color="p\u0302", title=f"p\u0302 spatial distributions for k={k_oi} for case-1 cells"),
        "emb/p-hat": plo
        + p9.aes(x=apl.obsm(obsm_sl_oi, sP.peel.INP_KEY), y=apl.obsm(obsm_sl_oi, sP.PAT_KEY), color=apl.obs("group"))
        + p9.geom_point(alpha=1)
        + p9.facet_wrap(apl.obs("group"))
        + p9.scale_x_continuous(expand=(0.1, 0))
        + p9.labs(x=f"embedding [k={k_oi}]", y="p\u0302", title=f"p\u0302 / embedding for k={k_oi}"),
        "sample p-hat": vln_pl
        + p9.aes(x=apl.obs(sample_lab))
        + p9.scale_x_discrete(limits=lambda e: list(reversed(e)))
        + p9.coord_flip(),
        "group p-hat": vln_pl
        + p9.aes(
            x=apl.obs(cond_lab),
            fill=np.where(
                apl.obsm(obsm_sl_oi, sP.BIN_KEY).fetch(adata) & (apl.obs(cond_lab).fetch(adata) != ctl_val), "1", "0"
            ),
        )
        + p9.scale_x_discrete(limits=lambda e: sorted(e, key=lambda x: (ctl_val not in x, x))),
        "group p-hat (ctl split)": vln_pl
        + p9.aes(x=apl.obs(cond_lab), fill=np.where(apl.obsm(obsm_sl_oi, sP.BIN_KEY).fetch(adata), "1", "0"))
        + p9.scale_x_discrete(limits=lambda e: sorted(e, key=lambda x: (ctl_val not in x, x))),
    }
