from collections.abc import Callable, Mapping
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
import uapl
from plotnine import theme_matplotlib, theme_set, theme_update
from scipy.io import mmread
from sklearn.decomposition import NMF

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
    theme_set(theme_matplotlib)
    theme_update()


def pl_k_sP(
    adata: ad.AnnData,
    k_oi: int,
    cond_col: str,
    sample_col: str,
    ctl_val: str,
    fmt_str: str = "sP_p_hat_k={k}",
    spatial_obsm: str = "spatial",
) -> Mapping[str, p9.ggplot]:
    obsm_sl_oi = fmt_str.format(k=k_oi)
    adata = adata[np.ravel(np.argsort(adata.obsm[obsm_sl_oi][sP.BIN_KEY])), :]
    plo = uapl.PlotAdata(adata)

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
    cent_sp = lambda c: p9.stage(uapl.obsm(spatial_obsm, c), after_stat=f"{c}-({c}.max() - {c}.min())/2")  # noqa: E731

    sp_pl = [
        p9.aes(x=cent_sp("x"), y=cent_sp("y")),
        p9.geom_point(shape="h", size=0.1),
        p9.coord_fixed(),
        p9.theme(axis_text=p9.element_blank(), axis_ticks=p9.element_blank(), aspect_ratio=1),
        p9.labs(x="x", y="y"),
    ]
    vln_pl = (
        plo
        + p9.aes(y=uapl.obsm(obsm_sl_oi, sP.PAT_KEY))
        + p9.geom_violin(
            scale="count",
        )
        + p9.geom_boxplot(width=0.1, outlier_shape="", position=p9.position_dodge(0.9), show_legend=False)
        + p9.labs(y="p\u0302", title=f"p\u0302 distributions for k={k_oi}", fill="")
        + p9.geom_hline(yintercept=p_hat_thresh, linetype="dashed", alpha=0.8, color="red")
        + p9.guides(color="none")
    )
    non_ctl_val = "/".join(adata.obs[cond_col].loc[lambda e: e != ctl_val].unique().tolist())

    return {
        "spatial p-hat": sum(sp_pl, start=plo)
        + p9.aes(color=uapl.obsm(obsm_sl_oi, sP.PAT_KEY))
        + p9.scale_color_cmap("cividis")
        + p9.labs(color="p\u0302", title=f"p\u0302 spatial distributions for k={k_oi}")
        + p9.facet_wrap([uapl.obs(cond_col), uapl.obs(sample_col)], nrow=2, shrink=True),
        "spatial bin": sum(sp_pl, start=plo)
        + p9.aes(color=uapl.obs(cond_col).fetch(adata) + np.where(uapl.obsm(obsm_sl_oi, sP.BIN_KEY).fetch(adata), "-1", "-0"))
        + p9.labs(color="refined\nlabel", title=f"refined label spatial distributions for k={k_oi}")
        + p9.facet_wrap([uapl.obs(cond_col), uapl.obs(sample_col)], nrow=2, shrink=True),
        "spatial p-hat (ctl/split)": sum(sp_pl, start=uapl.PlotAdata(adata[adata.obs[cond_col] == ctl_val]))
        + p9.aes(color=uapl.obsm(obsm_sl_oi, sP.PAT_KEY))
        + p9.scale_color_cmap("cividis")
        + p9.labs(color="p\u0302", title=f"p\u0302 spatial distributions for k={k_oi} for control cells (split by refinement)")
        + p9.facet_grid(
            cols=uapl.obs(sample_col),
            rows=uapl.obsm(obsm_sl_oi, sP.BIN_KEY),
            labeller=lambda e: {"False": f"{ctl_val}-0", "True": f"{ctl_val}-1"}.get(e, e),  # ty:ignore[invalid-argument-type]
            shrink=True,
        ),
        "spatial p-hat (case/split)": sum(sp_pl, start=uapl.PlotAdata(adata[adata.obs[cond_col] != ctl_val]))
        + p9.aes(color=uapl.obsm(obsm_sl_oi, sP.PAT_KEY))
        + p9.scale_color_cmap("cividis")
        + p9.labs(color="p\u0302", title=f"p\u0302 spatial distributions for k={k_oi} for case cells (split by refinement)")
        + p9.facet_grid(
            cols=uapl.obs(sample_col),
            rows=uapl.obsm(obsm_sl_oi, sP.BIN_KEY),
            labeller=lambda e: {"False": f"{non_ctl_val}-0", "True": f"{non_ctl_val}-1"}.get(e, e),  # ty:ignore[invalid-argument-type]
            shrink=True,
        ),
        "emb/p-hat": plo
        + p9.aes(x=uapl.obsm(obsm_sl_oi, sP.peel.INP_KEY), y=uapl.obsm(obsm_sl_oi, sP.PAT_KEY), color=uapl.obs(cond_col))
        + p9.geom_point(alpha=1)
        + p9.facet_wrap(uapl.obs(cond_col))
        + p9.scale_x_continuous(expand=(0.1, 0))
        + p9.labs(x=f"embedding [k={k_oi}]", y="p\u0302", title=f"p\u0302 / embedding for k={k_oi}"),
        "sample p-hat": vln_pl
        + p9.aes(x=uapl.obs(sample_col))
        + p9.scale_x_discrete(limits=lambda e: list(reversed(e)))
        + p9.coord_flip(),
        "group p-hat": vln_pl
        + p9.aes(
            x=uapl.obs(cond_col),
            fill=np.where(
                uapl.obsm(obsm_sl_oi, sP.BIN_KEY).fetch(adata) & (uapl.obs(cond_col).fetch(adata) != ctl_val), "1", "0"
            ),
        )
        + p9.scale_x_discrete(limits=lambda e: sorted(e, key=lambda x: (ctl_val not in x, x))),
        "group p-hat (ctl split)": vln_pl
        + p9.aes(x=uapl.obs(cond_col), fill=np.where(uapl.obsm(obsm_sl_oi, sP.BIN_KEY).fetch(adata), "1", "0"))
        + p9.scale_x_discrete(limits=lambda e: sorted(e, key=lambda x: (ctl_val not in x, x))),
    }


def run_deg_groups(adata: ad.AnnData, k_oi: int, cond_col: str, ctl_val: str, fmt_str: str = "sP_p_hat_k={k}") -> pd.DataFrame:
    obsm_sl_oi = fmt_str.format(k=k_oi)
    case_val = adata.obs[cond_col].loc[lambda e: e != ctl_val].unique().tolist()
    assert len(case_val) == 1, "cannot have more than one non control value for this to work"
    case_val = case_val[0]

    deg_conf = {
        "case_vs_control": ({uapl.obs(cond_col): case_val},),
        #
        "case1_vs_rest": ({uapl.obs(cond_col): case_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): True},),
        "case1_vs_control": (
            {uapl.obs(cond_col): case_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): True},
            {uapl.obs(cond_col): ctl_val},
        ),
        "case1_vs_case0": (
            {uapl.obs(cond_col): case_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): True},
            {uapl.obs(cond_col): case_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): False},
        ),
        "case1_vs_control0": (
            {uapl.obs(cond_col): case_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): True},
            {uapl.obs(cond_col): ctl_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): False},
        ),
        #
        "case0_vs_control": (
            {uapl.obs(cond_col): case_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): False},
            {uapl.obs(cond_col): ctl_val},
        ),
        "case0_vs_control0": (
            {uapl.obs(cond_col): case_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): False},
            {uapl.obs(cond_col): ctl_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): False},
        ),
        #
        "control1_vs_control0": (
            {uapl.obs(cond_col): ctl_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): True},
            {uapl.obs(cond_col): ctl_val, uapl.obsm(obsm_sl_oi, sP.BIN_KEY): False},
        ),
    }

    deg_results = pd.concat({k: sP.diag.run_deg_mwu(adata, *v) for k, v in deg_conf.items()}, names=["cmp"]).reset_index(
        drop=False
    )
    deg_results["cmp"] = pd.Categorical(deg_results["cmp"], categories=deg_conf.keys(), ordered=True)

    return deg_results


def comp_spat_stats(adata: ad.AnnData, sP_key: str, sample_col: str) -> pd.DataFrame:
    out = {"k": [], "stat": [], "sample": [], "val": []}
    for sample, sub in adata.obs.groupby(sample_col, observed=True).indices.items():
        sub = sc.pp.neighbors(adata[sub, :], use_rep="spatial", copy=True)
        for k in range(sub.obsm["X_nmf"].shape[1]):
            k_pat = sub.obsm[sP_key.format(k=k)][sP.PAT_KEY]

            for stat, stat_fn in [
                ("moran's i", sc.metrics.morans_i),
                ("geary's c", sc.metrics.gearys_c),
            ]:
                out["k"].append(k)
                out["stat"].append(stat)
                out["sample"].append(sample)

                na_filt = np.asarray(k_pat.notna())
                out["val"].append(stat_fn(sub[na_filt, :], k_pat.loc[na_filt]))

    return pd.DataFrame(out)
