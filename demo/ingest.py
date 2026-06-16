from collections.abc import Callable, Mapping
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from sklearn.decomposition import NMF


def run_proc_per_ds[T](adatas: Mapping[T, sc.AnnData]) -> None:
    # scale by sample
    for adata in adatas.values():
        adata.raw = adata
        adata.layers["counts"] = adata.X
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
        mmread(next(ds_path.glob("*.mtx")), spmatrix=False).tocsc().T,
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
            **(row[["Sample details", "Animal", "Timepoint", "Condition", "GoodQuality"]]),
        )
        for _, row in meta.iterrows()
    }


def load_psc_ds(ds_path: Path, **kwargs) -> ad.AnnData:
    obs = pd.read_csv(
        ds_path / "barcodes.tsv.gz",
        delimiter="\t",
        header=None,
        names=["cell"],
    ).set_index("cell")

    adata = ad.AnnData(
        mmread(ds_path / "matrix.mtx.gz", spmatrix=False).tocsc().T,
        var=pd.read_csv(
            ds_path / "features.tsv.gz",
            delimiter="\t",
            header=None,
            names=["gene", "SYMBOL"],
            usecols=[0, 1],
        ).set_index("gene"),
        obs=obs.index.to_frame(),
        obsm={
            "spatial": pd.read_csv(
                ds_path / "tissue_positions_list.csv.gz",
                delimiter=",",
                header=None,
                usecols=[0, 4, 5],
                names=["cell", "x", "y"],
            )
            .set_index("cell")
            .loc[obs.index]
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
