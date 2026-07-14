from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Any, Iterator

import anndata as ad
import numpy as np
import pandas as pd
import plotnine as p9
from matplotlib.figure import Figure

# once anndata accessors (see https://anndata.scverse.org/en/latest/accessors.html) land in stable
# the FetchBase system here should be replaced to use accessors instead, as they fulfill the intended functionality


class FetchBase(ABC):
    @property
    @staticmethod
    @abstractmethod
    def by_var(cls) -> bool: ...

    @abstractmethod
    def fetch(self, adata: ad.AnnData) -> np.ndarray: ...

    # need an iter method since facet_grid calls the list constructor on
    # its input in the case that a single faceting variable is provided
    def __iter__(self) -> Iterator:
        return iter([self])


def to_1d_arr(x: Any) -> np.ndarray:
    if hasattr(x, "todense") and isinstance(x.todense, Callable):
        x = x.todense()

    return np.asarray(x).ravel()


def col_from_df_or_arr(inp: Any, j: str | int) -> Any:
    if isinstance(inp, pd.DataFrame):
        return inp.iloc[:, j] if isinstance(j, int) else inp.loc[:, j]

    assert isinstance(j, int), "must index non-dataframe by integer"
    return inp[:, j]


@dataclass(frozen=True)
class FetchX(FetchBase):
    k: str
    g: str | None
    by_var = False

    def fetch(self, adata: ad.AnnData) -> np.ndarray:
        if self.g is None:
            return to_1d_arr(adata[:, self.k].X)
        return to_1d_arr(adata[:, self.g].layers[self.k])


@dataclass(frozen=True)
class FetchT(FetchBase):
    k: str
    c: str | None
    by_var = True

    def fetch(self, adata: ad.AnnData) -> np.ndarray:
        if self.c is None:
            return to_1d_arr(adata[self.k, :].X)
        return to_1d_arr(adata[self.c, :].layers[self.k])


@dataclass(frozen=True)
class FetchObs(FetchBase):
    k: str
    by_var = False

    def fetch(self, adata: ad.AnnData) -> np.ndarray:
        return to_1d_arr(adata.obs[self.k])


@dataclass(frozen=True)
class FetchVar(FetchBase):
    k: str
    by_var = True

    def fetch(self, adata: ad.AnnData) -> np.ndarray:
        return to_1d_arr(adata.var[self.k])


@dataclass(frozen=True)
class FetchObsm(FetchBase):
    k: str
    i: str | int
    by_var = False

    def fetch(self, adata: ad.AnnData) -> np.ndarray:
        return to_1d_arr(col_from_df_or_arr(adata.obsm[self.k], self.i))


@dataclass(frozen=True)
class FetchVarm(FetchBase):
    k: str
    i: str | int
    by_var = True

    def fetch(self, adata: ad.AnnData) -> np.ndarray:
        return to_1d_arr(col_from_df_or_arr(adata.varm[self.k], self.i))


@dataclass(frozen=True)
class FetchObsp(FetchBase):
    k: str
    c: str
    by_var = False

    def fetch(self, adata: ad.AnnData) -> np.ndarray:
        return to_1d_arr(adata[self.c, :].obsp[self.k])


@dataclass(frozen=True)
class FetchVarp(FetchBase):
    k: str
    g: str
    by_var = True

    def fetch(self, adata: ad.AnnData) -> np.ndarray:
        return to_1d_arr(adata[:, self.g].varp[self.k])


# we set the type hint for fetch functions to Any to avoid type
# errors when dealing with type-hinted plotnine functions
# in actuality, all fetching functions return a FetchBase


def X(k: str, g: str | None = None) -> Any:
    return FetchX(k, g)


def T(k: str, c: str | None = None) -> Any:
    return FetchT(k, c)


def obs(k: str) -> Any:
    return FetchObs(k)


def var(k: str) -> Any:
    return FetchVar(k)


def obsm(k: str, i: str | int) -> Any:
    return FetchObsm(k, i)


def varm(k: str, i: str | int) -> Any:
    return FetchVarm(k, i)


def obsp(k: str, i: str) -> Any:
    return FetchObsp(k, i)


def varp(k: str, i: str) -> Any:
    return FetchVarp(k, i)


class MangleKeyLabeller(p9.labeller):
    __inner: p9.labeller
    __mangle_dict: dict[str, str]

    def __init__(self, mangle_dict: dict[str, str], inner: p9.labeller):
        self.__mangle_dict = mangle_dict
        self.__inner = inner

    # location of strip_label_details type is not stable, so we do not provide an annotation for back-compat/stability
    def __call__(self, label_info):
        out = self.__inner(label_info)
        # we need to replace all references to the mangled keys we use internally if something like label_both is used as a labeller
        for var in out.variables.keys():
            out.variables[var] = reduce(
                lambda a, c: a.replace(c, self.__mangle_dict[c]),
                self.__mangle_dict.keys(),
                out.variables[var],
            )

        return out

    # forward everything else to inner labeller class
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)

        return getattr(self.__inner, attr)


class PlotAdata(p9.ggplot):
    __adata: ad.AnnData
    __vwidx: None | tuple
    data: pd.DataFrame

    def __init__(self, adata: ad.AnnData, mapping: p9.aes | None = None):
        self.__adata = adata
        self.__vwidx = (adata._oidx, adata._vidx) if adata.is_view else None
        super().__init__(pd.DataFrame(), mapping)

    # this returns typing.Self, but for back-compat we remove the annotation
    # TODO: re-add annotation when 3.11 becomes standard
    def __deepcopy__(self, memo):
        # add the anndata to memo to avoid actually copying it in the super method
        memo[id(self.__adata)] = self.__adata
        out = self.__class__.__new__(self.__class__)
        out.__dict__ = super().__deepcopy__(memo).__dict__

        assert isinstance(out, type(self))
        assert out.__adata is self.__adata

        return out

    def __fetch(self, spec: Any, by_var: bool) -> None | tuple[str, str]:
        if isinstance(spec, p9.stage):
            # catch possible infinite recursion case
            assert not isinstance(spec.start, p9.stage), "stage cannot have start be itself a staged computation"
            ks = self.__fetch(spec.start, by_var)
            if ks is not None:
                sk, _ = ks
                spec.start = sk
            return None

        if not isinstance(spec, FetchBase):
            return None

        assert spec.by_var == by_var, (
            "all data in the plot must be keyed to the same axis (e.g. cannot have one aesthetic fetched along `obs` index and another fetched along `var` index)"
        )

        hs = hash(spec)
        sk = f"__{'MP'[hs > 0]}{abs(hs):X}"

        match spec:
            case FetchX(k, _) | FetchT(k, _) | FetchObs(k) | FetchVar(k):
                nm = k
            case FetchObsm(k, i) | FetchVarm(k, i) | FetchObsp(k, i) | FetchVarp(k, i):
                nm = f"{k}:{i}"

        # only fetch if key isn't present in self.data
        if sk not in self.data:
            self.data[sk] = spec.fetch(self.__adata)

        return sk, nm

    def __facet_h(self, facets: list, by_var: bool) -> dict[str, str]:
        o = dict()
        for i in range(len(facets)):
            ks = self.__fetch(facets[i], by_var)
            if ks is not None:
                sk, nm = ks
                facets[i] = sk
                o[sk] = nm
        return o

    def _setup(self) -> Figure:
        if self.__vwidx is not None:
            # for some reason, the held anndata view becomes lost post-construction
            # so we manually keep the indices and re-gen the view before we call __fetch
            # @TODO: find exact reason for this / find way to sidestep this
            assert self.__adata._adata_ref is not None, (
                "should never happen: provided anndata was a view during construction of object but is no longer one when drawing figure"
            )
            self.__adata = self.__adata._adata_ref[self.__vwidx[0], self.__vwidx[1]]

        maps = [self, *self.layers]
        by_var = bool(
            next(aes for e in maps if len(e.mapping) > 0 for aes in e.mapping.values() if isinstance(aes, FetchBase)).by_var
        )

        for e in maps:
            for k in e.mapping.keys():
                ks = self.__fetch(e.mapping[k], by_var)
                if ks is not None:
                    sk, nm = ks
                    e.mapping[k] = sk
                    # check that a label hasn't been manually provided already for that aesthetic
                    if getattr(self.labels, k, None) is None:
                        self.labels.update(p9.labs(**{k: nm}).labels)

        lbl_dict = dict()
        match self.facet:
            case p9.facet_grid():
                lbl_dict |= self.__facet_h(self.facet.rows, by_var)
                lbl_dict |= self.__facet_h(self.facet.cols, by_var)

            case p9.facet_wrap():
                # facet_wrap constructor makes the vars field just its input when just one faceting
                # variable is provided so we manually wrap it in a list if we catch this case occurring
                if isinstance(self.facet.vars, FetchBase):
                    self.facet.vars = [self.facet.vars]  # ty:ignore[invalid-assignment]
                lbl_dict |= self.__facet_h(self.facet.vars, by_var)  # ty:ignore[invalid-argument-type]

        if len(lbl_dict) > 0:
            self.facet.labeller = MangleKeyLabeller(lbl_dict, self.facet.labeller)

        return super()._setup()
