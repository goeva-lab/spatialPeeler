# spatialPeeler (prototype package)

jupytext notebooks providing demonstrative use are provided in the [`./demo`](./demo) subdirectory.

repo is split into [`apl`](./src/apl/) (generic [`plotnine`](https://plotnine.org/guide/install.html)-based AnnData plotting utilities, will probably be moved into a separate repo eventually) and [`spatialPeeler`](./src/spatialpeeler/) packages.

## installation

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is used for package management, installation can be done via the following command:

```bash
uv pip install -e '.'
```
