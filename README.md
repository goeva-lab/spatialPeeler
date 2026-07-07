# spatialPeeler (prototype package)

jupytext notebooks providing demonstrative use are provided in the [`./demo`](./demo) subdirectory.

repo is split into [`apl`](./src/apl/) (generic [`plotnine`](https://plotnine.org/guide/install.html)-based AnnData plotting utilities, will probably be moved into a separate repo eventually) and [`spatialPeeler`](./src/spatialpeeler/) packages.

## installation

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is used for package management, installation can be done via the following command:

```bash
uv pip install -e '.'
```

## running the notebooks

running the notebooks provided in the [`./demo`](./demo) folder requires installing dev dependencies and setting up a jupyter kernel w/ the current python executable:

```bash
uv pip install -e '.[dev]'
python -m ipykernel install --name spatialPeeler --user
```

generating `ipynb` files for all the the jupytext notebooks can be done as follows (remove the `--execute` flag if you wish to run the cells manually):

```bash
jupytext --to 'notebook' --execute 'demo/demo-'*'.py'
```
