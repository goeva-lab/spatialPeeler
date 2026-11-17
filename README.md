# spatialPeeler

this package provides an implementation of the `spatialPeeler` method for discovery of condition specific gene programs in spatial -omics data.

## installation

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is used for package management, installation can be done via the following command:

```bash
uv pip install 'git+https://github.com/goeva-lab/spatialPeeler.git'
```

## basics

the `spatialpeeler` package exposes the `run_fact_reg` and `run_gene_reg` entrypoints for executing the workflow.\
moreover, the `spatialpeeler.diag` module exposes helpers useful in for assessing pipeline outputs.

a demonstration of an end-to-end workflow can be seen below:

```python
import spatialpeeler as sP

SP_FACT_PAT_STR = "sP_fact_k={k}"

adata.obsm["X_nmf"] = NMF(n_components=30).fit_transform(adata.X)

# run per-factor regression
for k in range(adata.obsm["X_nmf"].shape[1]):
    sP.run_fact_reg(adata, "condition", "ctl", k, SP_FACT_PAT_STR.format(k=k))

# inspect regression betas
beta_df = sP.diag.sweep_uns(adata, SP_FACT_PAT_STR)

# assess spatial statistics
stat_df = sP.diag.comp_spat_stats(adata, SP_FACT_PAT_STR, "fov")

# assess gene-wise effects for factor of interest
k = FACTOR_OF_INTEREST
sP.run_gene_reg(adata, SP_FACT_PAT_STR.format(k=k), f"sP_gene_k={k}")
```

## development

building documentation requires installing additional development dependencies and installing a jupyter kernel named `spatialPeeler`.
assuming a repository checkout at the current working directory, this can be done as follows:

```bash
uv pip install -e '.[dev]'
uv run --no-sync --frozen -m ipykernel install --user --name 'spatialPeeler'
```

### building the documentation website

[`sphinx`](https://www.sphinx-doc.org/en/master/) is used for rendering documentation, which can be built and previewed locally at [`localhost:8000`](http://localhost:8000) via the following command:

```bash
uv run --no-sync --frozen sphinx-autobuild './docs' './docs/_build/html' -b 'dirhtml' -n --watch './src'
```

> [!NOTE]
> this will run all demo notebook code blocks, which can be very computationally expensive.
> to avoid doing this, you can pass `-D 'nb_execution_mode=off'` to the above `sphinx-autobuild` command as an additional argument.
