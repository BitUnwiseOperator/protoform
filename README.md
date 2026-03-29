# protoform

A progressive, from-scratch transformer tutorial built on Shakespeare.

> **Work in progress** — v1.0 will be a full end-to-end guide to building a
> transformer from scratch. Notebooks, API docs, and interactive examples are
> being added stage by stage.

## Quickstart

Requires Python 3.12+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/BitUnwiseOperator/protoform.git
cd protoform
uv sync --extra dev --extra docs
```

## Docs

Build the Sphinx docs locally to view tutorials and API reference:

```bash
make html
```

Then open `docs/build/html/index.html` in your browser.

## Notebooks

The notebooks in `docs/source/notebooks/` are meant to be tinkered with.
Open them in Jupyter and experiment:

```bash
uv run jupyter notebook docs/source/notebooks/
```

Available notebooks:

- **00 — What is a Tensor?** — Linear algebra foundations with Plotly visualizations
- **01 — Tokenizer** — Character-level tokenization from scratch
- **02 — Datasets** — Loading and splitting Shakespeare for training
- **99 — Python Idioms** — Reference notebook for Python patterns used throughout

The docs build re-executes any notebook you've modified. If you break
something and the build fails, reset the notebooks and clear the cache:

```bash
git checkout docs/source/notebooks/
make clean
make html
```

## Development

```bash
make test       # run tests
make check      # lint and typecheck
```

## License

[Apache 2.0](LICENSE)
