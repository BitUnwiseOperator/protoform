SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run --extra docs sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

# ── Code quality ──────────────────────────────────────────────────────────────

.PHONY: lint format typecheck check test html help

lint:
	uv run --extra dev ruff check .

format:
	uv run --extra dev ruff format .

typecheck:
	uv run --extra dev pyright .

check: lint typecheck

test:
	uv run --extra dev pytest

# ── Docs ──────────────────────────────────────────────────────────────────────

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all: route unknown targets to Sphinx (e.g. `make clean`, `make dirhtml`)
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
