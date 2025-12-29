# Makefile for RheoQCM Sphinx documentation
#
# Usage:
#   make html      - Build HTML documentation
#   make clean     - Remove build directory
#   make linkcheck - Check external links
#   make doctest   - Run doctests in documentation
#   make livehtml  - Build with auto-reload (requires sphinx-autobuild)

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html linkcheck doctest livehtml

help:
	@echo "RheoQCM Documentation Build Targets:"
	@echo ""
	@echo "  make html       - Build HTML documentation"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make linkcheck  - Check all external links"
	@echo "  make doctest    - Run embedded doctests"
	@echo "  make livehtml   - Live-reload development server"
	@echo ""
	@echo "Output: $(BUILDDIR)/html/"

clean:
	rm -rf "$(BUILDDIR)"
	rm -rf "$(SOURCEDIR)/api/generated"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo ""
	@echo "Build finished. HTML pages are in $(BUILDDIR)/html/"

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

doctest:
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) --watch ../src

# Catch-all target: route all unknown targets to Sphinx
%:
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
