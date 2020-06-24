# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = cotk
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

public:
	@cd meta && python update_doc.py -D public && cd ..
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

internal:
	@cd meta && python update_doc.py -D internal && cd ..
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

checkpublic:
	@cd meta && python update_doc.py --check -D public && cd ..

public_only_source:
	@cd meta && python update_doc.py -D public --only_source && cd ..
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)