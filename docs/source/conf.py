project = "Starling"
copyright = "2024, contribs"
author = "contribs"

extensions = [
    "autodocsumm",
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}

source_suffix = [".rst", ".md"]

nb_execution_timeout = -1

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
