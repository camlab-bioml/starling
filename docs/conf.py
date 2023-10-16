"""Sphinx configuration."""
project = "starling"
author = "Jett Lee"
copyright = "2023, Jett Lee"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
