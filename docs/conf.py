"""Sphinx configuration."""
project = "ST"
author = "Jett (Yuju) Lee"
copyright = "2023, Jett (Yuju) Lee"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
