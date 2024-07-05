# SegmentaTion AwaRe cLusterING (STARLING)

![build](https://github.com/camlab-bioml/starling/actions/workflows/main.yml/badge.svg)
![](https://img.shields.io/badge/Python-3.9-blue)
![](https://img.shields.io/badge/Python-3.10-blue)


STARLING is a probabilistic model for clustering cells measured with spatial expression assays (e.g. IMC, MIBI, etc...) while accounting for segmentation errors.

It outputs:
1. Clusters that account for segmentation errors in the data (i.e. should no longer show implausible marker co-expression)
2. Assignments for every cell in the dataset to those clusters
3. A segmentation error probability for each cell

A **preprint** describing the method and introducing a novel benchmarking workflow is available: [Lee et al. (2024) _Segmentation error aware clustering for highly multiplexed imaging_](https://www.biorxiv.org/content/10.1101/2024.02.29.582827v1)

A **tutorial** outlining basic usage is available [here][tutorial].

![Model](https://github.com/camlab-bioml/starling/raw/main/starling-schematic600x.png)

## Requirements

Python 3.9 or 3.10 are required to run starling. If your current version of python is not one of these, we recommend using [pyenv](https://github.com/pyenv/pyenv) to install a compatible version alongside your current one. Alternately, you could use the Docker configuration described below.

## Installation

### Install with pip

`pip install biostarling` and then import the module `from starling import starling`

### Building from source

Starling can be cloned and installed locally (typically <10 minutes) via the Github repository,

```
git clone https://github.com/camlab-bioml/starling.git && cd starling
```

After cloning the repository, the next step is to install the required dependencies. There are three recommended methods:

### 1. Use `requirements.txt` and your own virtual environment:

We use virtualenvwrapper (4.8.4) to create and activated a standalone virtual environment for _starling_:

```
pip install virtualenvwrapper==4.8.4
mkvirtualenv starling
```

For convenience, one can install packages in the tested environment:

```
pip install -r requirements.txt
```

The virtual environment can be activated and deactivated subsequently:

```
workon starling
deactivate
```

### 2. Use Poetry and `pyproject.toml`.

[Poetry](https://python-poetry.org/) is a packaging and dependency management tool can simplify code development and deployment. If you do not have Poetry installed, you can find instructions [here](https://python-poetry.org/docs/).

Once poetry is installed, navigate to the `starling` directory and run `poetry install`. This will download the required packages into a virtual environment and install Starling in development mode. The location and state of the virtual environment may depend on your system. For more details, see [the documentation](https://python-poetry.org/docs/managing-environments/).


### 3. Use Docker

If you have Docker installed on your system, you can run `docker build -t starling .` from the project root in order to build the image locally. You can then open a shell within the image with a command like `docker run --rm -it starling bash`.

## Getting started

With starling installed, please proceed to the [online documentation][docs] or launch the [interactive notebook tutorial][tutorial] to learn more about the package's features.

## Authors

This software is authored by: Jett (Yuju) Lee, Conor Klamann, Kieran R Campbell

Lunenfeld-Tanenbaum Research Institute & University of Toronto

<!-- github-only -->

[tutorial]: https://colab.research.google.com/github/camlab-bioml/starling/blob/main/docs/source/tutorial/getting-started.ipynb
[license]: https://github.com/camlab-bioml/starling/blob/main/LICENSE
[docs]: https://camlab-bioml.github.io/starling/
