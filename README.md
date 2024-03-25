## SegmentaTion AwaRe cLusterING (STARLING)

![build](https://github.com/camlab-bioml/starling/actions/workflows/main.yml/badge.svg)

STARLING is a probabilistic model for clustering cells measured with spatial expression assays (e.g. IMC, MIBI, etc...) while accounting for segmentation errors.

It outputs:
1. Clusters that account for segmentation errors in the data (i.e. should no longer show implausible marker co-expression)
2. Assignments for every cell in the dataset to those clusters
3. A segmentation error probability for each cell

A **preprint** describing the method and introducing a novel benchmarking workflow is available: [Lee et al. (2024) _Segmentation error aware clustering for highly multiplexed imaging_](https://www.biorxiv.org/content/10.1101/2024.02.29.582827v1)

A **tutorial** outlining basic usage is available [here](https://github.com/camlab-bioml/starling/blob/main/docs/source/tutorial/getting-started.ipynb).

![Model](https://github.com/camlab-bioml/starling/raw/main/starling-schematic600x.png)

## Installation

_starling_ can be cloned and installed locally using access to the Github repository,

```
git clone https://github.com/camlab-bioml/starling.git && cd starling
```

After cloning the repository, the next step is to install the required dependencies. There are two recommended methods:

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


A list of minimal required packages needed for _starling_ can be found in setup.py if creating a new virtual environment is not an option.

## Getting started

Launch the interactive tutorial: [jupyter notebook][tutorial]

## Authors

This software is authored by: Jett (Yuju) Lee, Conor Klamann, Kieran R Campbell

Lunenfeld-Tanenbaum Research Institute & University of Toronto

<!-- github-only -->

[tutorial]: https://colab.research.google.com/github/camlab-bioml/starling/blob/main/docs/source/tutorial/getting-started.ipynb
[license]: https://github.com/camlab-bioml/starling/blob/main/LICENSE
