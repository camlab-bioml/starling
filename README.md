# STARLING TOOL (ST)

[![PyPI](https://img.shields.io/pypi/v/starling-tool.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/starling-tool.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/starling-tool)][python version]
[![License](https://github.com/camlab-bioml/starling-tool/blob/main/LICENSE)][license]

[![Read the documentation at https://starling-tool.readthedocs.io/](https://img.shields.io/readthedocs/starling-tool/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/Usually-zz/starling-tool/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/Usually-zz/starling-tool/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/starling-tool/
[status]: https://pypi.org/project/starling-tool/
[python version]: https://pypi.org/project/starling-tool
[read the docs]: https://starling-tool.readthedocs.io/
[tests]: https://github.com/Usually-zz/starling-tool/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/Usually-zz/starling-tool
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

Highly multiplexed imaging technologies such as Imaging Mass Cytometry (IMC) enable the quantification of the expression proteins in tissue sections while retaining spatial information. Data preprocessing pipelines subsequently segment the data to single cells, recording their average expression profile along with spatial characteristics (area, morphology, location etc.). However, segmentation of the resulting images to single cells remains a challenge, with doublets -- an area erroneously segmented as a single-cell that is composed of more than one 'true' single cell -- being frequent in densely packed tissues. This results in cells with implausible protein co-expression combinations, confounding the interpretation of important cellular populations across tissues.

While doublets have been extensively discussed in the context of single-cell RNA-sequencing analysis, there is currently no method to cluster IMC data while accounting for such segmentation errors. Therefore, we introduce SegmentaTion AwaRe cLusterING (STARLING), a probabilistic method tailored for densely packed tissues profiled with IMC that clusters the cells explicitly allowing for doublets resulting from mis-segmentation. To benchmark STARLING against a range of existing clustering methods, we further develop a novel evaluation score that penalizes methods that return clusters with biologically-implausible marker co-expression combinations. Finally, we generate IMC data of the human tonsil -- a densely packed human secondary lymphoid organ -- and demonstrate cellular states captured by STARLING identify known cell types not visible with other methods and important for understanding the dynamics of immune response.
## Requirements

- TODO

## Installation

You can install _STARLING-TOOL (ST)_ via [pip] from [PyPI]:

```console
$ pip install starling-tool
```

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_STARLING-TOOL (ST)_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/Usually-zz/starling-tool/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/Usually-zz/starling-tool/blob/main/LICENSE
[contributor guide]: https://github.com/Usually-zz/starling-tool/blob/main/CONTRIBUTING.md
[command-line reference]: https://starling-tool.readthedocs.io/en/latest/usage.html
