## SegmentaTion AwaRe cLusterING (starling)

Highly multiplexed imaging technologies such as Imaging Mass Cytometry (IMC) enable the quantification of the expression proteins in tissue sections while retaining spatial information. Data preprocessing pipelines subsequently segment the data to single cells, recording their average expression profile along with spatial characteristics (area, morphology, location etc.). However, segmentation of the resulting images to single cells remains a challenge, with doublets -- an area erroneously segmented as a single-cell that is composed of more than one 'true' single cell -- being frequent in densely packed tissues. This results in cells with implausible protein co-expression combinations, confounding the interpretation of important cellular populations across tissues.

While doublets have been extensively discussed in the context of single-cell RNA-sequencing analysis, there is currently no method to cluster IMC data while accounting for such segmentation errors. Therefore, we introduce SegmentaTion AwaRe cLusterING (STARLING), a probabilistic method tailored for densely packed tissues profiled with IMC that clusters the cells explicitly allowing for doublets resulting from mis-segmentation. To benchmark STARLING against a range of existing clustering methods, we further develop a novel evaluation score that penalizes methods that return clusters with biologically-implausible marker co-expression combinations. Finally, we generate IMC data of the human tonsil -- a densely packed human secondary lymphoid organ -- and demonstrate cellular states captured by STARLING identify known cell types not visible with other methods and important for understanding the dynamics of immune response.

## Installation

_starling_ can be cloned and installed locally using access to the Github repository

```
git clone https://github.com/camlab-bioml/starling.git && cd starling
```

we use virtualenvwrapper (4.8.4) to create and activated a standalone virtual environment for _starling_,

```
pip install virtualenvwrapper==4.8.4
mkvirtualenv starling
```

for convenience, one can install packages in the tested environment,

```
pip install -r requirements.txt
```

the virtual environment can be activated and deactivated subsequently

```
workon starling 
deactivate
```

Note: a list of required packages can be found in setup.py if one does not want to create a new virtual environment. 

## Getting started

Launch the interactive tutorial: [jupyter notebook][tutorial]

## License

Distributed under the terms of the [MIT license][license],
_starling_ is free and open source software.

## Authors

| Jett (Yuju) Lee & Kieran Campbell
| Lunenfeld-Tanenbaum Research Institute & University of Toronto

<!-- github-only -->

[tutorial]: https://github.com/camlab-bioml/starling/blob/main/docs/tutorial.ipynb
[license]: https://github.com/camlab-bioml/starling/blob/main/LICENSE