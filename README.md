# STARLING
Highly multiplexed imaging technologies such as Imaging Mass Cytometry (IMC) enable the quantification of the expression proteins in tissue sections while retaining spatial information. Data preprocessing pipelines subsequently segment the data to single cells, recording their average expression profile along with spatial characteristics (area, morphology, location etc.). However, segmentation of the resulting images to single cells remains a challenge, with doublets -- an area erroneously segmented as a single-cell that is composed of more than one 'true' single cell -- being frequent in densely packed tissues. This results in cells with implausible protein co-expression combinations, confounding the interpretation of important cellular populations across tissues.

While doublets have been extensively discussed in the context of single-cell RNA-sequencing analysis, there is currently no method to cluster IMC data while accounting for such segmentation errors. Therefore, we introduce SegmentaTion AwaRe cLusterING (STARLING), a probabilistic method tailored for densely packed tissues profiled with IMC that clusters the cells explicitly allowing for doublets resulting from mis-segmentation. To benchmark STARLING against a range of existing clustering methods, we further develop a novel evaluation score that penalizes methods that return clusters with biologically-implausible marker co-expression combinations. Finally, we generate IMC data of the human tonsil -- a densely packed human secondary lymphoid organ -- and demonstrate cellular states captured by STARLING identify known cell types not visible with other methods and important for understanding the dynamics of immune response.

# Quick start:
1. pip install -m requirements.txt to setup a new environment.
2. A sample input file (sample_input.h5ad) can be found.
3. Run STARLING via starling.py (sample file must be in the same directory).
4. Note: the information can be retrieved in annData object.
   - st.adata.varm['init_exp_centroids'] -- initial expression cluster centroids (P x C matrix)
   - st.adata.varm['st_exp_centroids'] -- STARLING expression cluster centroids (P x C matrix)
   - st.adata.uns['init_cell_size_centroids'] -- initial cell size centroids if STARLING models cell size
   - st.adata.uns['st_cell_size_centroids'] -- initial & STARLING cell size centroids if STARLING models cell size
   - st.adata.obsm['assignment_prob_matrix'] -- cell assignment distributions (N x C maxtrix)
   - st.adata.obs['doublet'] -- doublet indicator
   - st.adata.obs['doublet_prob'] -- doublet probabilities
   - st.adata.obs['init_label'] -- inital assignments
   - st.adata.obs['st_label'] -- STARLING assignments
   - st.adata.obs['max_assign_prob'] -- STARLING max probabilites of assignments
      - N: # of cells; C: # of clusters; P: # of proteins
6. tutorial.ipynb is an example of running STARLING
