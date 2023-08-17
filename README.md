# STARLING
Highly multiplexed imaging technologies such as Imaging Mass Cytometry (IMC) enable the quantification of the expression proteins in tissue sections while retaining spatial information. Data preprocessing pipelines subsequently segment the data to single cells, recording their average expression profile along with spatial characteristics (area, morphology, location etc.). However, segmentation of the resulting images to single cells remains a challenge, with doublets -- an area erroneously segmented as a single-cell that is composed of more than one 'true' single cell -- being frequent in densely packed tissues. This results in cells with implausible protein co-expression combinations, confounding the interpretation of important cellular populations across tissues.

While doublets have been extensively discussed in the context of single-cell RNA-sequencing analysis, there is currently no method to cluster IMC data while accounting for such segmentation errors. Therefore, we introduce SegmentaTion AwaRe cLusterING (STARLING), a probabilistic method tailored for densely packed tissues profiled with IMC that clusters the cells explicitly allowing for doublets resulting from mis-segmentation. To benchmark STARLING against a range of existing clustering methods, we further develop a novel evaluation score that penalizes methods that return clusters with biologically-implausible marker co-expression combinations. Finally, we generate IMC data of the human tonsil -- a densely packed human secondary lymphoid organ -- and demonstrate cellular states captured by STARLING identify known cell types not visible with other methods and important for understanding the dynamics of immune response.

# Quick start:  
1. A sample input file (sample_input.h5ad) can be found.
2. Run STARLING via _7script_train.py (sample file must be in the same directory).
3. STARLING (ST) object (model.pt) is saved via torch.save.
4. Note: the information can be retrieved by running torch.load('model.pt')
   - st.init_X -- raw expression matrix
   - st.tr_adata.X -- tranformed expression matrix
   - st.init_e -- GMM clustering
   - st.init_l -- GMM cluster assignments
   - st.e -- ST clustering
   - st.label -- ST cluster assignments
   - st.singlet_prob -- cell imperfect segmentation probabilities
   - st.singlet_assig_prob -- cell cluster assignment probabilities
