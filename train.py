import os
import argparse

import numpy as np
import pandas as pd

import pytorch_lightning as pl

from utils import yaml_config_hook, save_model, functions

def train(input_file_name):
    sample_data = functions.sampleData(input_file_name, channels, sample_size, cofactor, model_cell_size, model_overlap)

    init_cen, init_var, init_label = functions.init_clustering(sample_data.tr_mat, initial_clustering_method, k)
    sample_data.tr_h5ad.obs['{}_label_nc{}'.format(initial_clustering_method, k)] = init_label

    init_model_params = functions.model_paramters(model_cell_size, init_cen, init_var)
    setup_starling = functions.init_setup(init_model_params, sample_data.train_df, sample_data.val_df)

    fit_starling = functions.model(setup_starling.model_params, dist_option, model_cell_size, model_overlap, model_regularizer)

    trainer = pl.Trainer(max_epochs=200, accelerator='auto', devices='auto')
    trainer.fit(fit_starling, setup_starling)
    
    cen, var, label, prob_singlet, prob_cluster_assig = functions.starling(sample_data.tr_mat, setup_starling.model_params, dist_option, model_cell_size, model_overlap)
    sample_data.tr_h5ad.obs['star_{}_label_nc{}'.format(initial_clustering_method, k)] = label

    ## save labels & centroids
    pretty_printing = sample_data.tr_h5ad.var_names

    if model_cell_size:
        pretty_printing = np.hstack((pretty_printing, 'size'))

    pd.DataFrame(init_cen, columns = pretty_printing).to_csv(code_dir + "/output/init_centroids.csv")
    pd.DataFrame(cen, columns = pretty_printing).to_csv(code_dir + "/output/star_centroids.csv")
    sample_data.tr_h5ad.obs.to_csv(code_dir + "/output/star_labels.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = '/home/campbell/yulee/project/starling'
    config = yaml_config_hook(code_dir + '/config/config.yaml')
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    k = args.k ## number of cluster 
    sample_size = args.sample_size ## subsample size
    #code_dir = args.code_dir ## directory where this file is
    #input_dir = args.dataset_dir ## input data directory
    #output_dir = args.output_dir ## output directory

    cofactor = args.cofactor ## factor of the data
    model_regularizer = args.model_regularizer ## model regularizer (nll model + doublet loss)

    dist_option = args.dist ## model distribution (normal or t distribution)
    model_overlap = args.model_overlap ## model the size of cell in z plane
    model_cell_size = args.model_cell_size ## model cell size or not

    initial_clustering_method = args.initial_clustering_method ## choose among Kmeans (KM), GMM, Phenograph (PG), FlowSOM (FS)

    cohort = args.cohort
    if cohort == 'basel':
        channels = yaml_config_hook(code_dir + '/config/basel_channel.yaml')['pretty_channels']
    elif cohort == 'meta':
        channels = yaml_config_hook(code_dir + '/config/meta_channel.yaml')['pretty_channels']
    elif cohort == 'tonsil':
        channels = yaml_config_hook(code_dir + '/config/tonsil_channel.yaml')['pretty_channels']

    ## input file name
    input_file_name = '{}/example_input/{}_dc.csv'.format(code_dir, cohort)
    train(input_file_name)
