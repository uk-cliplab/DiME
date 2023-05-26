This folder has an implementation of a multiview autoencoder that learns representations with DiME

Folder structure:
    utils/data_utils/* - functions to create multiview datasets
    utils/* - functions that facilitate model training and evaluation

    losses.py - implementations of various loss functions
    models.py - implementations of various netework architectures
    run_experiments.py - Main vehicle for creating, training, and evaluating models

    disentanglement.ipynb - Shows how to use conditional entropy to disentangle shared and exclusive components.
    multiview_mi - shows rotated/noisy dataset and TSNE embeddings of what DiME learns

Conda environment
    - conda env create -f environment.yml
    - conda activate mnist

Usage:
    - There are two notebooks, disentanglement.ipynb and multiview_mi.ipynb, that make it easy to see some experiments

    - Otherwise, run_experiments.py is the main entry point for interacting with the codebase. There are many parameters you can use when you call to it, which are detailed within the file. The purpose is to set up large experiments easily. Below is a simple example of using it. 
    
    -python3 -u run_experiments.py -e "cifar10" -bs 3000 -ld 20 -m mv_doe --dataset "cifar10" -lr 0.001 --num-epochs 250

    -python3 run_experiments.py -ld 5,10 -m mv_doe_postclass,mv_doe_postclassfrozen,mv_cca_postclass,mv_cca_postclassfrozen,mv_classifier -bs 3000 --dataset 'rot_noisy'  -e 'mnist' -np 4 --arch CNN --evaluate