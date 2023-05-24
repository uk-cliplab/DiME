import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.multiprocessing as mp
import itertools
import os
import models
import utils.visual_utils as vu
import utils.data_utils.multiview_dataset_maker
import utils.training_utils as tu
import utils.comparison_utils as cv
import argparse


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--experiment-name', required=True, help='Experiment name for saved files', type=str)

parser.add_argument('-bs', '--batch-size', nargs='?', const=[1000, 3000], default=[1000, 3000], help='Comma-delimited list of batchsizes.  Ex) --batch-size 1000,5000',
                             type=lambda s: [int(item) for item in s.split(',')])

parser.add_argument('-ld', '--latent-dim', nargs='?', const=[2, 5, 10, 20], default=[2, 5, 10, 20], help='Comma-delimited list of latent dims.  Ex) --latent-dim 2,5,10',
                             type=lambda s: [int(item) for item in s.split(',')])

parser.add_argument('-m', '--models', required=True, help='Comma-delimited list of batchsizes.  Ex) --latent-dim recon,doe_recon',
                             type=lambda s: [item for item in s.split(',')])

parser.add_argument('-lr', type=float, nargs='?', const=0.005, default=0.005, help='Learning rate for trained models')

parser.add_argument('--num-epochs', type=int, nargs='?', const=50, default=50, help='Number of epochs to train each model')

parser.add_argument('--linear-size', type=int, nargs='?', const=1024, default=1024, help='Size of linear layers in model')

parser.add_argument('--force-retrain', type=bool, nargs='?', const=True, default=False, help='Force retraining a model if it already exists')

parser.add_argument('-np', '--num-parallel', type=int, nargs='?', const=1, default=1, help='Number of parallel workers')

parser.add_argument('--dataset', required=True, type=str, choices=["rot_noisy", "cifar10"], help='Which dataset to use')

parser.add_argument('--arch', type=str, nargs='?', default="CNN", choices=["CNN", "bigCNN"], help='Architecture')

parser.add_argument('--verbose', type=bool, nargs='?', const=True, default=False, help='Show verbose progress')

parser.add_argument('--eval', type=bool, nargs='?', const=True, default=False, help='Set evlauation mode')

parser.add_argument('--traineval', type=bool, nargs='?', const=True, default=False, help='Set train and evaluation mode')

parser.add_argument('--device', type=int, nargs='?', const=0, default=0, help='which cuda device to use')

parser.add_argument('--recon', type=int, nargs='?', const=0, default=0, help='how many recon dims to use')

parser.add_argument('--single', type=bool, nargs='?', default=False, help='if true, uses a single encoder')

args = parser.parse_args()

if args.dataset == 'rot_noisy':
    img_size = (1, 28, 28)
    num_classes=10
elif args.dataset == 'cifar10':
    img_size = (3, 32, 32)
    num_classes=10

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:%d" % args.device)

def create_loaders(batch_size):
    """
    Create rot_noisy dataloaders
    """
    if args.dataset == 'rot_noisy':
        multiview_dataset = utils.data_utils.multiview_dataset_maker.rot_noisy_multiview_dataset(dataset_id=args.dataset)


        multiview_test =  utils.data_utils.multiview_dataset_maker.rot_noisy_multiview_dataset(dataset_id=args.dataset, is_testset=True)
    
    if args.dataset == 'cifar10':
        multiview_dataset = utils.data_utils.multiview_dataset_maker.rot_noisy_multiview_dataset(dataset_id=args.dataset)


        multiview_test =  utils.data_utils.multiview_dataset_maker.rot_noisy_multiview_dataset(dataset_id=args.dataset, is_testset=True)

    multiview_loader = DataLoader(dataset=multiview_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, pin_memory=True)
    multiview_test_loader = DataLoader(dataset=multiview_test, 
                            batch_size=batch_size, 
                            shuffle=True, pin_memory=True)

    print('loaded datasets')
    return (multiview_loader, multiview_test_loader, batch_size)

def train_model(conf):
    """
    Train a model with a specific configuration
    """
    model_name, latent_dim, loaders = conf
    train_loader, test_loader, batch_size = loaders

    if not args.force_retrain and \
        tu.model_exists(model_name, batch_size, args.num_epochs, latent_dim, experiment_name=args.experiment_name, architecture=args.arch):

        print("Found model for (%s, %d, %d)" % (model_name, latent_dim, batch_size))
    
    else:
        print("Training model for (%s, %d, %d)" % (model_name, latent_dim, batch_size))

        tu.train_model(model_name, train_loader, test_loader, device=device, lr=args.lr, recon_dims=args.recon,
            num_epochs=args.num_epochs, latent_dim=latent_dim, linear_size = args.linear_size, img_size=img_size, architecture=args.arch,
            show_progress=args.verbose, num_classes=num_classes, show_subprogress=args.verbose, experiment_name=args.experiment_name, single_encoder=args.single, dataset_name=args.dataset)


        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def evaluate_models():
    results = []
    for i in range(len(args.batch_size)):
        _, test_loader, _ = create_loaders(args.batch_size[i])
        acc = cv.compare_classification_accuracy(test_loader, latent_dims=args.latent_dim, num_classes=num_classes,
            batch_size=args.batch_size[i], epoch=args.num_epochs, experiment_name=args.experiment_name, model_names=args.models, architecture=args.arch, img_size=img_size, device=device)

        results.append(acc)

    results = np.concatenate(results, axis=1)

    plt.plot(args.batch_size, results[:, 0], label='DiME')
    plt.plot(args.batch_size, results[:, 1], label='CCA')
    plt.plot(args.batch_size, results[:, 2], label='Classifier')
    plt.legend()

def run():
    """
    Training many models with different configurations on rot_noisy dataset. 
    Can be in parallel if -np argument is supplied
    """
    # setup dataloaders
    dataloaders = []
    for bs in args.batch_size:
        dataloaders.append(create_loaders(bs))

    # get all configurations of models
    configurations = itertools.product(args.models, args.latent_dim, dataloaders)

    if not args.eval or args.traineval:
        # start parallel running
        multi_pool = mp.Pool(processes=args.num_parallel)
        multi_pool.map(train_model, configurations)
        multi_pool.close() 
        multi_pool.join()
    if args.eval or args.traineval:
        evaluate_models()

if __name__ == "__main__":
    # needed for pytorch parallelism
    torch.multiprocessing.set_start_method('spawn')
    run()
