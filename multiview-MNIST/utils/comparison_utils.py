import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from IPython import display
import torch
import torchvision
from sklearn.manifold import TSNE
import seaborn as sns
import os
import glob
import models
import utils.visual_utils as vu
import utils.training_utils as tu
import pandas as pd
import utils.evaluation_metrics as em

def all_models_for_latent(test_loader, latent_dim, batch_size, epoch, experiment_name="", model_names=None):
    """
    Load a bunch of models, calculate their latent space TSNE embeddings, and save the resulting figure
    """
    print("LOADING MODELS")
    assert experiment_name != ""
    if not os.path.exists("figures/%s" % experiment_name):
        os.makedirs("figures/%s" % experiment_name)
    
    # which models to load and compare
    model_names = constants.all_model_names if model_names is None else model_names

    num_models = len(model_names)
    
    # initialize figure
    fig, ax = plt.subplots(1, num_models, figsize=(0.5+int(3*num_models), 3))
    for idx, name in enumerate(model_names):
        # load model. set to noisy view

        cur_model = tu.load_model(name, batch_size, epoch, latent_dim, experiment_name=experiment_name, architecture='CNN')
        cur_model.change_view(0)
        device = next(cur_model.parameters()).device
        is_last_plot = idx == num_models-1
        
        cur_ax = ax[idx] if num_models > 1 else ax
        
        # setup data
        batch_data, batch_labels = next(iter(test_loader))
        _, batch_data = batch_data
        
        # get embeddings and run TSNE
        enc, _, _ = cur_model(batch_data.to(device))
        enc = enc.cpu().detach().numpy()
        embedding_tsne = TSNE(2, random_state=0)
        embedding_tsne_result = embedding_tsne.fit_transform(enc)

        # plot it
        palette = sns.color_palette("bright", 10)
        sns.scatterplot(embedding_tsne_result[:,0], embedding_tsne_result[:,1], ax=cur_ax, hue=batch_labels, palette=palette, legend=is_last_plot)

        lim = (embedding_tsne_result.min()-5, embedding_tsne_result.max()+5)

        cur_ax.set_xlim(lim)
        cur_ax.set_ylim(lim)
        cur_ax.set_aspect('equal')
        cur_ax.set_title('%s' % convert_modelname_to_components(name))
        cur_ax.set(xticklabels=[])
        cur_ax.set(yticklabels=[])
        
        cur_ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        
        if is_last_plot:
            cur_ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    plt.suptitle("TSNEs for L=%d" % (latent_dim, batch_size, epoch))
    plt.savefig("figures/%s/tsne_latent-b%d-e%d-L%d" % (experiment_name, batch_size, epoch, latent_dim))
    
def compare_classification_accuracy(test_loader, latent_dims, batch_size, epoch, experiment_name="", num_classes=10, comparison_type = "max", device=None, model_names=None, architecture='MLP', img_size=(1,28,28)):
    """
    Load a bunch of models, find their test accuracies, and print the results
    """
    assert comparison_type in ["max", "mean"]
    device = torch.device('cpu') if device is None else device

    # check if experiment has multiple runs to compare
    if 'multi' in experiment_name:
        folders = glob.glob("models/%s*" % experiment_name)
        folders = [f.split('/')[1] for f in folders]
    else:
        folders = [experiment_name]

    accuracies = np.zeros((len(folders), len(model_names), len(latent_dims)))
    for folderidx, folder in enumerate(folders):
        for latentidx, ldim in enumerate(latent_dims):
            for modelidx, name in enumerate(model_names):
                # load model. set to rotated view
                        
                #experiment_name = "slow_multi_mnist" if name != "mv_cka_postclassfrozen" else "adaptcka_multi_mnist"         
                # if folder[-2] == "1":
                #     folder = experiment_name + "10"
                # else:
                #     folder = experiment_name + folder[-1]

                print(folder)
                
                cur_model = tu.load_model(name, batch_size, epoch, ldim, num_classes=num_classes,experiment_name=folder, is_stacked=False, architecture=architecture, img_size=img_size)
                cur_model.change_view(0)
                cur_model.to(device)

                acc = em.test_model(cur_model, test_loader, device, multiview=True)
                print(name, acc)
                print(100*acc)
                accuracies[folderidx][modelidx][latentidx] = 100*acc

    print(accuracies)
    np.save("full_accuracies_%s" % experiment_name, accuracies)
    if comparison_type == "max":
        accuracies = np.amax(accuracies, axis=0)
        print(accuracies)

    return accuracies

def convert_modelname_to_components(model_name):
    """
    Breaks a model name, e.g. mv_recon_doe, down into base components, "Recon+DoE". Helpful for plot titles
    """
    comps = model_name.split('_')
    comps.remove('mv')
    comps = [i.capitalize() for i in comps]
    
    return '+'.join(comps)
    
def create_graph(accuracies=None):
    accuracies = np.load("full_accuracies_multi_mnistb.npy")
    latent_dims = list(range(1,15+1))
    model_names = ["DiME", "CCA", "Fully Supervised"]
    experiment_name = "multi_mnist"
    comparison_type = "mean"

    if comparison_type == "max":
        accuracies = np.amax(accuracies, axis=0)

        latent_col = np.array([l for l in latent_dims]).reshape(len(latent_dims), 1)

        formattedComps = np.hstack((latent_col, accuracies.T))
        df = pd.DataFrame(formattedComps, columns=(["Latent Dim"] + model_names))
        print(df.to_latex(index=False))
        plt.figure()
        df.plot(x="Latent Dim")
        plt.title("Testing Accuracy vs Latent Dim (max over 10 runs)")

    if comparison_type == "min":
        accuracies = np.amin(accuracies, axis=0)

        latent_col = np.array([l for l in latent_dims]).reshape(len(latent_dims), 1)

        formattedComps = np.hstack((latent_col, accuracies.T))
        df = pd.DataFrame(formattedComps, columns=(["Latent Dim"] + model_names))
        print(df.to_latex(index=False))
        plt.figure()
        df.plot(x="Latent Dim")
        plt.title("Testing Accuracy vs Latent Dim (min)")


    elif comparison_type == "mean":
        stds = np.std(accuracies, axis=0)
        means = np.mean(accuracies, axis=0)
        
        print(stds)
        print(means)
        for j in range(means.shape[0]):
            plt.plot(latent_dims, means[j], label=model_names[j])
            plt.fill_between(latent_dims,means[j]-stds[j],means[j]+stds[j],alpha=.1)

        plt.legend()
        plt.title("Supervised Finetuning Classification Accuracy vs Latent Dim")

        stds = stds.astype('str')
        means = means.astype('str')

        for i in range(stds.shape[0]):
            for j in range(stds.shape[1]):
                means[i][j] = "%.2f" % float(means[i][j])
                stds[i][j] = " ± %.2f" % float(stds[i][j])
        accuracies = np.char.add(means, stds)

        latent_col = np.array([l for l in latent_dims]).reshape(len(latent_dims), 1)
        formattedComps = np.hstack((latent_col, accuracies.T))
        df = pd.DataFrame(formattedComps, columns=(["Latent Dim"] + model_names))
        print(df.to_latex(index=False))

    plt.xlabel("Dimensionality")
    plt.ylabel("Testing Accuracy")
    plt.xticks(range(latent_dims[0], latent_dims[-1]+1))
    # plt.yticks(list(range(0, 109, 10)))
    plt.savefig("%s_%s_accuracies.png" % (comparison_type, experiment_name))