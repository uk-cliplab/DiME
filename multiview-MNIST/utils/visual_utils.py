import torch
import torchvision
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_losses(model_name, list_of_losses):
    """
    Plot training losses of model
    """
    plt.plot(list(range(len(list_of_losses))), list_of_losses)
    plt.title("%s - Training Loss vs Epochs" % model_name)
    plt.show()
    
def plot_tsne(model_name, model, data_loader, multiview=False, instant_plot=True, **kwargs):
    """
    Plot TSNE embedding of a multiview model. Has view 1 and view 2 side-by-side
    """

    # get device of model
    device = next(model.parameters()).device
    batch_data, batch_labels = next(iter(data_loader))
    

    perplexity = kwargs['perplexity'] if 'perplexity' in kwargs else 30
    ind = kwargs['ind'] if 'ind' in kwargs else 0

    fig, ax = plt.subplots(1,2,figsize=(13,6))

    # get view 1 embeddings
    model.change_view(0)
    enc, _, _,_ = model(batch_data[0].to(device))
    enc = enc.cpu().detach().numpy()
    embedding_tsne = TSNE(2, perplexity=perplexity)
    embedding_tsne_result = embedding_tsne.fit_transform(enc)

    batch_labels = [i.item()%10 for i in batch_labels]
    palette = sns.color_palette("bright", len(set(batch_labels)))

    sns.scatterplot(embedding_tsne_result[:,0], embedding_tsne_result[:,1], ax=ax[0], hue=batch_labels, legend=False, palette=palette)
    
    # plot view 1 embeddings
    ax[0].tick_params(bottom = False)
    ax[0].tick_params(left = False)
    ax[0].set(xticklabels=[])
    ax[0].set(yticklabels=[])
    #ax[0].set_aspect('equal')
    ax[0].set_title('MNIST Image (View 1)')
    
    # get view 2 embeddings
    model.change_view(1)
    enc, _, _,_ = model(batch_data[1].to(device))
    enc = enc.cpu().detach().numpy()
    embedding_tsne = TSNE(2, perplexity=perplexity)
    embedding_tsne_result = embedding_tsne.fit_transform(enc)
    
    sns.scatterplot(embedding_tsne_result[:,0], embedding_tsne_result[:,1], ax=ax[1], hue=batch_labels, legend='full', palette=palette)
    
    # plot view 2 embeddings
    ax[1].tick_params(bottom = False)
    ax[1].tick_params(left = False)
    ax[1].set(xticklabels=[])
    ax[1].set(yticklabels=[])
    #ax[1].set_aspect('equal')
    ax[1].set_title('MNIST Image (View 2)')


    if 'custom_labels' in kwargs:
        plt.legend(labels=kwargs['custom_labels'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0) 
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0) 
    
    plt.show()
    
def plot_latent_space(model_name, model, data_loader, multiview=False):
    """
    Plot raw latent space of a model. Assumes that it is 2-D
    """
    # get device of model
    device = next(model.parameters()).device
    batch_data, batch_labels = next(iter(data_loader))
    
#     if multiview:
#         _, batch_data = batch_data
#         model.change_view(1)
    
    fig, ax = plt.subplots(1,2,figsize=(13,6))
        
    model.change_view(0)
    enc, _, _,_ = model(batch_data[0].to(device))
    enc = enc.cpu().detach().numpy()
    
    colors = ['red','green','blue','purple', 'yellow', 'orange', 'black', 'cyan', 'magenta', 'gray']


    for g in np.unique(batch_labels):
        ix = np.where(batch_labels == g)

        if enc.shape[1] == 1:
            ax[0].hist(enc[ix, 0], bins=10)
        else:
            ax[0].scatter(enc[ix,0], enc[ix,1], c = colors[g], label = g, s = 100)

    #ax.set_title('%s Noisy embeddings' % model_name)
    ax[0].tick_params(bottom = False)
    ax[0].tick_params(left = False)
    ax[0].set(xticklabels=[])
    ax[0].set(yticklabels=[])
    ax[0].set_title('Rotated Embeddings')
    
            
    model.change_view(1)
    enc, _, _,_ = model(batch_data[1].to(device))
    enc = enc.cpu().detach().numpy()
    
    colors = ['red','green','blue','purple', 'yellow', 'orange', 'black', 'cyan', 'magenta', 'gray']
    for g in np.unique(batch_labels):
        ix = np.where(batch_labels == g)

        if enc.shape[1] == 1:
            ax[1].hist(enc[ix, 0], bins=10)
        else:
            ax[1].scatter(enc[ix,0], enc[ix,1], c = colors[g], label = g, s = 100)

    #ax.set_title('%s Noisy embeddings' % model_name)
    ax[1].tick_params(bottom = False)
    ax[1].tick_params(left = False)
    ax[1].set(xticklabels=[])
    ax[1].set(yticklabels=[])
    ax[1].set_title('Noisy Embeddings')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    #plt.savefig("multilatent/%d.png" % model_name, bbox_inches='tight') 

    
    plt.show()

    
def see_reconstructions(model_name, model, data_loader, multiview=False, used_dataset=1, img_size=(1, 28, 28)):
    """
    Plot model reconstructions. Side-by-side of view 1 and view 2
    """
    device = next(model.parameters()).device
    batch_data, batch_labels = next(iter(data_loader))
    
    if multiview:
        if used_dataset == 1:
            _, batch_data = batch_data
        else:
            batch_data, _ = batch_data
    
    model.change_view(used_dataset)

    _, dec, _, _ = model(batch_data.to(device))
    dec = dec.cpu().detach()
    
    true_imgs = torch.zeros((25, *img_size))
    reconstruction_imgs = torch.zeros((25, *img_size))

    for i in range(25):
        true_imgs[i] = batch_data[i].reshape(img_size)
        reconstruction_imgs[i] = dec[i].reshape(img_size)
    
    true_grid = torchvision.utils.make_grid(true_imgs, nrow=5)
    reconstruction_grid = torchvision.utils.make_grid(reconstruction_imgs, nrow=5)
    
    plt.figure(figsize=[8, 8])
    fig, ax = plt.subplots(1,2)
    
    ax[0].imshow(true_grid.permute((1, 2, 0)))
    ax[1].imshow(reconstruction_grid.permute((1, 2, 0)))
    ax[0].set_title("True Images")
    ax[1].set_title("%s Reconstructed Images" % model_name)

    plt.show()
    
def visualize_multiview_dataset(data_loader, img_size=(1, 28, 28), sidesize=8):
    """
    Plot small batch of multiview dataset. 25 image grid, Side-by-side of view 1 and view 2
    """

    rotated_imgs = torch.zeros((sidesize**2, *img_size))
    noisy_imgs = torch.zeros((sidesize**2, *img_size))
    
    batch_data, batch_labels =next(iter(data_loader))
    rotated_data, noisy_data = batch_data

    for i in range(sidesize**2):
        rotated_imgs[i] = rotated_data[i].reshape(img_size)
        noisy_imgs[i] = noisy_data[i].reshape(img_size)

    fig, ax = plt.subplots(1,2, figsize=[10,10])

    rotated_grid = torchvision.utils.make_grid(rotated_imgs, nrow=sidesize)
    ax[0].imshow(rotated_grid.permute((1, 2, 0)))
    ax[0].set_title("View 1 (rotated)")

    ax[0].tick_params(bottom = False)
    ax[0].tick_params(left = False)
    ax[0].set(xticklabels=[])
    ax[0].set(yticklabels=[])

    noisy_grid = torchvision.utils.make_grid(noisy_imgs, nrow=sidesize)
    ax[1].imshow(noisy_grid.permute((1, 2, 0)))
    ax[1].set_title("View 2 (noisy)")

    ax[1].tick_params(bottom = False)
    ax[1].tick_params(left = False)
    ax[1].set(xticklabels=[])
    ax[1].set(yticklabels=[])
    plt.show()
    
def full_model_visualization(model_name, model, data_loader, img_size=(1,28,28), ismultiview=True, visualize_tsne=False):
    """
    Helper function to generate multiple visualizations
    """
    #see_reconstructions(model_name, model, data_loader, multiview=ismultiview, used_dataset=0, img_size=img_size)
    # see_reconstructions(model_name, model, data_loader, multiview=ismultiview, used_dataset=1)
    #plot_losses(model_name, training_losses)
    if visualize_tsne:
        #plot_tsne(model_name, model, data_loader, multiview=ismultiview, used_dataset=0)
        plot_tsne(model_name, model, data_loader, multiview=ismultiview)