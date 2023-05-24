from asyncio import new_event_loop
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from models import *
from utils.visual_utils import *
from losses import * 
from utils.evaluation_metrics import *
    
def mvmim_epoch(model, trainloader, optimiser, device, recon_losses=None, latent_losses=None, classifier_losses=None, show_progress=False, epoch_num=None):
    """
    Runs an epoch for a multiview autoencoder
    """

    # check that losses are properly set
    if recon_losses == [] and latent_losses == [] and classifier_losses == []:
        raise ValueError("MUST CHOOSE A LOSS")
    
    # store losses to save them per epoch
    train_loss = 0.0
    stored_recon_losses, stored_latent_losses, stored_classifier_losses = 0.0, 0.0, 0.0

    model.train()
    for batch_idx, (data, labels) in enumerate(trainloader):
        rotated_data, noisy_data = data
        rotated_data = rotated_data.to(device)
        noisy_data = noisy_data.to(device)
        labels = labels.to(device)

        # get model outputs
        model.change_view(0)
        rotated_embeddings, rotated_reconstructions, full_rotated_embeddings, rotated_classifications = model(rotated_data)

        model.change_view(1)
        noisy_embeddings, noisy_reconstructions, full_noisy_embeddings, noisy_classifications = model(noisy_data)

        # compute recon losses
        recon_loss = torch.zeros(1, device=device)
        for loss_fun in recon_losses:  
            rotated_reconstruction_loss = loss_fun(rotated_reconstructions, rotated_data)
            noisy_reconstruction_loss = loss_fun(noisy_reconstructions, noisy_data)
            recon_loss = recon_loss + noisy_reconstruction_loss + rotated_reconstruction_loss
        
        # compute latent losses
        latent_loss = torch.zeros(1, device=device)
        for loss_fun in latent_losses:
            if loss_fun.__name__ == 'Conditional_Entropy_Embedding_Loss':
                new_lossA = loss_fun(full_rotated_embeddings[:, :-model.recon_latent_dim], full_rotated_embeddings[:, -model.recon_latent_dim:], labels=labels, sigma_y=model.sigma_y, printstuff=False)
                new_lossB = loss_fun(full_noisy_embeddings[:, :-model.recon_latent_dim], full_noisy_embeddings[:, -model.recon_latent_dim:], labels=labels, sigma_y=model.sigma_y, printstuff=False)
                new_loss = new_lossA + new_lossB
            else:
                new_loss = loss_fun(rotated_embeddings, noisy_embeddings, sigma_x=model.sigma_x, sigma_y=model.sigma_y)

            latent_loss = latent_loss + new_loss
        
        # compute classifier losses
        classifier_loss = torch.zeros(1, device=device)
        for loss_fun in classifier_losses:
            rotated_classifier_loss = torch.nn.functional.cross_entropy(rotated_classifications, labels)
            noisy_classifier_loss = torch.nn.functional.cross_entropy(noisy_classifications, labels)
            
            classifier_loss = classifier_loss + rotated_classifier_loss + noisy_classifier_loss
    
        
        # compute overall loss
        total_loss = recon_loss + classifier_loss - latent_loss
        
        # backprop
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()
        train_loss += total_loss.item() * rotated_data.shape[0]
        
        # log
        if show_progress and (batch_idx % 30 == 0):
            print("Batch %d / %d  - Total %.2f - Recon %.3f - Latent %.6f - Classifier %.3f - Sigx %.2f - Sigy %.2f" % 
                  (batch_idx, len(trainloader), total_loss.item(), recon_loss.item(), latent_loss.item(), classifier_loss.item(), model.sigma_x, model.sigma_y))
            
        # store losses for later
        stored_recon_losses += recon_loss.item() * rotated_data.shape[0]
        stored_latent_losses += latent_loss.item() * rotated_data.shape[0]
        stored_classifier_losses += classifier_loss.item() * rotated_data.shape[0]

    train_loss /= len(trainloader.dataset)
    stored_recon_losses /= len(trainloader.dataset)
    stored_latent_losses /= len(trainloader.dataset)
    stored_classifier_losses /= len(trainloader.dataset)
        
    return train_loss, stored_recon_losses, stored_latent_losses, stored_classifier_losses

def train_model(model_name, train_loader, test_loader, device, latent_dim, recon_dims=0, lr=0.005, num_epochs=50, show_progress=False, show_subprogress=False, 
linear_size=1024, img_size=(1,28,28), num_classes=10, visualize_tsne=False, save_finished_model=True, experiment_name="", architecture="MLP", single_encoder=False, dataset_name=None):
    """
    Initializes a model and handles its training
    One of the main functions in the codebase

    TODO - clean this rats nest of a function signature
    """
    
    ismultiview = 'mv' in model_name
    batch_size = int(len(train_loader.dataset) / len(train_loader))
    is_stacked = dataset_name == "stacked"

    # find out what losses to use based on the model name
    if "dualclassifier" in model_name:
        is_dualclassifier = True

    else:
        epoch_functor = mvmim_epoch
        model = Multiview_Autoencoder(latent_dim, num_classes=num_classes, recon_latent_dim=recon_dims, linear_size = linear_size, img_size=img_size, architecture=architecture, single_encoder=single_encoder, is_stacked=is_stacked)
    
    model = model.to(device)
    
    # choose losses
    recon_loss, latent_loss, classifier_loss = [], [], []
    if "doe" in model_name:
        latent_loss.append(DOE_Fixed_Embedding_Loss)
    if "cca" in model_name:
        latent_loss.append(CCA_Embedding_Loss)
    if "cka" in model_name:
        latent_loss.append(CKA_Loss)
    if "recon" in model_name:
        recon_loss.append(nn.MSELoss())
    if "classifier" in model_name:
        classifier_loss.append(nn.CrossEntropyLoss())
    if "conditional" in model_name:
        latent_loss.append(Conditional_Entropy_Embedding_Loss)

    
    # setup optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # check if model is using a pretrained feature extractor
    if "postclass" in model_name:
        latent_loss = []
        recon_loss = []
        classifier_loss.append(nn.CrossEntropyLoss())

        basename = '_'.join(model_name.split('_')[:-1])
        print(basename)
        model = load_model(basename, batch_size=batch_size, epoch=num_epochs, num_classes=num_classes, latent_dim=latent_dim, experiment_name=experiment_name, 
            linear_size=linear_size, img_size=img_size, architecture=architecture)
        model.train()
        model = model.to(device)

        # freeze pretrained feature extractor
        if "frozen" in model_name:
            print('freezing')

            if is_stacked:
                params = list(model.classifier_view0_0.parameters()) + list(model.classifier_view1_0.parameters()) + \
                         list(model.classifier_view0_1.parameters()) + list(model.classifier_view1_1.parameters()) + \
                         list(model.classifier_view0_2.parameters()) + list(model.classifier_view1_2.parameters())
            else:
                params = list(model.classifier_view0.parameters()) + list(model.classifier_view1.parameters())
            optimiser = torch.optim.Adam(params, lr=lr)

        else:
            print('not_freezing')
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=20, gamma=0.5)
    
    for epoch in range(num_epochs):
        if "class" in model_name:
            view0_acc = test_model(model, test_loader, device, multiview=True, which_view=0)
            view1_acc = test_model(model, test_loader, device, multiview=True, which_view=1)

            print("Epoch %d   Acc %.5f   %.5f" % (epoch+1, view0_acc, view1_acc), optimiser.param_groups[0]['lr'])
        else:
            print("Epoch %d" % (epoch+1))

        # run an epoch
        train_loss, train_recon_loss, train_latent_loss, train_classifier_loss = epoch_functor(model, train_loader, 
                                optimiser, device, recon_losses=recon_loss, latent_losses=latent_loss, classifier_losses=classifier_loss, 
                                show_progress=show_subprogress, epoch_num=epoch)

        # increment scheduler
        scheduler.step()
        
        if visualize_tsne and (epoch+1)%5 == 0:
            full_model_visualization(epoch, model, test_loader, img_size=img_size, ismultiview=ismultiview, visualize_tsne=True)  
    
        if save_finished_model and ((epoch+1) % 10 == 0 or (epoch+1)==num_epochs):
            save_model(model, model_name, batch_size, epoch, latent_dim, optimiser, experiment_name=experiment_name, architecture=architecture)
    
    return model

def save_model(model, model_name, batch_size, epoch, latent_dim, optimizer, experiment_name="", architecture='MLP'):
    """
    Save model and optimizer state dicts so that training may continue later
    """
    filepath = "models/%s/%s/%s-b%d-e%d-L%d.params" % (experiment_name, model_name, architecture, batch_size, epoch+1, latent_dim)
    if not os.path.exists("models/%s/%s" % (experiment_name, model_name)):
        os.makedirs("models/%s/%s" % (experiment_name, model_name))

    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, filepath)
    
def save_losses(model_name, loss_array, batch_size, latent_dim, experiment_name=""):
    """
    Save losses to a file for plotting purposes later. TODO - move this to tensorboard
    """
    filepath = "training_losses/%s%s-b%d-L%d-a101.npy" % (experiment_name, model_name, batch_size, latent_dim)
    
    np.save(filepath, loss_array)

def save_sigmas(model_name, model, batch_size, latent_dim, experiment_name=""):
    """
    Save kernel bandwidths to a file for plotting purposes later. TODO - move this to tensorboard
    """
    filepath = "sigmas/x-%s%s-b%d-L%d.txt" % (experiment_name, model_name, batch_size, latent_dim)
    np.savetxt(filepath, model.sigma_x_history)

    filepath = "sigmas/y-%s%s-b%d-L%d.txt" % (experiment_name, model_name, batch_size, latent_dim)
    np.savetxt(filepath, model.sigma_y_history)
    
def load_model(model_name, batch_size, epoch, latent_dim, experiment_name="", architecture='MLP', **kwargs):
    """
    Load model from a file and, by default, set it to eval mode
    """

    recon_latent_dim = kwargs['recon_latent_dim'] if 'recon_latent_dim' in kwargs else 0
    img_size = (1,28,28) if 'img_size' not in kwargs else kwargs['img_size']
    linear_size = 1024 if 'linear_size' not in kwargs else kwargs['linear_size']
    num_classes = 10 if 'num_classes' not in kwargs else kwargs['num_classes']

    filepath = "models/%s/%s/%s-b%d-e%d-L%d.params" % (experiment_name, model_name, architecture, batch_size, epoch, latent_dim)

    model = Multiview_Autoencoder(latent_dim, recon_latent_dim=recon_latent_dim, img_size=img_size, num_classes=num_classes, linear_size=linear_size, architecture=architecture)


    checkpoint = torch.load(filepath, map_location='cpu')
   
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model


def model_exists(model_name, batch_size, epoch, latent_dim, experiment_name="", architecture='CNN'):
    """
    Check if a model with certain configuration already exists in the models folder
    """

    filepath = "models/%s/%s/%s-b%d-e%d-L%d.params" % (experiment_name, model_name, architecture, batch_size, epoch, latent_dim)
    print(filepath)
    
    return os.path.exists(filepath)
