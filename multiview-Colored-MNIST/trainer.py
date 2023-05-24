import torch
import matplotlib.pyplot as plt
import numpy as np
import math

from utils.visual_utils import display_mult_images

from losses import diffEntropies, diffEntropiesLabels, latentFactorDivergence,conditionalEntropy, latentFactorDivergenceModified, mutualInformation, matrixJRDivergence
from utils.shuffled_pairs import find_pair
from models import BigModel,BigOldModel, BigOld_Convolutional_Encoder, BigOld_Convolutional_Decoder, Simple_Classifier
from sklearn import manifold
import torch.nn.functional as F

def mutiviewAEShuffled(dataloader,
    tradeoff_1=1.0,
    tradeoff_2=1.0,
    shared_dim=10,
    latent_dim = 4,
    exclusive_dim=2,
    exclusive_dim1 = 1,
    exclusive_dim2 = 1,
    batch_size = 128,
    n_epochs=10,
    use_cuda=True,
    print_every=200,
    plot_every=1000,
    save_every = 10,
    path_ckpoint = None
):
    z_dim = shared_dim + exclusive_dim + latent_dim
    # model = BigModel(latent_dim = z_dim)
    
    model = BigOldModel(z_dim = z_dim)

    
    if use_cuda:
        device = torch.device('cuda:1')
        model = model.to(device)

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=1.0)
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            view1 = images.bg
            view2 = images.fg
            digit_label = images.digit_label
            view1, view2 = find_pair(view1,view2,digit_label)
            
            x = torch.cat((view1, view2), 0)
            label_view = torch.zeros(x.shape[0],dtype = torch.int64)   
            label_view[:view1.shape[0]] = 1

            prior_exclusive = torch.randint(1, 13, (x.shape[0],exclusive_dim))/12 # prior of 12 colors

            x = x.to(device)
            prior_exclusive = prior_exclusive.to(device)
            label_view = label_view.to(device)
                
            z, x_reconstructed = model(x)
            z_shared = z[:,:shared_dim]
            z_latent = z[:,shared_dim:(shared_dim+latent_dim)]
            z_exclusive = z[:,-exclusive_dim:]
            z_shared_v1 = z_shared[:view1.shape[0],:]
            z_shared_v2 = z_shared[view1.shape[0]:,:]


            doe = diffEntropies(z_shared_v1,z_shared_v2)
            # div_exclusive = latentFactorDivergence(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            div_exclusive = latentFactorDivergenceModified(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            
            cond_entropy = conditionalEntropy(z_shared,z_exclusive) + conditionalEntropy(z_latent,z_exclusive)
            # mutual_Information = mutualInformation(z_shared,z_unshared)
            # cond_entropy = (conditionalEntropy(z_shared_v1,z_exclusive_v1) + conditionalEntropy(z_shared_v2,z_exclusive_v2))/2
            nll = (x_reconstructed - x).pow(2).mean()
            loss = 5*nll - tradeoff_1 * doe + tradeoff_2 *div_exclusive - 0.01*cond_entropy 
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % print_every == 0:
                print("Negative log likelihood is {:.5f}, DoE is {:.5f}, divergence exclusive is {:.5f}, Cond entropy is {:.5f}".format(
                    nll.item(), doe.item(),div_exclusive.item(), cond_entropy.item()))
            if i % plot_every == 0:

                print("epoch {:.1f}".format(epoch))
                samples = x_reconstructed[torch.randint(0,2*batch_size,(25,))]
                samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
                display_mult_images(samples, 5, 5)

                plt.show()
        if epoch % save_every == 0:
            if path_ckpoint == None:
                path = r'multiview_epoch{}.pth'.format(epoch)
            # don't include pth    
            path = path_ckpoint + '_epoch{}.pth'.format(epoch)
                
            torch.save(model.state_dict(), path)
                
    return model

def mutiviewTranslate(dataloader,
    tradeoff_1=1.0,
    tradeoff_2=1.0,
    shared_dim=10,
    exclusive_dim=2,
    exclusive_dim1 = 1,
    exclusive_dim2 = 1,
    batch_size = 128,
    n_epochs=10,
    use_cuda=True,
    print_every=200,
    plot_every=1000,
    save_every = 10,
    path_ckpoint = None
):
    z_dim = shared_dim + exclusive_dim
    # model = BigModel(latent_dim = z_dim)
    model = BigOldModel(z_dim = z_dim)

    
    if use_cuda:
        device = torch.device('cuda:1')
        model = model.to(device)

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=1.0)
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            view1 = images.bg
            view2 = images.fg
            
            x = torch.cat((view1, view2), 0)
            label_view = torch.zeros(x.shape[0],dtype = torch.int64)   
            label_view[:view1.shape[0]] = 1

            prior_exclusive = torch.rand(x.shape[0], exclusive_dim)
    
            x = x.to(device)
            prior_exclusive = prior_exclusive.to(device)
            label_view = label_view.to(device)
                
            z, x_reconstructed = model(x)
            z_shared = z[:,:shared_dim]
            z_exclusive = z[:,-exclusive_dim:]
            z_shared_v1 = z_shared[:view1.shape[0],:]
            z_shared_v2 = z_shared[view1.shape[0]:,:]
            z_exclusive_v1 = z_exclusive[:view1.shape[0],:]
            z_exclusive_v2 = z_exclusive[view1.shape[0]:,:]
            z_swapped_v1 = torch.cat((z_shared_v2,z_exclusive_v1),1)
            z_swapped_v2 = torch.cat((z_shared_v1,z_exclusive_v2),1)
            z_swapped = torch.cat((z_swapped_v1,z_swapped_v2),0)
            x_cross_domain = model.decoder(z_swapped)


            doe = diffEntropies(z_shared_v1,z_shared_v2)
            div_exclusive = latentFactorDivergence(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            # div_exclusive = latentFactorDivergenceModified(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            cond_entropy = conditionalEntropy(z_shared,z_exclusive)
            # cond_entropy = conditionalEntropy(z_shared_v1,z_exclusive_v1) + conditionalEntropy(z_shared_v2,z_exclusive_v2)

            nll = ((x_reconstructed - x).pow(2).mean() + (x_cross_domain - x).pow(2).mean())/2
            loss = 5*nll - tradeoff_1 * doe + tradeoff_2 *div_exclusive - 0.07*cond_entropy 
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % print_every == 0:
                print("Negative log likelihood is {:.5f}, DoE is {:.5f}, divergence exclusive is {:.5f}, conditional Entropy is {:.5f}".format(
                    nll.item(), doe.item(),div_exclusive.item(), cond_entropy.item()))
            if i % plot_every == 0:

                print("epoch {:.1f}".format(epoch))
                samples = x_reconstructed[torch.randint(0,2*batch_size,(25,))]
                samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
                display_mult_images(samples, 5, 5)

                plt.show()
        if epoch % save_every == 0:
            if path_ckpoint == None:
                path = r'multiview_epoch{}.pth'.format(epoch)
            # don't include pth    
            path = path_ckpoint + '_epoch{}.pth'.format(epoch)
                
            torch.save(model.state_dict(), path)
                
    return model



def mutiviewAE(dataloader,
    tradeoff_1=1.0,
    tradeoff_2=1.0,
    shared_dim=10,
    exclusive_dim=2,
    exclusive_dim1 = 1,
    exclusive_dim2 = 1,
    batch_size = 128,
    n_epochs=10,
    use_cuda=True,
    print_every=200,
    plot_every=1000,
    save_every = 10,
    path_ckpoint = None
):
    z_dim = shared_dim + exclusive_dim
    # model = BigModel(latent_dim = z_dim)
    model = BigOldModel(z_dim = z_dim)

    
    if use_cuda:
        device = torch.device('cuda:1')
        model = model.to(device)

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=1.0)
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            view1 = images.bg
            view2 = images.fg
            
            x = torch.cat((view1, view2), 0)
            label_view = torch.zeros(x.shape[0],dtype = torch.int64)   
            label_view[:view1.shape[0]] = 1

            prior_exclusive = torch.rand(x.shape[0], exclusive_dim) - 1 
    
            x = x.to(device)
            prior_exclusive = prior_exclusive.to(device)
            label_view = label_view.to(device)
                
            z, x_reconstructed = model(x)
            z_shared = z[:,:shared_dim]
            z_exclusive = z[:,-exclusive_dim:]
            z_shared_v1 = z_shared[:view1.shape[0],:]
            z_shared_v2 = z_shared[view1.shape[0]:,:]
            # z_exclusive_v1 = z_exclusive[:view1.shape[0],:]
            # z_exclusive_v2 = z_exclusive[:view1.shape[0],:]

            
            L1 = (torch.abs(z_shared_v1 - z_shared_v2)).mean()
            doe = diffEntropies(z_shared_v1,z_shared_v2)
            # div_exclusive = latentFactorDivergence(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            div_exclusive = latentFactorDivergenceModified(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            
            cond_entropy = conditionalEntropy(z_shared,z_exclusive)
            # cond_entropy = conditionalEntropy(z_shared_v1,z_exclusive_v1) + conditionalEntropy(z_shared_v2,z_exclusive_v2)
            nll = (x_reconstructed - x).pow(2).mean()
            loss = 5*nll - tradeoff_1 * doe + tradeoff_2 *div_exclusive - 0.05*cond_entropy + 0.1*L1 # before 0.01 + 
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % print_every == 0:
                print("Negative log likelihood is {:.5f}, DoE is {:.5f}, divergence exclusive is {:.5f}, conditional Entropy is {:.5f}, L1 is {:.5f}".format(
                    nll.item(), doe.item(),div_exclusive.item(), cond_entropy.item(), L1.item()))
            if i % plot_every == 0:

                print("epoch {:.1f}".format(epoch))
                samples = x_reconstructed[torch.randint(0,2*batch_size,(25,))]
                samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
                display_mult_images(samples, 5, 5)

                plt.show()
        if epoch % save_every == 0:
            if path_ckpoint == None:
                path = r'multiview_epoch{}.pth'.format(epoch)
            # don't include pth    
            path = path_ckpoint + '_epoch{}.pth'.format(epoch)
                
            torch.save(model.state_dict(), path)
                
    return model


def mutiviewShared(dataloader,
    shared_dim=10,
    batch_size = 128,
    n_epochs=10,
    use_cuda=True,
    print_every=100,
    plot_every = 500
):
    z_dim = shared_dim
    # model = BigModel(latent_dim = z_dim)
    model = BigOld_Convolutional_Encoder(latent_dim = z_dim)
    
    if use_cuda:
        device = torch.device('cuda:1')
        model = model.to(device)

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=1.0)
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            view1 = images.bg
            view2 = images.fg
            digit_label = images.digit_label
            labels = torch.cat((digit_label,digit_label),0)
            view1, view2 = find_pair(view1,view2,digit_label)
            
            x = torch.cat((view1, view2), 0)
            x = x.to(device)
                
            z = model(x)
            z_shared_v1 = z[:view1.shape[0],:]
            z_shared_v2 = z[view1.shape[0]:,:]
            doe = diffEntropies(z_shared_v1,z_shared_v2)

            loss = -1*doe 
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % print_every == 0:
                print(" DoE is {:.5f}".format(doe.item()))
            if i % plot_every == 0:

                print("epoch {:.1f}".format(epoch))
                
                shared_dim = 20
                tsne = manifold.TSNE()
                z_2d = tsne.fit_transform(z.cpu().detach().numpy())
                plt.scatter(z_2d[:,0],z_2d[:,1],c = labels)
                plt.show()
                
    return model

def multiviewExclusive(dataloader,
    shared_encoder, 
    tradeoff_1=1.0,
    shared_dim=10,
    latent_dim = 4,
    exclusive_dim=2,
    exclusive_dim1 = 1,
    exclusive_dim2 = 1,
    batch_size = 128,
    n_epochs=10,
    use_cuda=True,
    print_every=200,
    plot_every=1000,
    save_every = 10,
    path_ckpoint = None
):
    ex_dim = exclusive_dim + latent_dim
    z_dim = ex_dim + shared_dim    
    encoder_exclusive = BigOld_Convolutional_Encoder(latent_dim = ex_dim)
    decoder = BigOld_Convolutional_Decoder(latent_dim = z_dim)

    
    if use_cuda:
        device = torch.device('cuda:1')
        encoder_exclusive = encoder_exclusive.to(device)
        decoder = decoder.to(device)
        shared_encoder = shared_encoder.to(device)

    lr = 0.0001
    optimizer = torch.optim.Adam(list(encoder_exclusive.parameters()) + list(decoder.parameters()), lr=lr)
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            view1 = images.bg
            view2 = images.fg
            digit_label = images.digit_label
            view1, view2 = find_pair(view1,view2,digit_label)
            
            x = torch.cat((view1, view2), 0)
            label_view = torch.zeros(x.shape[0],dtype = torch.int64)   
            label_view[:view1.shape[0]] = 1

            prior_exclusive = torch.randint(1, 13, (x.shape[0],exclusive_dim))/12 # prior of 12 colors

            x = x.to(device)
            prior_exclusive = prior_exclusive.to(device)
            label_view = label_view.to(device)
                
            z = encoder_exclusive(x)
            
            z_exclusive = z[:,:exclusive_dim]
            z_latent = z[:,-latent_dim:]
            z_shared = shared_encoder(x)
            

            div_exclusive = latentFactorDivergence(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            # div_exclusive = latentFactorDivergenceModified(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            # cond_entropy = conditionalEntropy(z_shared,z) + conditionalEntropy(z_latent,z_exclusive)
            mutual_information = mutualInformation(z_shared,z) + mutualInformation(z_latent,z_exclusive)
            z_combined = torch.cat((z_shared,z),1)

            x_reconstructed = decoder(z_combined)
 
            nll = (x_reconstructed - x).pow(2).mean()
            loss = 5*nll  + tradeoff_1 *div_exclusive + mutual_information
            loss.backward()
            optimizer.step()
  

            if i % print_every == 0:
                print("Reconstruction error is {:.5f}, divergence exclusive is {:.5f}, mutual information is {:.5f}".format(
                    nll.item(),div_exclusive.item(), mutual_information.item()))
            if i % plot_every == 0:

                print("epoch {:.1f}".format(epoch))
                samples = x_reconstructed[torch.randint(0,2*batch_size,(25,))]
                samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
                display_mult_images(samples, 5, 5)

                plt.show()
        if epoch % save_every == 0:
            if path_ckpoint == None:
                path = r'multiview_exclusive_epoch{}.pth'.format(epoch)
            # don't include pth    
            path1 = path_ckpoint + '_encoder_epoch{}.pth'.format(epoch)
            path2 = path_ckpoint + '_decoder_epoch{}.pth'.format(epoch)    
            torch.save(encoder_exclusive.state_dict(), path1)
            torch.save(decoder.state_dict(), path2)
                
    return encoder_exclusive, decoder


def dimeDiva(dataloader,
    d_dim = 2,
    y_dim = 10,
    z_dim = 5,
    tradeoff_1 = 1,
    tradeoff_2 = 1,
    tradeoff_3 = 1,
    batch_size = 128,
    n_epochs=10,
    use_cuda=True,
    print_every=200,
    plot_every=1000,
    save_every = 10,
    path_ckpoint = None
):
    dim = d_dim + y_dim + z_dim
    # model = BigModel(latent_dim = z_dim)
    model = BigOldModel(z_dim = dim)

    
    if use_cuda:
        device = torch.device('cuda:1')
        model = model.to(device)

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            view1 = images.bg
            view2 = images.fg
            x = torch.cat((view1, view2), 0)
            label_digit = images.digit_label
            label_digit = torch.cat((label_digit,label_digit),0).long()
            label_domain = torch.zeros(x.shape[0],dtype = torch.int64)   
            label_domain[:view1.shape[0]] = 1
            label_domain = label_domain.long()

            prior_z = torch.randn(x.shape[0], z_dim)
    
            x = x.to(device)
            prior_z = prior_z.to(device)
            label_domain = label_domain.to(device)
            label_digit = label_digit.to(device)
            z, x_reconstructed = model(x)
            z_d = z[:,:d_dim]
            z_y = z[:,d_dim:(d_dim + y_dim)]
            z_z = z[:,-z_dim:]
            # z_exclusive_v1 = z_exclusive[:view1.shape[0],:]
            # z_exclusive_v2 = z_exclusive[:view1.shape[0],:]

            doe_domain = diffEntropiesLabels(z_d,label_domain, num_classes =2)
            doe_class = diffEntropiesLabels(z_y,label_digit, num_classes = 10)
            # div_exclusive = latentFactorDivergence(z_exclusive,label_view,prior_exclusive,exclusive_dim1,exclusive_dim2)
            div_prior_x = matrixJRDivergence(z_z,prior_z)
            
            
            # cond_entropy = conditionalEntropy(z_shared_v1,z_exclusive_v1) + conditionalEntropy(z_shared_v2,z_exclusive_v2)
            nll = (x_reconstructed - x).pow(2).mean()
            loss = nll - tradeoff_1 * doe_domain - tradeoff_2 *doe_class + tradeoff_3 * div_prior_x # before 0.01 + 
            loss.backward()
            optimizer.step()


            if i % print_every == 0:
                print("Rec error is {:.5f}, DiME domain is {:.5f}, DiME class is {:.5f}, divergence prior is {:.5f}".format(
                    nll.item(), doe_domain.item(),doe_class.item(), div_prior_x.item()))
            if i % plot_every == 0:

                print("epoch {:.1f}".format(epoch))
                samples = x_reconstructed[torch.randint(0,2*batch_size,(25,))]
                samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
                display_mult_images(samples, 5, 5)
                plt.show()

                tsne = manifold.TSNE(learning_rate = 'auto')
                z_2d = tsne.fit_transform(z_d.cpu().detach().numpy())
                plt.scatter(z_2d[:,0],z_2d[:,1],c = label_domain.cpu())
                plt.show()

                tsne = manifold.TSNE(learning_rate = 'auto')
                z_2d = tsne.fit_transform(z_y.cpu().detach().numpy())
                plt.scatter(z_2d[:,0],z_2d[:,1],c = label_digit.cpu())
                plt.show()


        if epoch % save_every == 0:
            if path_ckpoint == None:
                path = r'multiview_epoch{}.pth'.format(epoch)
            else:
                # don't include pth    
                path = path_ckpoint + '_epoch{}.pth'.format(epoch)
                
            torch.save(model.state_dict(), path)
                
    return model


def digit_classifier(dataloader,
    encoder,
    z_dim, 
    n_epochs=10,
    use_cuda=True,
):
    classifier = Simple_Classifier(latent_dim = z_dim , num_classes=10)
    if use_cuda:
        device = torch.device('cuda:1')
        encoder = encoder.to(device)
        classifier = classifier.to(device)


    lr = 0.0001
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            view1 = images.bg
            view2 = images.fg
            digit_label = images.digit_label
            view1, view2 = find_pair(view1,view2,digit_label)
            
            x = torch.cat((view1, view2), 0)
            x = x.to(device)
            labels = torch.cat((digit_label, digit_label), 0)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            
            z = encoder(x)
            outputs = classifier(z[:,:z_dim])  ## to avoid extra dimensions in the exclusive encoder
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
  
        print("epoch {:.1f} - Cross Entropy error is {:.5f}".format(
                    epoch, loss.item()))               
    return classifier

def test_classifier(classifier,encoder,test_loader,z_dim = 10, use_cuda=True,):
    if use_cuda:
        device = torch.device('cuda:1')
        encoder = encoder.to(device)
        classifier = classifier.to(device)
    classifier.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images in test_loader:
            view1 = images.bg
            view2 = images.fg
            digit_label = images.digit_label        
            x = torch.cat((view1, view2), 0)
            x = x.to(device)
            labels = torch.cat((digit_label, digit_label), 0)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)           
            z = encoder(x)
            outputs = classifier(z[:,:z_dim])
            test_loss += criterion(outputs,labels).item()
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= 2*len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, 2*len(test_loader.dataset),
        100. * correct / (2*len(test_loader.dataset))))


def color_classifier(dataloader,
    encoder,
    z_dim, 
    n_epochs=10,
    use_cuda=True,
    type_color = 'fg'
):
    classifier = Simple_Classifier(latent_dim = z_dim , num_classes=12)
    if use_cuda:
        device = torch.device('cuda:1')
        encoder = encoder.to(device)
        classifier = classifier.to(device)


    lr = 0.001
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            if type_color == 'fg':                
                x = images.fg
                labels = images.fg_label
            else:
                x =images.bg
                labels = images.bg_label
            x = x.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)            
            z = encoder(x)
            outputs = classifier(z[:,:z_dim])  ## to select the feature exclusive to that view
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
  
        print("epoch {:.1f} - Cross Entropy error is {:.5f}".format(
                    epoch, loss.item()))               
    return classifier

def test_color_classifier(classifier,encoder,test_loader,z_dim = 2,type_color = 'fg', use_cuda=True):
    if use_cuda:
        device = torch.device('cuda:1')
        encoder = encoder.to(device)
        classifier = classifier.to(device)
    classifier.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images in test_loader:
            if type_color == 'fg':                
                x = images.fg
                labels = images.fg_label
            else:
                x =images.bg
                labels = images.bg_label
            x = x.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)           
            z = encoder(x)
            outputs = classifier(z[:,:z_dim])
            test_loss += criterion(outputs,labels).item()
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / (len(test_loader.dataset))))