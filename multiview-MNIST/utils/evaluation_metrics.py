from dataclasses import dataclass
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from models import *
from utils.visual_utils import *
from losses import *

def get_kmeans_accuracy(model, testloader, device, show_confidence=False, multiview=False):
    """
    Evaluate a model by running Kmeans in the latent space.

    Matches labels by looking at the mode

    Can plot a confusion matrix by enabling show_confidence
    """
    acc = 0
    conf = np.zeros((10,10), dtype=int)
    
    # test with the Kmeans model
    for data, labels in testloader:
        if multiview:
            _, data = data
            model.change_view(1)
    
        data = data.to(device)
        enc, _, _ = model(data)
        enc = enc.cpu().detach().numpy()

        kmeans = KMeans(n_clusters=10, random_state=0)
        clusters = kmeans.fit_predict(enc)

        # fix potentially permuted labels
        clusterlabels = np.zeros_like(clusters)
        for i in range(10):
            mask = (clusters == i)
            clusterlabels[mask] = mode(labels[mask])[0]
        
        acc += accuracy_score(labels, clusterlabels) * (len(data) / 10000)
        conf += confusion_matrix(labels, clusterlabels)

    # plot confusion matrix
    if show_confidence:
        sns.heatmap(conf.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()
    
    return acc

def knn_accuracy(model, trainloader, testloader, device):
    """
    Evaluate a model by running KNN in the latent space.
    """
    model.eval()

    # get encodings
    encodings_list = []
    labels_list = []
    model.change_view(0)
    for data, labels in trainloader:
        data, _ = data
        data = data.to(device)
        enc, _,_,_= model(data)

        encodings_list.append(enc.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())

    encodings_list = np.concatenate(encodings_list)
    labels_list = np.concatenate(labels_list)

    # fit the KNN
    neigh = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    neigh.fit(encodings_list, labels_list)

    #test with the KNN model
    acc, total_samples = 0, 0
    for idx, (data, labels) in enumerate(testloader):
        data, _ = data
        data = data.to(device)
        enc, _,_,_= model(data)

        classifications = neigh.predict(enc.cpu().detach().numpy())
        acc += (classifications == labels.cpu().numpy()).sum()
        total_samples += labels.shape[0]

    return acc / total_samples

def spectral_cluster_acc(model, testloader, device, show_confidence=True, multiview=False):
    """
    Evaluate a model by finding spectral clustering embeddings, and then running Kmeans.

    Matches labels by looking at the mode

    Can plot a confusion matrix by enabling show_confidence
    """

    acc = 0
    conf = np.zeros((10,10), dtype=int)
    
    # test with the KNN model
    for data, labels in testloader:
        if multiview:
            _, data = data
            model.change_view(1)
    
        data = data.to(device)
        enc, _, _ = model(data)
        enc = enc.cpu().detach().numpy()

        kmeans = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', assign_labels='kmeans')
        clusters = kmeans.fit_predict(enc)

        # fix potentially permuted labels
        clusterlabels = np.zeros_like(clusters)
        for i in range(10):
            mask = (clusters == i)
            clusterlabels[mask] = mode(labels[mask])[0]
        
        acc += accuracy_score(labels, clusterlabels) * (len(data) / 10000)
        conf += confusion_matrix(labels, clusterlabels)

    if show_confidence:
        sns.heatmap(conf.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()
    
    return acc / total_samples

def avg_spectral_acc(model, testloader, device, num_iters=20):
    """
    Runs spectral_cluster_acc over num_iters iterations.

    Inputs:
        model
        testloader
        device
        num_iters
    Outputs:
        mean, std of spectral clustering accuracy
    """

    results = np.zeros(num_iters)
    for i in range(num_iters):
        acc = spectral_cluster_acc(model, testloader, device, multiview=True)
        results[i] = acc

    return np.mean(results), np.std(results)


def test_model(model, test_loader, device, multiview=True, which_view=0):
    """
    For models with classification functionality, returns accuracy on a test set
    """

    acc = 0
    total_samples = 0

    model.eval()
    model.change_view(which_view)

    for idx, (data, labels) in enumerate(test_loader):
        if multiview:
            if which_view == 0:
                data, _ = data
            else:
                _, data = data

        data, labels = data.to(device), labels.to(device)

        _, _, _, classifier_outputs = model(data)

        if not model.is_stacked:
            prob = nn.functional.softmax(classifier_outputs, dim = 1)
            pred = torch.max(prob, dim=1)[1].detach().cpu().numpy()
            acc += (pred == labels.cpu().numpy()).sum()
            total_samples += labels.shape[0]
        else:
            c_a, c_b, c_c = classifier_outputs

            l_c = labels % 10
            l_b = torch.floor(labels/10) % 10
            l_a = torch.floor(labels/100) % 10

 
            prob_a = nn.functional.softmax(c_a, dim = 1)
            pred_a = torch.max(prob_a, dim=1)[1].detach().cpu().numpy()

            prob_b = nn.functional.softmax(c_b, dim = 1)
            pred_b = torch.max(prob_b, dim=1)[1].detach().cpu().numpy()

            prob_c = nn.functional.softmax(c_c, dim = 1)
            pred_c = torch.max(prob_c, dim=1)[1].detach().cpu().numpy()

            combined_predictions = (pred_a*100) + (pred_b * 10) + pred_c
            print(labels[0:10], pred_a[0:10], pred_b[0:10], pred_c[0:10])

            print(combined_predictions[0:10])

            acc += (combined_predictions == labels.cpu().numpy()).sum()
            total_samples += labels.shape[0]


    print(acc, total_samples)
    accuracy = acc / total_samples

    return accuracy

def get_nuclear_norm(model, test_loader, device, multiview=True):
    total_rot_embeddings = None
    total_noisy_embeddings = None

    for idx, (data, labels) in enumerate(test_loader):
        rot_data, noisy_data = data
        rot_data, noisy_data, labels = rot_data.to(device), noisy_data.to(device), labels.to(device)

        model.change_view(0)
        rot_embeddings, _, _,_ = model(rot_data)
        if total_rot_embeddings is None:
            total_rot_embeddings = rot_embeddings
        else:
            total_rot_embeddings = torch.cat((total_rot_embeddings, rot_embeddings))

        model.change_view(1)
        noisy_embeddings, _, _,_ = model(noisy_data)
        if total_noisy_embeddings is None:
            total_noisy_embeddings = noisy_embeddings
        else:
            total_noisy_embeddings = torch.cat((total_noisy_embeddings, noisy_embeddings))

    rot_cov = total_rot_embeddings.T @ total_rot_embeddings / (total_rot_embeddings.shape[0]-1)
    noisy_cov = total_noisy_embeddings.T @ total_noisy_embeddings / (total_noisy_embeddings.shape[0]-1)

    rot_eigs, _ = torch.symeig(rot_cov)
    noisy_eigs, _ = torch.symeig(noisy_cov)
    
    return rot_eigs, noisy_eigs