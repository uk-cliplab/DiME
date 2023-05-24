import torch as torch
import numpy as np

import repitl.kernel_utils as ku
import repitl.matrix_itl as itl

def divergenceJR(X,Y,sigma,alpha, weighted = False, n_rff = None):
    if weighted:
        divergence = divergenceJR_weighted(X,Y,sigma,alpha)
    else:
        divergence = divergenceJR_unweighted(X,Y,sigma,alpha)  
    return divergence

def divergenceJR_weighted(X,Y,sigma,alpha):
    # Getting number of samples from each class
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    XY = torch.cat((X,Y))
    # Kernel of the mixture
    K = ku.gaussianKernel(XY,XY, sigma)
    # Kernel of the labels    
    Kl = torch.matmul(L, L.t())

    # Weight matrices
    D1 = np.sqrt((N+M)/(2*N))*torch.ones(N,1,dtype=X.dtype) # (n1+n2)
    D2 = np.sqrt((N+M)/(2*M))*torch.ones(M,1,dtype=X.dtype)
    D = torch.cat((D1,D2))
    # Weighted kernels
    K_weighted = D*K*torch.t(D)
    Kl_weighted  = D*Kl*torch.t(D)
    
    # Computing divergence
    Hxy = itl.matrixAlphaEntropy(K_weighted, alpha=alpha)
    Hj = itl.matrixAlphaJointEntropy([K_weighted, Kl], alpha=alpha)
    Hl = approx.matrixAlphaEntropyLabel(L, alpha, weighted = True)
    divergence  = Hxy + Hl - Hj
    return divergence

def divergenceJR_unweighted(X,Y,sigma,alpha):
    # Getting number of samples from each class
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    XY = torch.cat((X,Y))
    # Kernel of the mixture
    K = ku.gaussianKernel(XY,XY, sigma)
    # Kernel of the labels  
    Kl = torch.matmul(L, L.t())
    # Computing divergence
    Hxy = itl.matrixAlphaEntropy(K, alpha=alpha)
    Hj = itl.matrixAlphaJointEntropy([K, Kl], alpha=alpha)
    Hl = approx.matrixAlphaEntropyLabel(L, alpha)
    divergence  = Hxy + Hl - Hj
    return divergence
