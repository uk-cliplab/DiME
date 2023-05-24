import numpy as np
import torch
import math
from torch.utils.data.dataloader import DataLoader

import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
import repitl.divergences as div
import repitl.difference_of_entropies as dope

def matrixJRDivergence(x, y):
    alpha = 1.01
    sigma = np.sqrt(x.shape[1]/4)
    # sigma = 0.1
    return div.divergenceJR(x, y, sigma, alpha)

def diffEntropies(x,y):
    alpha = 1.01
    # sigma = np.sqrt(x.shape[1]/2)
    sigma = np.sqrt(x.shape[1]/4)
    Kx = ku.gaussianKernel(x,x, sigma)
    Ky = ku.gaussianKernel(y,y, sigma)
    DoE = dope.doe(Kx, Ky, alpha, n_iters=1)
    return DoE

def diffEntropiesLabels(x,label,num_classes=2):
    alpha = 1.01
    # sigma = np.sqrt(x.shape[1]/2)
    sigma = np.sqrt(x.shape[1]/4)
    Kx = ku.gaussianKernel(x,x, sigma)
    L = torch.nn.functional.one_hot(label,num_classes = num_classes).type(x.dtype)
    Kl = torch.matmul(L, L.t())
    DoE = dope.doe(Kx, Kl, alpha, n_iters=1)
    return DoE

def latentFactorDivergence(Vex,l,C,dim1,dim2):
    dim = C.shape[1] # assert dim = dim1 + dim2
    n = C.shape[0]
    V1ex  = Vex[l==0,:]
    V2ex  = Vex[l==1,:]
    oneHot = torch.nn.functional.one_hot(l).type(C.dtype)
    oneHot = torch.reshape(oneHot, (n, 2))
    L = torch.cat((oneHot[:,0:1].repeat(1,dim1),oneHot[:,1:].repeat(1,dim2)),axis = 1)
    CL = C*L   # Conditional distributions
    C1 = CL[l==0,:]
    C2 = CL[l==1,:]
    dV1 = matrixJRDivergence(V1ex, C1)
    dV2 = matrixJRDivergence(V2ex, C2)
    return dV1 + dV2

def latentFactorDivergenceModified(Vex,l,C,dim1,dim2):
    dim = C.shape[1] # assert dim = dim1 + dim2
    n = C.shape[0]
    V1ex  = Vex[l==0,:]
    V2ex  = Vex[l==1,:]
    oneHot = torch.nn.functional.one_hot(l).type(C.dtype)
    oneHot = torch.reshape(oneHot, (n, 2))
    L = torch.cat((oneHot[:,0:1].repeat(1,dim1),oneHot[:,1:].repeat(1,dim2)),axis = 1)
    CL = C*L   # Conditional distributions
    C1 = CL[l==0,:]
    C2 = CL[l==1,:]
    dV1 = matrixJRDivergence(V1ex[:,dim1:], C1[:,dim1:])
    dV2 = matrixJRDivergence(V2ex[:,:dim1], C2[:,:dim1])
    return dV1 + dV2

def conditionalEntropy(x,y):
    alpha = 1.01
    sigma = np.sqrt(x.shape[1]/4)
    Kx = ku.gaussianKernel(x,x, sigma)
    Ky = ku.gaussianKernel(y,y, sigma)
    condE = itl.matrixAlphaConditionalEntropy(Kx,Ky,alpha)
    return condE

def mutualInformation(x,y):
    alpha = 1.01
    sigma = np.sqrt(x.shape[1]/4)
    Kx = ku.gaussianKernel(x,x, sigma)
    Ky = ku.gaussianKernel(y,y, sigma)
    MI = itl.matrixAlphaMutualInformation(Kx,Ky,alpha)
    return MI