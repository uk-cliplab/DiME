import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent
import torch

"""
Here are implementations for a bunch of model loss functions.

DiME, CKA, and CCA are defined here.
"""

def DOE_Median_Embedding_Loss(rot_enc, noisy_enc, **kwargs):
    """
    DiME loss with bandwidth set as proportional to distance between points. sqrt(sum(dists) / (n^2 - n))
    """
    batch_size = rot_enc.shape[0] 
    n_iters = kwargs['n_iters'] if 'n_iters' in kwargs else 5

    dists = ku.squaredEuclideanDistance(rot_enc, rot_enc)
    sigmax = torch.sqrt(torch.sum(dists) / ( ((batch_size*2)**2 - (batch_size*2)) * 2 ))

    dists = ku.squaredEuclideanDistance(noisy_enc, noisy_enc)
    sigmay = torch.sqrt(torch.sum(dists) / ( ((batch_size*2)**2 - (batch_size*2)) * 2 ))

    Kx =  ku.gaussianKernel(rot_enc, rot_enc, sigmax)
    Ky = ku.gaussianKernel(noisy_enc, noisy_enc, sigmay)
    
    return dent.doe(Kx, Ky, alpha=1.01, n_iters=n_iters)

def DOE_Embedding_Loss(rot_enc, noisy_enc, **kwargs):
    """
    DiME loss with bandwidth as something the model learns for each latent space.
    """
    assert kwargs['sigma_x'] and kwargs['sigma_y']
    batch_size = rot_enc.shape[0] 
    
    Kx =  ku.gaussianKernel(rot_enc, rot_enc,  kwargs['sigma_x'])
    Ky = ku.gaussianKernel(noisy_enc, noisy_enc, kwargs['sigma_y'])
    
    return dent.doe(Kx, Ky, alpha=1.01, n_iters=5)

def DOE_Fixed_Embedding_Loss(rot_enc, noisy_enc, **kwargs):
    """
    DiME loss with bandwidth fixed at sqrt(D/2)
    """

    n_iters = kwargs['n_iters'] if 'n_iters' in kwargs else 5

    sigmax = torch.sqrt(torch.tensor(rot_enc.shape[1]/2))
    sigmay = torch.sqrt(torch.tensor(noisy_enc.shape[1]/2))

    Kx =  ku.gaussianKernel(rot_enc, rot_enc, sigmax)
    Ky = ku.gaussianKernel(noisy_enc, noisy_enc, sigmay)
    return dent.doe(Kx, Ky, alpha=1.01, n_iters=n_iters)

def CCA_Embedding_Loss(H1, H2, **kwargs):
    """
    Implementation of CCA loss function

    Sourced from an open-source repo which will be linked to in the final version
    """
    use_all_singular_values = True
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-3

    H1, H2 = H1.t(), H2.t()

    o1 = o2 = H1.size(0)
    
    m = H1.size(1) ** 0.5

    H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
    H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

    SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                H1bar.t()) + r1 * torch.eye(o1, device=H1.device)

    SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                H2bar.t()) + r2 * torch.eye(o2, device=H1.device)
    
    # Calculating the root inverse of covariance matrices by using eigen decomposition
    try:
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
    except Exception as e:
        # print(H1)
        # print(SigmaHat11)
        # print(H2)
        # print(SigmaHat22)
        raise

    # Added to increase stability
    posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    SigmaHat11RootInv = torch.matmul(
        torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
    SigmaHat22RootInv = torch.matmul(
        torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

    Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                     SigmaHat12), SigmaHat22RootInv)


    if use_all_singular_values:
        # all singular values are used to calculate the correlation
        tmp = torch.trace(torch.matmul(Tval.t(), Tval))
        corr = torch.sqrt(tmp)
    else:
        # just the top self.outdim_size singular values are used
        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(H1.device)) # regularization for more stability
        U, V = torch.symeig(trace_TT, eigenvectors=True)
        U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(H1.device))
        U = U.topk(10)[0]
        corr = torch.sum(torch.sqrt(U))
    
    return corr

def Conditional_Entropy_Embedding_Loss(X, Y, **kwargs):
    """
    Experimental conditional entropy loss function

    returns S(Y|X) = S(X,Y) - S(Y)

    used learned sigma for y kernel
    """
    sigmax = torch.sqrt(torch.tensor(X.shape[1]/2))
    sigmay = torch.sqrt(torch.tensor(Y.shape[1]/2))

    Kx = ku.gaussianKernel(X, X, sigmax)
    Ky = ku.gaussianKernel(Y, Y, sigmay) #kwargs['sigma_y'])

    return 0.001*(itl.matrixAlphaJointEntropy([Kx, Ky], 1.01) - itl.matrixAlphaEntropy(Ky, 1.01))

# calculate hilbert-smhmidt independence criterion
def HSIC(A, B):
    # construct centering matrix
    bs = A.shape[0]
    H = torch.eye(bs, device=A.device) - (1/bs)*torch.ones((bs, bs),device=A.device)

    t = torch.trace(A @ H @ B @ H)

    return t / (bs - 1)**2

# calculate centered-kernel alignment loss between view 1 and view 2
def CKA_Loss(rot_enc, noisy_enc, **kwargs):
    batch_size = rot_enc.shape[0]

    # find kernel sigmas
    dists = ku.squaredEuclideanDistance(rot_enc, rot_enc)
    sigmax = torch.sqrt(torch.sum(dists) / ( ((batch_size*2)**2 - (batch_size*2)) * 2 ))

    dists = ku.squaredEuclideanDistance(noisy_enc, noisy_enc)
    sigmay = torch.sqrt(torch.sum(dists) / ( ((batch_size*2)**2 - (batch_size*2)) * 2 ))

    # compute kernels
    rot_kernel = ku.gaussianKernel(rot_enc, rot_enc, sigmax)
    noisy_kernel = ku.gaussianKernel(noisy_enc, noisy_enc, sigmay)

    # compute hsic terms
    cross_hsic = HSIC(rot_kernel, noisy_kernel) # cross
    rot_hsic = HSIC(rot_kernel, rot_kernel)  # marginal
    noisy_hsic = HSIC(noisy_kernel, noisy_kernel) # marginal

    # compute cka loss
    result = cross_hsic / torch.sqrt(rot_hsic * noisy_hsic)