from .. import _magisolver_base as _base

import numpy as np
import torch
from tqdm.notebook import trange

'''
Polymorphic expressions.
'''

def to_tensor(arr, dtype, device="cuda", requires_grad=False):
    if type(dtype) is str:
        dtype = eval("torch." + dtype)
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    return torch.tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)

def to_arr(tensor):
    return tensor.detach().cpu().numpy()

def is_tensor(arg):
    return torch.is_tensor(arg)
    
def vmap(func):
    return torch.func.vmap(func, in_dims=0)

def replicate(tensor, dims, dtype, device):
    '''
    Replicate existing matrices, convert to tensors.
    '''
    return torch.tensor(tensor, dtype=dtype, device=device).repeat(*dims)

def normal(mean, sd, random_seed, dtype, device):
    if random_seed is not None:
        torch.random.manual_seed(random_seed)
    return torch.normal(mean=mean, std=sd)

def pad_tensor(tensor, axis):
    return torch.unsqueeze(tensor, axis=axis)

def concat(tensors, axis):
    return torch.concat(tensors, axis=axis)

def reshape(tensor, shape):
    return torch.reshape(tensor, shape)

def tensor_sum(tensor, axis):
    return torch.sum(tensor, axis=axis)

def tensor_mean(tensor, axis):
    return torch.mean(tensor, axis=axis)

def tensor_exp(tensor):
    return torch.exp(tensor)

def tensor_log(tensor):
    return torch.log(tensor)

def batch_diag(tensor):
    return torch.diagonal(tensor, dim1=1, dim2=2)

def embed_diagonal(tensor):
    return torch.diag_embed(tensor)

def permute(tensor, permutation):
    return torch.permute(tensor, permutation)

def square_distances(tensor):
    return torch.cdist(tensor, tensor)**2

def tensor_median(tensor):
    return torch.median(tensor)

def tile(tensor, shape):
    return torch.tile(tensor, shape)

def prepare_particles(tensor):
    return torch.clone(tensor)

def tensor_abs(tensor):
    return torch.abs(tensor)

def tensor_allsmall(tensor, ref, atol, rtol):
    return torch.all(torch.abs(tensor) <= atol + rtol * torch.abs(ref))

def tensor_max(tensor):
    return torch.max(tensor).item()

def clone(tensor):
    return torch.clone(tensor)

def gradient_step(optimizer, gradient, tensor):
    tensor.grad = gradient
    optimizer.step()

def Adam(params, lr=0.001):
    return torch.optim.Adam(params, lr=lr)

def autograd(loss_fn, args, opt):
    opt.zero_grad()
    loss = loss_fn(*args)
    loss.backward()
    opt.step()
    return loss.item()

def stack(tensors, axis):
    return torch.stack(tensors, axis=axis)

def update_tensor(tensor, col_indices, updates):
    tensor[:,col_indices] = updates
    return tensor

def slice_2(tensor, indices):
    return tensor[:,indices]

def clip(tensor):
    return torch.clamp(tensor, min=0)

### pSVGD modifications

def batch_mahalanobis(tensor, diag=None):
    '''
    Compute row-wise batched Mahalanobis distance for tensor, assuming a diagonal PSD matrix.
    '''
    if diag is None:
        diag = torch.eye(tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
    # else:
    #     diag = diag / diag.mean()
                         
    tensor_adj = tensor @ diag.abs()**0.5
    diffs = tensor_adj.tile(tensor.shape[0], 1, 1) - tensor_adj.unsqueeze(1)
    return torch.sum(diffs**2, axis=2)

from sklearn.gaussian_process import kernels as skl_kernels
def compute_cov_prior(p, sig_uk, I, phis, v=2.01, dtype=torch.float32, device='cuda'):
    '''
    Compute the prior covariance matrix of all parameters.
    
    p (int) : dimension of ODE parameters
    sig_uk (int) : dimension of unknown sampling standard devations
    I (n x 1 numpy array) : discretization set
    phis (D x 2 numpy array) : prior Matern kernel parameters for each X component
    v (float) : Matern kernel degrees of freedom
    '''
    return torch.block_diag(torch.eye(p, dtype=dtype, device=device),
        *[torch.tensor((phi1 * skl_kernels.Matern(length_scale=phi2, nu=v))(I, I), dtype=dtype, device=device) for phi1, phi2 in phis],
                            torch.eye(0 if sig_uk is None else sig_uk, dtype=dtype, device=device))

def inverse(tensor):
    return torch.linalg.pinv(tensor)

def compute_subspace(gradient, cov_prior_inv=None, alpha=0.01):
    '''
    Compute the projection subspace.
    '''
    if cov_prior_inv is None:
        cov_prior_inv = torch.eye(gradient.shape[0], dtype=gradient.dtype, device=tensor.device)
                                  
    # estimate gradient information matrix
    H_hat = gradient.T @ gradient / gradient.shape[0]
    
    # convert generalized EV problem to normal EV problem
    # estimate number of dominant eigenpairs
    evals, evecs = torch.linalg.eigh(cov_prior_inv @ H_hat)
    sq_evals, magnitude_id = (evals**2).sort()
    cum_evals = sq_evals.cumsum(0) / sq_evals.sum()
    sig_evs = min(gradient.shape[1], (cum_evals >= alpha).sum() + 1) # add one for stability

    # save dominant eigenpairs
    lam = evals[magnitude_id].flip(0)[:sig_evs]
    psi = evecs[:,magnitude_id].flip(1)[:,:sig_evs]

    return lam, psi, torch.diag(lam + 1), sig_evs
    
    
#########################################################################################################

class MAGISolver(_base.baseMAGISolver):
    def __init__(self, ode, dfdx, dfdtheta, data, theta_guess, theta_conf=0,
                 sigmas=None, X_guess=1, mu=None, mu_dot=None, pos_X=False, pos_theta=False,
                 prior_temperature=None, bayesian_sigma=True):
        
        super()._configure_polymorphism(
            to_tensor=to_tensor,
            to_arr=to_arr,
            is_tensor=is_tensor,
            vmap=vmap,
            replicate=replicate,
            normal=normal,
            pad_tensor=pad_tensor,
            concat=concat,
            reshape=reshape,
            tensor_sum=tensor_sum,
            tensor_mean=tensor_mean,
            tensor_exp=tensor_exp,
            tensor_log=tensor_log,
            batch_diag=batch_diag,
            embed_diagonal=embed_diagonal,
            permute=permute,
            square_distances=square_distances,
            tensor_median=tensor_median,
            tile=tile,
            prepare_particles=prepare_particles,
            tensor_abs=tensor_abs,
            tensor_allsmall=tensor_allsmall,
            tensor_max=tensor_max,
            clone=clone,
            gradient_step=gradient_step,
            Adam=Adam,
            autograd=autograd,
            stack=stack,
            update_tensor=update_tensor,
            slice_2=slice_2,
            clip=clip,

            ### pSVGD modifications
            batch_mahalanobis=batch_mahalanobis,
            compute_cov_prior=compute_cov_prior,
            inverse=inverse,
            compute_subspace=compute_subspace
        )
        
        super().__init__(ode=ode, dfdx=dfdx, dfdtheta=dfdtheta, data=data, theta_guess=theta_guess,
                         theta_conf=theta_conf, sigmas=sigmas, X_guess=X_guess, mu=mu, mu_dot=mu_dot, pos_X=pos_X, pos_theta=pos_theta,
                         prior_temperature=prior_temperature, bayesian_sigma=bayesian_sigma)

    ### pSVGD modifications
    def solve(self, optimizer, optimizer_kwargs=dict(), max_iter=10, subspace_updates=1000, alpha=0.01,
              atol=1e-2, rtol=1e-8, bandwidth=-1, monitor_convergence=False):
        optimizer_kwargs['params'] = True
        return super().solve(optimizer, optimizer_kwargs, max_iter, subspace_updates, alpha,
                             atol, rtol, bandwidth, monitor_convergence)
