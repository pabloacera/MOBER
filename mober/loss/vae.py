import torch

from torch.distributions import Normal, kl_divergence
from torch.nn import functional

import torch

def negative_binomial_loss(y_pred, y_true):
    """
    Negative binomial loss function.
    Assumes PyTorch backend.
    
    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth values of predicted variable.
    y_pred : torch.Tensor
        n and p values of predicted distribution.
        
    Returns
    -------
    nll : torch.Tensor
         Negative log likelihood.
    """
    # Separate the parameters
    n, p = torch.unbind(y_pred, dim=-1)

    # Add one dimension to make the right shape
    n = n.unsqueeze(-1)
    p = p.unsqueeze(-1)
    
    # Calculate the negative log likelihood
    nll = (
        torch.lgamma(n) 
        + torch.lgamma(y_true + 1)
        - torch.lgamma(n + y_true)
        - n * torch.log(p)
        - y_true * torch.log(1 - p)
    )

    return nll


def loss_function_vae(dec, x, mu, stdev, kl_weight=1.0):
    # sum over genes, mean over samples, like trvae
    
    mean = torch.zeros_like(mu)
    scale = torch.ones_like(stdev)

    KLD = kl_divergence(Normal(mu, stdev), Normal(mean, scale)).mean(dim=1)
    
    #reconst_loss = functional.mse_loss(dec, x, reduction='none').mean(dim=1)
    
    NB_NLL = negative_binomial_loss(dec, x).mean(dim=1)
    
    return (NB_NLL + kl_weight * KLD).sum(dim=0)








