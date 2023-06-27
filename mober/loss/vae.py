import torch

from torch.distributions import Normal, kl_divergence, negative_binomial
from torch.nn import functional


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
    
    n, p =  torch.unbind(y_pred, dim=1)
    
    # Add one dimension to make the right shape
    #n = n.unsqueeze(-1)
    #p = p.unsqueeze(-1)
    
    # Calculate the negative log likelihood
    #print(torch.isnan(n).any())
    
    epsilon = 1e-8  # small constant
    
    nll = (torch.lgamma(n + epsilon) 
           + torch.lgamma(y_true + 1)
           - torch.lgamma(n + y_true + epsilon)
           - n * torch.log(p + epsilon)
           - y_true * torch.log(1 - p + epsilon)
          )
    '''
    if len(torch.isinf(nll).nonzero()) > 0:
        torch.save(n, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/n.pt')
        torch.save(p, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/p.pt')
        torch.save(nll, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/NLL.pt')
        torch.save(y_true, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/y_true.pt')
    '''
    return nll

def negative_binomial_loss_mu_alpha(y_pred, y_true):
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
    NNL_NB_loss : torch.Tensor
         Negative log likelihood.
    """
    # Separate the parameters
    
    mu, alpha =  torch.unbind(y_pred, dim=1)
    
    #if len(torch.isinf(nll).nonzero()) > 0:
    torch.save(mu, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/mu.pt')
    torch.save(alpha, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/alpha.pt')
    torch.save(y_true, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/y_true.pt')
    
    
    NNL_NB_loss = -negative_binomial.NegativeBinomial(total_count=1/alpha,
                                                      logits=alpha*mu).log_prob(y_true)
    
    NNL_NB_loss = torch.nan_to_num(NNL_NB_loss, nan=1e-7, posinf=1e15, neginf=-1e15)
    
    torch.save(NNL_NB_loss, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/NNL_NB_loss.pt')

    return NNL_NB_loss


def loss_function_vae(dec, x, mu, stdev, kl_weight=1.0):
    # sum over genes, mean over samples, like trvae
    
    mean = torch.zeros_like(mu)
    scale = torch.ones_like(stdev)

    KLD = kl_divergence(Normal(mu, stdev), Normal(mean, scale)).mean(dim=1)
    
    #reconst_loss = functional.mse_loss(dec, x, reduction='none').mean(dim=1)
    
    NB_NLL = negative_binomial_loss_mu_alpha(dec, x).mean(dim=1)
    
    #print(torch.isnan(NB_NLL).nonzero())
    
    print(NB_NLL.sum(dim=0),  KLD.sum(dim=0))
    
    return (NB_NLL + kl_weight * KLD).sum(dim=0)








