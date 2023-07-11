import torch

from torch.distributions import Normal, kl_divergence
from torch.nn import functional


from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import torch

from numpyro.distributions import constraints as numpyro_constraints
from numpyro.distributions.util import promote_shapes, validate_sample
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)


def log_nb_positive(
    x: Union[torch.Tensor, jnp.ndarray],
    mu: Union[torch.Tensor, jnp.ndarray],
    theta: Union[torch.Tensor, jnp.ndarray],
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res

def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion.

    Parameters
    ----------
    mu
        mean of the NB distribution.
    theta
        inverse overdispersion.
    eps
        constant used for numerical log stability. (Default value = 1e-6)

    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    if not (mu is None) == (theta is None):
        raise ValueError(
            "If using the mu/theta NB parameterization, both parameters must be specified"
        )
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def _convert_counts_logits_to_mean_disp(total_count, logits):
    """NB parameterizations conversion.

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    logits
        success logits.

    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.

    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


def _gamma(theta, mu):
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = Gamma(concentration=concentration, rate=rate)
    return gamma_d


class Poisson(PoissonTorch):
    """Poisson distribution.

    Parameters
    ----------
    rate
        rate of the Poisson distribution.
    validate_args
        whether to validate input.
    scale
        Normalized mean expression of the distribution.
        This optional parameter is not used in any computations, but allows to store
        normalization expression levels.
    """

    def __init__(
        self,
        rate: torch.Tensor,
        validate_args: Optional[bool] = None,
        scale: Optional[torch.Tensor] = None,
    ):
        super().__init__(rate=rate, validate_args=validate_args)
        self.scale = scale


class NegativeBinomial(Distribution):
    r"""Negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "scale": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
        else:
            mu, theta = broadcast_all(mu, theta)
        self.mu = mu
        self.theta = theta
        self.scale = scale
        super().__init__(validate_args=validate_args)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean**2) / self.theta

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: Optional[Union[torch.Size, Tuple]] = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = PoissonTorch(
            l_train
        ).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
        return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                print(
                    "The value argument must be within the support of the distribution"
                    )

        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)

    def _gamma(self):
        return _gamma(self.theta, self.mu)


def negative_binomial_loss_mu_theta(y_pred, y_true):
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
    
    mu, theta =  torch.unbind(y_pred, dim=1)
    
    #if len(torch.isinf(nll).nonzero()) > 0:
    
    #torch.save(mu, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/mu.pt')
    #torch.save(theta, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/theta.pt')
    #torch.save(y_true, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/y_true.pt')
    
    NNL_NB_loss = (-NegativeBinomial(mu=mu, theta=theta)
                    .log_prob(y_true)
                    )
    
    #NNL_NB_loss = torch.nan_to_num(NNL_NB_loss,
    #                               nan=1e-7, 
    #                               posinf=1e15, 
    #                               neginf=-1e15)
    
    #torch.save(NNL_NB_loss, '/Users/paceramateos/projects/MOBER/output_MOBER_2/metrics/NNL_NB_loss.pt')
    
    return NNL_NB_loss


def loss_function_vae(dec, x, mu, stdev, kl_weight=1.0):
    # sum over genes, mean over samples, like trvae
    
    mean = torch.zeros_like(mu)
    scale = torch.ones_like(stdev)

    #KLD = kl_divergence(Normal(mu, stdev), Normal(mean, scale)).mean(dim=1)
    KLD = kl_divergence(Normal(mu, stdev), Normal(mean, scale)).sum(dim=1)
    
    #reconst_loss = functional.mse_loss(dec, x, reduction='none').mean(dim=1)
    
    NB_NLL = negative_binomial_loss_mu_theta(dec, x).mean(dim=1)
    #NB_NLL = negative_binomial_loss_mu_theta(dec, x).sum(dim=1)
    #print(torch.isnan(NB_NLL).nonzero())
    
    #print((NB_NLL + kl_weight * KLD).sum(dim=0))
    
    return (NB_NLL + kl_weight * KLD).sum(dim=0)
    #return (NB_NLL + kl_weight * KLD).mean(dim=0)








