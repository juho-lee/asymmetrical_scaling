import scipy.stats as ss
import math

from torch.distributions import Pareto, Uniform, Gamma, HalfCauchy, Beta

import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn as nn

from torch.special import gammainc, gammaln
from torch.autograd import Function

class GammaInc(Function):

    @staticmethod
    def forward(ctx, a, x):
        ctx.save_for_backward(a, x)
        output = gammainc(a, x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, x = ctx.saved_tensors
        grad_a = grad_x = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_output * ((gammainc(a+1e-5, x) - gammainc(a, x))/1e-5)
        if ctx.needs_input_grad[1]:
            grad_x = grad_output * (x**(a-1)*torch.exp(-x) / torch.exp(gammaln(a)))

        return grad_a, grad_x

mygammainc = GammaInc.apply

# Sample from an iid finite dimensional approximation
# of the Generalized BFRY process (double power-law)
def sample_finite_GBFRY(alpha, tau, mu=1., shape=(1, 5)):
    # shape[0]: Number of samples
    # shape[1]: Number of atoms to use for the finite
    #           dimensional approximation
    tau = np.maximum(tau, 1.01)

    if np.isscalar(shape):
        out_features = 1
        in_features = shape
    else:
        out_features = shape[0]
        in_features = shape[1]

    c = mu * (tau - 1) / (tau - alpha)

    eta = 1 #/ math.gamma(1 - alpha)

    s_mat = Uniform(torch.zeros(shape),
                    torch.ones(shape)
                    ).sample()

    log_tl = np.log(alpha * in_features * tau / eta / (tau - alpha)) / alpha

    tens = torch.ones((2, out_features, in_features))
    tens[0, :, :] = torch.log(s_mat)
    tens[1, :, :] = torch.log(1 - s_mat) + alpha * logsumexp((log_tl, 0))
    log_w = -1 / alpha * torch.logsumexp(tens, axis=0)
    gamma_mat = Gamma(concentration=(1-alpha)*torch.ones(shape), rate=torch.ones(shape)).sample()

    pareto_mat = Pareto(scale=torch.ones(shape), alpha=tau * torch.ones(shape)).sample()

    return (c * torch.exp(log_w) * pareto_mat * gamma_mat).detach().numpy()


# Sample from an iid finite dimensional approximation
# of the Generalized BFRY process (double power-law)
def sample_finite_GGP(alpha, beta=1, mu=1., shape=(1, 5)):
    # shape[0]: Number of samples
    # shape[1]: Number of atoms to use for the finite
    #           dimensional approximation

    if np.isscalar(shape):
        out_features = 1
        in_features = shape
    else:
        out_features = shape[0]
        in_features = shape[1]

    eta = mu * beta**(1-alpha) #/ math.gamma(1 - alpha)

    s_mat = Uniform(torch.zeros(shape),
                    torch.ones(shape)
                    ).sample()

    log_tl = np.log(alpha * in_features / eta) / alpha

    tens = torch.ones((2, out_features, in_features))
    tens[0, :, :] = torch.log(s_mat)+alpha*np.log(beta)
    tens[1, :, :] = torch.log(1 - s_mat) + alpha * logsumexp((log_tl, np.log(beta)))
    log_w = -1 / alpha * torch.logsumexp(tens, axis=0)

    gamma_mat = Gamma((1-alpha)*torch.ones(shape), torch.ones(shape)).sample()

    return (torch.exp(log_w) * gamma_mat).detach().numpy()

def sample_finite_Stable(alpha, mu=1, shape=(1, 5)):
    # shape[0]: Number of samples
    # shape[1]: Number of atoms to use for the finite
    #           dimensional approximation

    if np.isscalar(shape):
        out_features = 1
        in_features = shape
    else:
        out_features = shape[0]
        in_features = shape[1]
        
    pareto_mat = Pareto(alpha*torch.ones(shape), torch.ones(shape)).sample()
    return mu*pareto_mat/(in_features)**(1/alpha)

class IIDInit:
    def __init__(self, p):
        self.is_static = True
        self.p = p

    def rvs(self, size):
        return 1 / self.p * torch.ones(size)

    def transform(self, x):
        return x

    def map_to_domain(self, x):
        return x

class InvGammaInit:
    def __init__(self, alpha, beta):
        self.is_static = False
        self.alpha = alpha
        self.beta = beta

    def rvs(self, size):
        shape_tensor = torch.ones(size)
        dist = Gamma(self.alpha*shape_tensor, self.beta*shape_tensor)
        return 1/dist.sample()

    def log_pdf(self, x):
        dist = Gamma(self.alpha, self.beta)
        return -2*torch.log(x) + dist.log_prob(1/x).to(x.device)

    def transform(self, x):
        return torch.log(x)

    def map_to_domain(self, x):
        return torch.exp(x)

class HorseshoeInit:
    def __init__(self, p):
        self.is_static = False
        self.p = p

    def rvs(self, size):
        shape_tensor = torch.ones(size)
        dist = HalfCauchy(shape_tensor)
        return (np.pi / 2 * dist.sample() / self.p)**2

    def log_pdf(self, x):
        dist = HalfCauchy(1)
        C = 2 / np.pi * torch.sqrt(x) * self.p
        return -torch.log(np.pi**2 / 2 / self.p**2 * C) + dist.log_prob(C).to(x.device)

    def transform(self, x):
        return torch.log(x)

    def map_to_domain(self, x):
        return torch.exp(x)
    
class BetaInit:
    def __init__(self, alpha=1, beta=1./2):
        self.is_static = False
        self.alpha = alpha
        self.beta = beta

    def rvs(self, size):
        shape_tensor = torch.ones(size)
        dist = Beta(self.alpha * shape_tensor, self.beta * shape_tensor)
        return dist.sample()

    def log_pdf(self, x):
        dist = Beta(self.alpha, self.beta)
        return dist.log_prob(x).to(x.device)

    def transform(self, x):
        return torch.logit(x, 1e-8)

    def map_to_domain(self, x):
        return torch.sigmoid(x)

class GBFRYInit:
    def __init__(self, alpha=0.5, tau=2, mu=1):
        self.is_static = False
        self.alpha = alpha
        self.tau = tau
        self.mu = mu

    def rvs(self, size):
        return torch.tensor(sample_finite_GBFRY(alpha=self.alpha, tau=self.tau, mu=self.mu, shape=size))
        #return sample_finite_GBFRY(alpha=self.alpha, tau=self.tau, mu=self.mu, shape=size)

    def log_pdf_(self, x):
        c = self.mu * (self.tau - 1) / (self.tau - self.alpha)
        return (-(1+self.tau)*x.log()+torch.log(torch.special.gammainc(
            torch.tensor(self.tau-self.alpha).to(x.device), x/c)))
    
    def log_pdf(self, x):
        c = self.mu * (self.tau - 1) / (self.tau - self.alpha)
        shape = x.shape
        
        x_ = x / c
        x_ = x_.clip(min=1e-25)
        if np.isscalar(shape):
            out_features = 1
            in_features = shape
        else:
            out_features = shape[0]
            in_features = shape[1]
            
        tl = (self.alpha * in_features * self.tau / (self.tau-self.alpha))**(1 / self.alpha)
        g_in = torch.tensor(self.tau-self.alpha).to(x.device)
        
        return (-(1+self.tau)*x.log()+torch.log(
            torch.special.gammainc(g_in, x_) -  torch.special.gammainc(g_in, x_*(tl+1))/(1+tl)**(self.tau-self.alpha)))

    def transform(self, x):
        return torch.log(x)

    def map_to_domain(self, x):
        return torch.exp(x)
    
    
class GBFRYInitLearnableAlpha(nn.Module):
    def __init__(self, tau=2, mu=1):
        super().__init__()
        self.is_static = False
        self.alpha_logit = nn.Parameter(torch.randn(1))
        #self.register_buffer('alpha_logit', torch.randn(1))
        self.tau = tau
        self.mu = mu

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit).clip(min=1e-2, max=1-1e-2)

    def rvs(self, size):
        return sample_finite_GBFRY(alpha=self.alpha, tau=self.tau, mu=self.mu, shape=size)

    def log_pdf(self, x):
        tau, alpha = self.tau, self.alpha
        c = self.mu * (tau - 1) / (tau - alpha)
        shape = x.shape

        x_ = x / c
        x_ = x_.clip(min=1e-25)
        if np.isscalar(shape):
            out_features = 1
            in_features = shape
        else:
            out_features = shape[0]
            in_features = shape[1]

        t = (alpha*in_features*tau / (tau-alpha))**(1/alpha)
        a = (-(1+tau)*x_.log() + torch.log(tau*alpha)
                - torch.special.gammaln(1-alpha)
                - torch.log((t+1)**alpha - 1))
        b = (torch.log(mygammainc(tau-alpha, x_)
                - mygammainc(tau-alpha, (t+1)*x_)/(t+1)**(tau-alpha))
                + torch.special.gammaln(tau-alpha))
        y = a + b
        return y

    def transform(self, x):
        return torch.log(x)

    def map_to_domain(self, x):
        return torch.exp(x)
    

class GGPInit:
    def __init__(self, alpha=0.5, beta=1, mu=1):
        self.is_static = False
        self.alpha = alpha
        self.beta = beta
        self.mu = mu

    def rvs(self, size):
        return torch.tensor(sample_finite_GGP(alpha=self.alpha, beta=self.beta, mu=self.mu, shape=size))
        #return sample_finite_GBFRY(alpha=self.alpha, tau=self.tau, mu=self.mu, shape=size)

    def log_pdf(self, x):
        alpha = self.alpha
        beta = self.beta
        eta = self.mu * beta**(1-alpha)
        
        shape = x.shape
        
        x = x.clip(min=1e-12)
        
        if np.isscalar(shape):
            out_features = 1
            in_features = shape
        else:
            out_features = shape[0]
            in_features = shape[1]
            
        tl = (alpha * in_features / eta)**(1 / alpha)
        
        res = torch.where(
            x > 1e-16,
            -(1+self.alpha)*x.log()-beta*x+torch.log(1-(-tl*x).exp()),
            -(1+self.alpha)*x.log()-beta*x+torch.log(tl*x)
        )
        if torch.isnan(torch.sum(res)):
            print(x)
            print(torch.sum((1+self.alpha)*x.log()))
            print(torch.sum(beta*x))
            print(torch.sum(torch.log(1-(-tl*x).exp())))
            assert False, "break"
            
        return res

    def transform(self, x):
        return torch.log(x)

    def map_to_domain(self, x):
        return torch.exp(x)
    
class StableInit:
    def __init__(self, alpha=0.5, mu=1):
        self.is_static = False
        self.alpha = alpha
        self.mu = mu 
        
    def rvs(self, size):
        return torch.tensor(sample_finite_Stable(alpha=self.alpha, mu=self.mu, shape=size))  
    
    def transform(self, x):
        return torch.log(x)

    def map_to_domain(self, x):
        return torch.exp(x)    
    
    
def lam_dist(p, name, tau=2, c=1, eta=1):
    # this function returns the a sampler for the variances lambda
    if name == 'iid':
        return IIDInit(p)

    if name == "invgamma":
        return InvGammaInit(alpha=2, beta=1/p)

    elif name == 'horseshoe':
        return HorseshoeInit(p)

    elif name == 'beta':
        return BetaInit(alpha=eta / (2*p), beta=0.5)

    elif name == 'reghorseshoe':
        def reghorseshoe_rvs(size):
            S = (ss.cauchy.rvs(size=size) / p) ** 2
            return S / (1 + c * S)

        return reghorseshoe_rvs
    elif name == 'betapareto':
        def betapareto_rvs(size):
            return ss.beta(0.5 / p, 0.5).rvs(size=size) / ss.beta(tau, 1).rvs(size=size)

        return betapareto_rvs
    elif name == 'bernoulli':
        return lambda size: ss.bernoulli(1 / p).rvs(size)
    elif name == 'gbfry':
        return lambda size: sample_finite_GBFRY(alpha=0.5, tau=2, mu=1, shape=size).flatten()
    elif name == 'gbfry_heavy':
        return lambda size: sample_finite_GBFRY(alpha=0.5, tau=1, mu=1, shape=size).flatten()

def lam_sampler(p, name, a=1.0, alpha = 0.5, tau=2, c=1, eta=1):
    # this function returns the a sampler for the variances lambda
    if name == 'iid':
        def iid_rvs(size):
            return 1 / p * np.ones(size)
        return iid_rvs
    
    if name == 'zipfian': # gets the name from the zipfian distribution in scipy    
        def zipfian_rvs(size):
            return ss.zipfian.pmf(np.arange(p), 1/alpha, size)
        return zipfian_rvs
    
    if name == 'const_and_zipfian': # constant + zipfian weights (deterministic)
        if (a>1 or a<0):
            raise ValueError('a should be between 0 and 1')
        if (alpha>=1 or alpha<=0) :
            raise ValueError('0<alpha<1')
        def const_and_zipfian_rvs(size):
            return a / p * np.ones(size) + (1-a)*ss.zipfian.pmf(np.arange(p), 1/alpha, size)
        return const_and_zipfian_rvs
    
    if name == "invgamma":
        return lambda size: ss.invgamma(a=2, scale=1 / p).rvs(size=size)

    elif name == 'horseshoe':
        def horseshoe_rvs(size):
            return (np.pi / 2 * ss.cauchy.rvs(size=size) / p) ** 2

        return horseshoe_rvs
    elif name == 'beta':
        return lambda size: ss.beta(eta / (2*p), 0.5).rvs(size=size)
    elif name == 'reghorseshoe':
        def reghorseshoe_rvs(size):
            S = (ss.cauchy.rvs(size=size) / p) ** 2
            return S / (1 + c * S)

        return reghorseshoe_rvs
    elif name == 'betapareto':
        def betapareto_rvs(size):
            return ss.beta(0.5 / p, 0.5).rvs(size=size) / ss.beta(tau, 1).rvs(size=size)

        return betapareto_rvs
    elif name == 'bernoulli':
        return lambda size: ss.bernoulli(1 / p).rvs(size)
    elif name == 'gbfry':
        return lambda size: sample_finite_GBFRY(alpha=0.5, tau=2, mu=1, shape=size).flatten()
    elif name == 'gbfry_heavy':
        return lambda size: sample_finite_GBFRY(alpha=0.5, tau=1, mu=1, shape=size).flatten()
    elif name == 'gbfry_heavy_heavy':
        return lambda size: sample_finite_GBFRY(alpha=0.8, tau=1.5, mu=1, shape=size).flatten()
    elif name == 'gbfry_heavy_light':
        return lambda size: sample_finite_GBFRY(alpha=0.8, tau=5, mu=1, shape=size).flatten()
    elif name == 'gbfry_light_heavy':
        return lambda size: sample_finite_GBFRY(alpha=0.2, tau=1.5, mu=1, shape=size).flatten()
    elif name == 'gbfry_light_light':
        return lambda size: sample_finite_GBFRY(alpha=0.2, tau=5, mu=1, shape=size).flatten()




