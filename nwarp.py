###################
# This script is modified based on: https://github.com/diozaka/diffseg
###################


import torch
import torch.nn as nn
import warnings
import math
import torch.nn.functional as F
from einops import rearrange
import sys
# from libcpab.cpab import Cpab

###############
### HELPERS ###
###############

def reset_parameters(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

class Constant(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes
        self.const = nn.Parameter(torch.Tensor(1, *output_sizes))

    # inp is an arbitrary tensor, whose values will be ignored;
    # output is self.const expanded over the first dimension of inp.
    # output.shape = (inp.shape[0], *output_sizes)
    def forward(self, inp):
        return self.const.expand(inp.shape[0], *((-1,)*len(self.output_sizes)))

    def reset_parameters(self):
        nn.init.uniform_(self.const, -1, 1) # U~[-1,1]

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
    def forward(self, inp):
        return inp.unsqueeze(self._dim)

class Square(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return inp*inp

class Abs(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.abs(inp)

class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.exp(inp)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.sin(inp)

#########################
### WARPING FUNCTIONS ###
#########################

class TSPStepWarp(nn.Module):
    def __init__(self, loc_net, width, power, min_step=0.0001, max_step=0.9999):
        # loc_net: nn.Module that takes shape (batch_size, seq_len, input_size)
        #          and produces shape (batch_size, n_seg-1) with logits that determine
        #          the modes of n_seg-1 TSP distributions.
        #          More precisely, modes for the TSP distributions are computed from
        #          the cumulative sum of the softmax over the logits. Logits thus
        #          encode the relative distances between consecutive modes.
        #          For numerical reasons, the modes are clamped to the
        #          interval [min_step, max_step].
        # width: width of the TSP distributions, from (0,1]
        # power: power of the TSP distributions, must be >=1 for unimodality

        super().__init__()
        if not isinstance(loc_net, nn.Module):
            raise ValueError("loc_net must be an instance of torch.nn.Module")

        self.loc_net = loc_net
        self.width = width
        self.power = power
        self.min_step = min_step
        self.max_step = max_step

    def _tsp_params(self, mode):
        # mode.shape = (n_modes,)
        # output.shape = ((n_modes, 1), (n_modes, 1), (n_modes, 1), double)
        a = torch.clamp(mode-self.width/2., 0., 1.-self.width).unsqueeze(1).to(mode.device) # max(0., min(1.-width, mode-width/2))
        b = torch.clamp(a+self.width, self.width, 1.).to(mode.device) # min(1., a+width)
        m = mode.unsqueeze(1) # [B*(n_seg-1)]
        n = self.power
        return a, b, m, n

    def _tsp_cdf(self, x, mode):
        # x.shape = (n_modes, seq_len)
        # mode.shape = (n_modes,)
        a, b, m, n = self._tsp_params(mode) # m: (n_modes,1)
        cdf = ((x <= m)*((m-a)/(b-a)*torch.pow(torch.clamp((x-a)/(m-a), 0., 1.).to(mode.device), n))
              +(m <  x)*(1.-(b-m)/(b-a)*torch.pow(torch.clamp((b-x)/(b-m), 0., 1.).to(mode.device), n)))
        return cdf # [B*(n_seg-1), seq_len]

    def forward(self, input_seq,mask):
        # input_seq.shape = (batch_size, seq_len, input_size)
        # output shape = (batch_size, seq_len)
        batch_size, seq_len, input_size = input_seq.shape

        # compute modes for all triangular mixture distributions from loc_net
        #modes = nn.functional.softmax(self.loc_net(input_seq), dim=1).cumsum(dim=1)[:,:-1] # last is always 1
        mu = self.loc_net(input_seq,mask)
        modes = nn.functional.softmax(torch.cat([torch.zeros((batch_size,1)).to(input_seq.device), # fix first logit to 0
                                                 mu], dim=1),
                                      dim=1).cumsum(dim=1)[:,:-1] # last boundary is always 1
        modes = torch.clamp(modes, self.min_step, self.max_step).to(input_seq.device) # [B,n_seg-1]
        _, n_steps = modes.shape # == n_seg-1

        xrange = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(n_steps*batch_size,-1).to(modes.device) # [B*(n_seg-1), seq_len]
        cdf = self._tsp_cdf(xrange, modes.flatten())

        # compute mixture cdf
        gamma = cdf.reshape(-1,n_steps,seq_len).sum(dim=1)/n_steps #[b,seq_len]
        return gamma, n_steps+1, modes, mu


# backend can be any nn.Module that takes shape (batch_size, seq_len, input_size)
# and produces shape (batch_size, seq_len); the output of the backend is normalized
# and integrated.
class VanillaWarp(nn.Module):
    def __init__(self, backend, nonneg_trans='abs'):
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise ValueError("backend must be an instance of torch.nn.Module")
        self.backend = backend
        self.normintegral = NormalizedIntegral(nonneg_trans)

    # input_seq.shape = (batch_size, seq_len, input_size)
    # output shape = (batch_size, seq_len)
    def forward(self, input_seq):
        gamma = self.normintegral(self.backend(input_seq)) 
        return gamma

class NormalizedIntegral(nn.Module):
    # {abs, square, relu}      -> warping variance more robust to input variance
    # {exp, softplus, sigmoid} -> warping variance increases with input variance, strongest for exp
    def __init__(self, nonneg):
        super().__init__()
        # higher warping variance
        if nonneg == 'square':
            self.nonnegativity = Square()
        elif nonneg == 'relu':
            warnings.warn('ReLU non-negativity does not necessarily result in a strictly monotonic warping function gamma! In the worst case, gamma == 0 everywhere.', RuntimeWarning)
            self.nonnegativity = nn.ReLU()
        elif nonneg == 'exp':
            self.nonnegativity = Exp()
        # lower warping variance
        elif nonneg == 'abs':
            self.nonnegativity = Abs()
        elif nonneg == 'sigmoid':
            self.nonnegativity = nn.Sigmoid()
        elif nonneg == 'softplus':
            self.nonnegativity = nn.Softplus()
        else:
            raise ValueError("unknown non-negativity transformation, try: abs, square, exp, relu, softplus, sigmoid")

    # input_seq.shape = (batch_size, seq_len)
    # output shape    = (batch_size, seq_len)
    def forward(self, input_seq):
        # transform sequences to alignment functions between 0 and 1
        dgamma = torch.cat([torch.zeros((1,1)), # fix entry to 0
                            self.nonnegativity(input_seq)], dim=1)
        gamma = torch.cumsum(dgamma, dim=1)
        gamma /= torch.max(gamma, dim=1)[0].unsqueeze(1)
        return gamma

##########################
### SEGMENTATION LAYER ###
##########################

class Resample(nn.Module):
    def __init__(self):
        super().__init__()

    # gamma.shape    = (batch_size, warped_len), alignment functions mapping to [0,1]
    # original.shape = (batch_size, original_len, original_dim)
    # output.shape   = (batch_size, warped_len, original_dim)
    # almat.shape    = (batch_size, warped_len, original_len)
    def forward(self, gamma, original, kernel='linear', return_alignments=False):
        # batch_size, warped_len = gamma.shape
        batch_size, warped_len = gamma.shape
        _, original_len, original_dim = original.shape
 
        gamma_scaled = gamma*(original_len-1)
        output = torch.zeros(batch_size, warped_len, original_dim)
        if return_alignments:
            almat = torch.zeros(batch_size, warped_len, original_len)
        else:
            almat = None

        if kernel == 'integer':
            for k in range(original_len):
                responsibility = (torch.floor(gamma_scaled+0.5)-k == 0).float()
                output += responsibility.unsqueeze(2).expand(-1,-1,original_dim)*original[:,k,:].unsqueeze(1)
                if return_alignments:
                    almat[:,:,k] = responsibility
        elif kernel == 'linear':
            for k in range(original_len):
                responsibility = torch.threshold(1-torch.abs(gamma_scaled-k), 0., 0.)
                output += responsibility.unsqueeze(2).expand(-1,-1,original_dim)*original[:,k,:].unsqueeze(1)
                if return_alignments:
                    almat[:,:,k] = responsibility
        else:
            raise ValueError("unknown interpolation kernel, try 'integer' or 'linear'")
        return output, almat

class Cal_M(nn.Module):
    def __init__(self, dim, output_len):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.Tanh(),
            nn.Linear(dim*2,output_len, bias=False))

        self.reduce_d = nn.Linear(dim, 1)
    
    def forward(self, h, mask=None, mask_value=-1e30):
        # input: (batch_size, seq_len, input_size)
        # output: (batch_size, n_seg-1)

        mask = rearrange(mask, 'b l -> b l 1')
        attn = self.attn(h) # [B,L,L']
        if mask is not None:
            attn = mask * attn + (1-mask)*mask_value
        
        attn = F.softmax(attn, dim=-2)
        h = torch.matmul(h.transpose(-1, -2), attn).squeeze(-1) # [B,D L']
        h = self.reduce_d(rearrange(h, 'b d l -> b l d')).squeeze(-1) # [b l']
        # h = torch.tanh(h)
        return h
        

class Almtx(nn.Module):
    def __init__(self, opt, dim, K, width=0.125, power=16., min_step=0.0001, max_step=0.9999):
        super().__init__()
        
        m_enc = Cal_M(dim, K-1).to(opt.device)
        self.warp = TSPStepWarp(m_enc, width=width, power=power, min_step=min_step, max_step=max_step).to(opt.device)

    # gamma.shape    = (batch_size, warped_len), alignment functions mapping to [0,1]
    # original.shape = (batch_size, original_len, original_dim)
    # output.shape   = (batch_size, warped_len, original_dim)
    # almat.shape    = (batch_size, warped_len, original_len)
    def forward(self, input_seq, mask, kernel='linear', test_bound=False):
        
        gamma, original_len, modes, mu = self.warp(input_seq,mask)
        batch_size, warped_len = gamma.shape
        # _, original_len, original_dim = input_seq.shape
        gamma_scaled = gamma*(original_len-1)
        almat = torch.zeros(batch_size, warped_len, original_len)

        if kernel == 'integer':
            for k in range(original_len):
                responsibility = (torch.floor(gamma_scaled+0.5)-k == 0).float()
                almat[:,:,k] = responsibility
        elif kernel == 'linear':
            for k in range(original_len):
                responsibility = torch.threshold(1-torch.abs(gamma_scaled-k), 0., 0.)
                almat[:,:,k] = responsibility
        else:
            raise ValueError("unknown interpolation kernel, try 'integer' or 'linear'")
        
        if test_bound:
            return gamma_scaled, almat, modes, mu
        else:
            return gamma_scaled, almat


class ParameterWarp(nn.Module):
    # paramg_size = number of global parameters (same in every segment)
    # paraml_size = number of local parameters (differ in every segment)
    def __init__(self, n_seg, paramg_size, paraml_size, warp, verbose=True):
        super().__init__()
        self.warp = warp
        self.thetag = nn.Parameter(torch.Tensor(paramg_size)) #  [P]
        self.thetal = nn.Parameter(torch.Tensor(n_seg, paraml_size)) #[K,P]
        self.reset_parameters()
        self.verbose = verbose
        self.resample = Resample()

    def reset_parameters(self):
        fan_in = self.thetag.shape[0]+self.thetal.shape[1]
        nn.init.uniform_(self.thetal, -1./math.sqrt(fan_in), 1./math.sqrt(fan_in))
        nn.init.uniform_(self.thetag, -1./math.sqrt(fan_in), 1./math.sqrt(fan_in))

    # input_seq.shape = (batch_size, seq_len, input_size)
    # output shape    = (batch_size, seq_len, param_size)
    # resample_kernel = {'linear', 'integer'}
    def forward(self, input_seq, resample_kernel='linear'):
        batch_size, seq_len, _ = input_seq.shape

        # construct the full parameter vector (first global, then local components)
        theta = torch.cat((self.thetag.unsqueeze(0).expand(self.thetal.shape[0], -1), self.thetal), dim=1) #[k,paramg_size+paraml_size]

        # warp the full parameter vector to the length of input_seq with
        # the warping functions predicted by self.warp
        gamma = self.warp(input_seq)
        output, almat = self.resample(gamma, theta.unsqueeze(0).expand(batch_size,-1,-1),
                                      kernel=resample_kernel, return_alignments=True)
        if self.verbose:
            return output, almat, gamma
        else:
            self._almat = almat
            self._gamma = gamma
            return output

#################################
### SEGMENTATION-AWARE MODELS ###
#################################

class NormalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    # Returns the logpdfs of the sample sequence parametrized by the params sequence.
    #
    # sample.shape = (batch_size, seq_len, 1)
    # params.shape = (batch_size, seq_len, 2)
    # output shape = (batch_size, seq_len, 1)
    def forward(self, sample, params):
        return -0.5*(params[:,:,1].unsqueeze(2)
                     + torch.Tensor([math.log(2*math.pi)])
                     + (torch.square(sample - params[:,:,0].unsqueeze(2))
                        / torch.exp(params[:,:,1].unsqueeze(2))))


class GeneralizedLinearModel(nn.Module):
    # Output y_{itd} for instance i, time t and output dimension d is computed from
    # the covariate vector z_{it} and the parameter vector w_{itd} by a Generalized
    # Linear Model (GLM) of the form
    #
    #    \eta_{itd} = w_{itd}' z_{it} for all d,
    #        y_{it} = \sigma(\eta_{it})
    #
    # where \eta_{itd} is the linear predictor for the output dimension d,
    # and \sigma is the inverse link function that performs a non-linear
    # transformation of the linear predictors \eta_{it}.
    # In the simplest case, the (vector-valued) inverse link function decomposes
    # over the dimensions in the sense
    #
    #    \sigma(\eta_{it})_d := \hat{\sigma}(\eta_{itd})
    #
    # for a (real-valued) non-linear function \hat{\sigma}, but this is not
    # necessarily so.
    #
    # Note that the same covariates z_{it} of length `covariates_size` are used
    # for all `output_size` output dimensions, but with different coefficients.
    # Therefore, `output_size`, `covariates_size` and `param_size` must obey
    #
    #    param_size = covariates_size*output_size.
    #
    # invlink_func can be any callable, preferably nn.Module, that takes a Tensor
    # of shape (batch_size, seq_len, output_size) and outputs a Tensor of the same
    # shape. It should operate only across the last dimension and broadcast over
    # the first two dimensions for consistency.
    def __init__(self, output_size=1, invlink_func=None):
        super().__init__()
        self.output_size = output_size

        if invlink_func is None:
            self.invlink_func = nn.Identity()
        else:
            self.invlink_func = invlink_func

    # covariates.shape = (batch_size, seq_len, covariates_size)
    # params.shape     = (batch_size, seq_len, param_size)
    # output shape     = (batch_size, seq_len, output_size)
    def forward(self, covariates, params):
        batch_size, seq_len, covariates_size = covariates.shape
        param_size = params.shape[2]
        if param_size != covariates_size*self.output_size:
            raise ValueError("we need one parameter per covariate and output dimension")
        linear_predictors = (params.reshape(batch_size, seq_len,
                                            covariates_size,
                                            param_size//covariates_size)*covariates.unsqueeze(-1)).sum(-2)
        return self.invlink_func(linear_predictors)

