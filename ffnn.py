import numpy as np
import scipy.stats as ss

# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# from torch.nn.functional import relu
from torch.nn import init
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class Scaling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.log_lambdas = nn.Parameter(
            torch.Tensor(in_features).fill_(np.log(1.0 / in_features))
        )

    def reset_parameters(self, lam_rvs=None, dtype=np.float32):
        device = self.log_lambdas.device
        if lam_rvs is None:
            nn.init.constant_(self.log_lambdas, np.log(1.0 / self.in_features)).to(
                device
            )
        else:
            lam = torch.from_numpy(lam_rvs(self.in_features).astype(dtype)).to(device)
            self.log_lambdas.data = lam.log()

    def forward(self, x):
        return x * (0.5 * self.log_lambdas).exp()[None]


class ScaledFCLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_output_layer=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.is_output_layer = is_output_layer
        self.activation = nn.ReLU() if not is_output_layer else nn.Identity()
        self.scale = Scaling(out_features) if not is_output_layer else nn.Identity()

    def reset_parameters(
        self, lam_rvs=None, dtype=np.float32, sample_U=False, sigma_v=1.0, sigma_b=1.0
    ):

        if self.is_output_layer:
            U = np.random.choice([-1, 1], size=self.linear.weight.shape)
            U = torch.from_numpy(U.astype(dtype)).to(self.linear.weight.device)
            self.linear.weight.data = U
        else:
            nn.init.normal_(self.linear.weight, mean=0.0, std=sigma_v)
            if self.linear.bias is not None:
                nn.init.normal_(self.linear.bias, mean=0.0, std=sigma_b)
            self.scale.reset_parameters(lam_rvs=lam_rvs, dtype=dtype)

    @property
    def log_lambdas(self):
        return self.scale.log_lambdas

    @property
    def v(self):
        return self.linear.weight

    def forward(self, x):
        return self.scale(self.activation(self.linear(x)))


class FFNN(nn.Module):
    def __init__(
        self,
        input_size,
        num_hidden_layers,
        hidden_size,
        output_size,
        bias=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.L = num_hidden_layers
        self.p = hidden_size
        self.output_size = output_size
        self.bias = bias

        self.input_layer = Scaling(input_size)
        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(
                ScaledFCLayer(
                    input_size if i == 0 else hidden_size, hidden_size, bias=bias
                )
            )

        self.output_layer = ScaledFCLayer(
            hidden_size, output_size, bias=bias, is_output_layer=True
        )

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def init_weights(self, lam_rvs=None, dtype=np.float32, sigma_v=1.0, sigma_b=1.0):
        self.input_layer.reset_parameters()
        for i, layer in enumerate(self.hidden_layers):
            layer.reset_parameters(
                lam_rvs=lam_rvs,
                dtype=dtype,
                sigma_v=sigma_v,
                sigma_b=sigma_b,
            )
        self.output_layer.reset_parameters(sigma_v=sigma_v, sigma_b=sigma_b)

    def freeze_grad_lambdas(self):
        self.input_layer.log_lambdas.requires_grad = False
        for layer in self.hidden_layers:
            layer.scale.log_lambdas.requires_grad = False

    def freeze_grad_bias(self):
        if not self.bias:
            for layer in self.hidden_layers:
                layer.linear.bias.requires_grad = False
            self.output_layer.linear.bias.requires_grad = False

    def freeze_output_v(self):
        self.output_layer.v.requires_grad = False
