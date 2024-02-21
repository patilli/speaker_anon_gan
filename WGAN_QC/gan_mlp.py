from torch import nn
from torch.nn import functional as F
import torch


class GeneratorMLP(nn.Module):

    def __init__(self, z_dim, output_dim):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
          nn.Linear(z_dim, z_dim*2),
          nn.GELU(),
          nn.Linear(z_dim*2, z_dim*8),
          nn.GELU(),
          nn.Linear(z_dim*8, z_dim*16),
          nn.GELU(),
          nn.Linear(z_dim*16, output_dim)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def sample_latent(self, n_samples, z_size):
        return torch.randn((n_samples, z_size))

class CriticMLP(nn.Module):

    def __init__(self, z_dim, output_dim):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
          nn.Linear(output_dim, z_dim*16),
          nn.GELU(),
          nn.Linear(z_dim*16, z_dim*8),
          nn.GELU(),
          nn.Linear(z_dim*8, z_dim*2),
          nn.GELU(),
          nn.Linear(z_dim*2, z_dim),
          nn.GELU(),
          nn.Linear(z_dim, 1)
        )

    def forward(self, x):
        out = self.layers(x)
        return out
