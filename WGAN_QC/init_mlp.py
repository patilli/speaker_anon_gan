
from WGAN_QC.gan_mlp import GeneratorMLP
from WGAN_QC.gan_mlp import CriticMLP

def init_mlp(parameters):
    
    generator = GeneratorMLP(parameters['z_dim'], parameters['data_dim'][-1])
    critic = CriticMLP(parameters['z_dim'], parameters['data_dim'][-1])

    return generator, critic