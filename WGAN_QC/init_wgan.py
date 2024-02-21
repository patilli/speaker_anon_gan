import logging

import torch

from WGAN_QC.resnet_init import init_resnet
from WGAN_QC.init_mlp import init_mlp
from WGAN_QC.wgan_qc import WassersteinGanQuadraticCost

LOGGER = logging.getLogger(__name__)

def create_wgan(logger, parameters, device, optimizer='adam'):
    if parameters['model'] == "resnet":
        generator, discriminator = init_resnet(parameters)
    elif parameters['model'] == "mlp":
        generator, discriminator = init_mlp(parameters)
    else:
        raise NotImplementedError

    LOGGER.info(generator)
    LOGGER.info(discriminator)

    n_parameters = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    LOGGER.info(f'generator number of params: {n_parameters}')

    n_parameters = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    LOGGER.info(f'discriminator number of params: {n_parameters}')

    if optimizer == 'adam':
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'])
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'])
    elif optimizer == 'rmsprop':
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=parameters['learning_rate'])
        optimizer_d = torch.optim.RMSprop(generator.parameters(), lr=parameters['learning_rate'])

    criterion = torch.nn.MSELoss()

    gan = WassersteinGanQuadraticCost(generator, 
                                      discriminator, 
                                      optimizer_g, 
                                      optimizer_d, 
                                      criterion=criterion, 
                                      data_dimensions=parameters['data_dim'], 
                                      epochs=parameters['epochs'], 
                                      batch_size=parameters['batch_size'], 
                                      device=device, 
                                      n_max_iterations=parameters['n_max_iterations'], 
                                      use_cuda=torch.cuda.is_available(),
                                      gamma=parameters['gamma'])
    
    return gan
