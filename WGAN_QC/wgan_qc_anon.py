import logging
import os
import time
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cvxopt import matrix, solvers, sparse, spmatrix
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)

class WassersteinGanQuadraticCost():
    def __init__(self, generator, discriminator_dist, discriminator_asv, gen_optimizer, dis_dist_optimizer,
                 dis_asv_optimizer, criterion, epochs, n_max_iterations, data_dimensions, batch_size, device,
                 gamma=1, K=-1, milestones=[150000,250000], lr_anneal=1.0, use_cuda=False, device_ids=None,
                 model_id=None):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D_dist = discriminator_dist
        self.D_dist_opt = dis_dist_optimizer
        self.D_asv = discriminator_asv
        self.D_asv_opt = dis_asv_optimizer
        self.losses = {'D': [], 'WD': [], 'D_ASV': []}
        self.num_steps = 0
        self.gen_steps = 0
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.n_max_iterations = n_max_iterations
        # put in the shape of a dataset sample
        self.data_dim = np.prod(data_dimensions) #data_dimensions[0] * data_dimensions[1] * data_dimensions[2]
        self.batch_size = batch_size
        self.device = device
        self.criterion = criterion
        self.asv_criterion = nn.CrossEntropyLoss()
        self.mone = torch.FloatTensor([-1]).to(device)
        self.tensorboard_counter = 0
        self.model_name = f'{model_id}_{datetime.now().strftime("%d-%m-%y")}'
        self.current_epoch = 0
        self.triplet_loss = nn.TripletMarginLoss(swap=True)

        self.discriminators = {
            'real': 1.0,
            'speaker': 0.0,
            'triplet': 0.0
        }
        self.discriminators_after_threshold = {
            'real': 0.5,
            'speaker': 0.0,
            'triplet': 0.5
        }
        self.discriminator_threshold = 0.00001
        self.discriminator_threshold_passed = False

        if K <= 0:
            self.K = 1 / self.data_dim
        else:
            self.K = K
        self.Kr = np.sqrt(self.K)
        self.LAMBDA = 2 * self.Kr * gamma * 2

        if device_ids:
            self.G = nn.DataParallel(self.G, device_ids=device_ids)
            self.D_dist = nn.DataParallel(self.D_dist, device_ids=device_ids)
            self.D_asv = nn.DataParallel(self.D_asv, device_ids=device_ids)

        self.schedulerD_dist = self._build_lr_scheduler_(self.D_dist_opt, milestones, lr_anneal)
        self.schedulerD_asv = self._build_lr_scheduler_(self.D_asv_opt, milestones, lr_anneal)
        self.schedulerG = self._build_lr_scheduler_(self.G_opt, milestones, lr_anneal)

        self.c, self.A, self.pStart = self._prepare_linear_programming_solver_(self.batch_size)
    
    def _build_lr_scheduler_(self, optimizer, milestones, lr_anneal, last_epoch=-1):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_anneal, last_epoch=-1)
        return scheduler
            
    def _quadratic_wasserstein_distance_(self, real, generated):
        num_r = real.size(0)
        num_f = generated.size(0)
        real_flat = real.view(num_r, -1)
        fake_flat = generated.view(num_f, -1)

        real3D = real_flat.unsqueeze(1).expand(num_r, num_f, self.data_dim)
        fake3D = fake_flat.unsqueeze(0).expand(num_r, num_f, self.data_dim)
        # compute squared L2 distance
        dif = real3D - fake3D
        dist = 0.5 * dif.pow(2).sum(2).squeeze()
        
        return self.K * dist
    
    def _prepare_linear_programming_solver_(self, batch_size):
        A = spmatrix(1.0, range(batch_size), [0]*batch_size, (batch_size,batch_size))
        for i in range(1,batch_size):
            Ai = spmatrix(1.0, range(batch_size), [i]*batch_size, (batch_size,batch_size))
            A = sparse([A,Ai])

        D = spmatrix(-1.0, range(batch_size), range(batch_size), (batch_size,batch_size))
        DM = D
        for i in range(1,batch_size):
            DM = sparse([DM, D])

        A = sparse([[A],[DM]])

        cr = matrix([-1.0/batch_size]*batch_size)
        cf = matrix([1.0/batch_size]*batch_size)
        c = matrix([cr,cf])

        pStart = {}
        pStart['x'] = matrix([matrix([1.0]*batch_size),matrix([-1.0]*batch_size)])
        pStart['s'] = matrix([1.0]*(2*batch_size))

        return c, A, pStart

    def _linear_programming_(self, distance, batch_size):
        b = matrix(distance.cpu().double().detach().numpy().flatten())
        sol = solvers.lp(self.c, self.A, b, primalstart=self.pStart, solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
        offset = 0.5 * (sum(sol['x'])) / batch_size
        sol['x'] = sol['x'] - offset
        self.pStart['x'] = sol['x']
        self.pStart['s'] = sol['s']

        return sol
    
    def _approx_OT_(self, sol):
        # Compute the OT mapping for each fake dataset
        ResMat = np.array(sol['z']).reshape((self.batch_size, self.batch_size))
        mapping = torch.from_numpy(np.argmax(ResMat, axis=0)).long().to(self.device)
        
        return mapping

    def _optimal_transport_regularization_(self, output_fake, fake, real_fake_diff):
        output_fake_grad = torch.ones(output_fake.size()).to(self.device)
        gradients = torch_grad(outputs=output_fake, inputs=fake,
                                  grad_outputs=output_fake_grad,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]  
        n = gradients.size(0)    
        RegLoss = 0.5 * ((gradients.view(n, -1).norm(dim=1) / (2*self.Kr) - self.Kr/2 * real_fake_diff.view(n, -1).norm(dim=1)).pow(2)).mean()
        fake.requires_grad = False

        return RegLoss

    def _critic_deep_regression_(self, speaker_embeds, speaker_ids, opt_iterations=1):
        logger.debug('Start train iteration critic')
        if self.use_cuda:
            speaker_embeds       = speaker_embeds.to(self.device)
            speaker_ids = speaker_ids.to(self.device)
        
        for p in self.D_dist.parameters():  # reset requires_grad
            p.requires_grad = True     # they are set to False below in netG update

        
        self.G.train()
        self.D_dist.train()

        # Get generated fake dataset
        logger.debug('Generate samples')
        start = time.time()
        generated_data = self.sample_generator(self.batch_size, speaker_embeds)
        end = time.time()
        logger.debug(f'Time for generating samples: {round(end - start, 4)}')

        # compute wasserstein distance
        logger.debug('Start computing wasserstein distance')
        start = time.time()
        distance = self._quadratic_wasserstein_distance_(speaker_embeds, generated_data)
        end = time.time()
        logger.debug(f'Time for computing wasserstein distance: {round(end - start, 4)}')
        logger.info('Distance for linear programming: {:.9f}'.format(distance.mean()))
        # solve linear programming problem
        start = time.time()
        sol = self._linear_programming_(distance, self.batch_size)
        end = time.time()
        logger.debug(f'Time for solving linear programming problem: {round(end - start, 4)}')
        # approximate optimal transport
        start = time.time()
        mapping = self._approx_OT_(sol)
        end = time.time()
        logger.debug(f'Time for approximating optimal transport: {round(end - start, 4)}')
        real_ordered = speaker_embeds[mapping]  # match real and fake
        real_fake_diff = real_ordered - generated_data
        
        # construct target
        target = torch.from_numpy(np.array(sol['x'])).float()
        target = target.squeeze().to(self.device)

        for i in range(opt_iterations):
            logger.debug('Start opt train iteration critic')
            start = time.time()
            self.D_dist.zero_grad() # ???
            self.D_dist_opt.zero_grad()
            generated_data.requires_grad_()
            if generated_data.grad is not None:
                generated_data.grad.data.zero_()
            output_real = self.D_dist(speaker_embeds)
            output_fake = self.D_dist(generated_data)
            output_real, output_fake = output_real.squeeze(), output_fake.squeeze()
            output_R_mean = output_real.mean(0).view(1)
            output_F_mean = output_fake.mean(0).view(1)

            logger.debug('Output real mean:    {:.9f}'.format(output_R_mean[0]))
            logger.debug('Output fake mean:    {:.9f}'.format(output_F_mean[0]))

            logger.debug('Target real mean:    {:.9f}'.format(target[:self.batch_size].mean()))
            logger.debug('Target fake mean:    {:.9f}'.format(target[self.batch_size:].mean()))

            L2LossD_real = self.criterion(output_R_mean[0], target[:self.batch_size].mean())
            L2LossD_fake = self.criterion(output_fake, target[self.batch_size:])
            L2LossD = 0.5 * L2LossD_real + 0.5 * L2LossD_fake

            logger.debug('L2LossD real:        {:.9f}'.format(L2LossD_real))
            logger.debug('L2LossD fake:        {:.9f}'.format(L2LossD_fake))
            logger.debug('L2LossD:             {:.9f}'.format(L2LossD))

            reg_loss_D = self._optimal_transport_regularization_(output_fake, generated_data, real_fake_diff)
            logger.debug('OTR loss:            {:.9f}'.format(reg_loss_D))

            total_loss = L2LossD + self.LAMBDA * reg_loss_D

            logger.info('Total loss:          {:.9f}'.format(total_loss))

            self.losses['D'].append(float(total_loss.data))

            total_loss.backward()
            self.D_dist_opt.step()

            # this is supposed to be the wasserstein distance
            wasserstein_distance = output_R_mean - output_F_mean
            self.losses['WD'].append(float(wasserstein_distance.data))
            end = time.time()
            logger.debug(f'Time for opt train iteration critic: {round(end - start, 4)}')
            logger.debug('End opt train iteration critic')


    def _speaker_discriminator_train_iteration(self, speaker_embeds, speaker_ids):
        logger.debug('Start train iteration speaker discriminator')
        start = time.time()
        for p in self.D_asv.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        self.D_asv.zero_grad()
        self.D_asv_opt.zero_grad()

        generated_data = self.sample_generator(self.batch_size, speaker_embeds)
        generated_data.requires_grad_()
        if generated_data.grad is not None:
            generated_data.grad.data.zero_()
        output_speaker_real = self.D_asv(speaker_embeds)
        output_speaker_fake = self.D_asv(generated_data)

        speaker_loss_real = self.asv_criterion(output_speaker_real, speaker_ids)
        speaker_loss_fake = self.asv_criterion(output_speaker_fake, speaker_ids)
        speaker_loss = 0.5 * speaker_loss_real + 0.5 * speaker_loss_fake

        self.losses['D_ASV'].append(float(speaker_loss.data))
        logger.info(f'Speaker Recognition loss: {float(speaker_loss.data)}')

        speaker_loss.backward()
        self.D_asv_opt.step()
        end = time.time()
        logger.debug(f'Time for train iteration speaker discriminator: {round(end - start, 4)}')

    def _create_triplet_samples(self, speaker_ids):
        combinations = []
        for speaker in torch.unique(speaker_ids):
            this_speaker_indices = torch.where(speaker_ids == speaker)[0]
            if len(this_speaker_indices) > 1:
                other_speaker_indices = torch.where(speaker_ids != speaker)[0]
                positive_combinations = torch.combinations(this_speaker_indices)
                negative_choices = other_speaker_indices[torch.randint(len(other_speaker_indices),
                                                                       (len(positive_combinations), ))].unsqueeze(1)
                combinations.append(torch.cat([positive_combinations, negative_choices], dim=1))
        combinations = torch.cat(combinations, dim=0)
        return combinations

    def _generator_train_iteration(self, speaker_embeds, speaker_ids):
        logger.debug('Start train Iteration Generator')
        for p in self.D_dist.parameters():
            p.requires_grad = False  # freeze critic
        
        self.G.zero_grad()
        self.G_opt.zero_grad()
        fake = self.G(speaker_embeds)

        # Wassersteing GAN
        start = time.time()
        output_fake = self.D_dist(fake)
        loss_fake = self.criterion(output_fake, torch.zeros(output_fake.shape).to(self.device))
        g_loss = self.discriminators.get('real', 1.0) * loss_fake
        logger.info(f'G loss fake: {loss_fake.data}')
        end = time.time()
        logger.debug(f'Time for computing g loss real: {round(end - start, 4)}')

        if self.discriminators.get('speaker', 0.0) > 0.0:
            start = time.time()
            output_speaker = self.D_asv(fake)
            loss_speaker = self.asv_criterion(output_speaker, speaker_ids)
            g_loss += self.discriminators['speaker'] * loss_speaker
            logger.info(f'G loss speaker: {loss_speaker.data}')
            end = time.time()
            logger.debug(f'Time for computing g loss speaker: {round(end - start, 4)}')

        if self.discriminators.get('triplet', 0.0) > 0.0:
            start = time.time()
            triplet_samples = fake[self._create_triplet_samples(speaker_ids)]
            anchors = triplet_samples[:, 0, :]
            positives = triplet_samples[:, 1, :]
            negatives = triplet_samples[:, 2, :]
            triplet_loss = self.triplet_loss(anchors, positives, negatives)
            g_loss += self.discriminators['triplet'] * triplet_loss
            logger.info(f'Triplet loss: {triplet_loss.data}')
            end = time.time()
            logger.debug(f'Time for computing g triplet loss: {round(end - start, 4)}')

        logger.info(f'Total generator loss: {g_loss.data}')

        g_loss.backward()
        self.G_opt.step()

        self.schedulerD_dist.step()

        if self.discriminators.get('speaker', 0.0) > 0.0:
            self.schedulerD_asv.step()

        self.schedulerG.step()
        logger.debug('End train Iteration Generator')

        if not self.discriminator_threshold_passed and g_loss.data < self.discriminator_threshold:
            logger.info('-- Discriminator threshold passed!')
            self.discriminator_threshold_passed = True
            self.discriminators = self.discriminators_after_threshold

    def _train_epoch(self, data_loader, models_dir):
        for i, data in enumerate(data_loader):
            logger.debug(f'---- Batch {i} ----')
            #if self.use_cuda:
            #    dataset = dataset.cuda()
            speaker_embeds, speaker_ids = data
            self.num_steps += 1
            self.tensorboard_counter += 1
            if self.gen_steps >= self.n_max_iterations:
                logger.info("Break dataloader loop - Max iterations reached - Stop training")
                return
            self._critic_deep_regression_(speaker_embeds, speaker_ids)
            if self.discriminators.get('speaker', 0.0) > 0.0:
                self._speaker_discriminator_train_iteration(speaker_embeds, speaker_ids)
            self._generator_train_iteration(speaker_embeds, speaker_ids)

            if self.num_steps % 50 == 0:
                self._save_models(models_dir)

    def _save_models(self, models_dir):
        timestamp = datetime.now().strftime('%d-%m-%y_%H:%M')
        checkpoint = f'{timestamp}.pth'

        if isinstance(self.G, nn.DataParallel):
            G = self.G.module
        else:
            G = self.G
        generator_settings = {
            'vec_type': 'ecapa+xvector',
            'model_name': self.model_name,
            'model_path': checkpoint,
            'emb_level': 'utt',
            'input_dim': G.input_dim,
            'output_dim': G.output_dim,
            'dropout_rate': G.dropout_rate
        }
        self._save_single_model(model=G, model_dir=models_dir / 'gan_generator', settings=generator_settings,
                                checkpoint=checkpoint)

        if isinstance(self.D_dist, nn.DataParallel):
            D_dist = self.D_dist.module
        else:
            D_dist = self.D_dist
        discr_dist_settings = {
            'model_name': self.model_name,
            'model_path': checkpoint,
            'input_dim': D_dist.input_dim,
            'output_dim': D_dist.output_dim
        }
        self._save_single_model(model=D_dist, model_dir=models_dir / 'gan_discriminator_dist',
                                settings=discr_dist_settings, checkpoint=checkpoint)

        if self.discriminators.get('speaker', 0.0) > 0.0:
            if isinstance(self.D_asv, nn.DataParallel):
                D_asv = self.D_asv.module
            else:
                D_asv = self.D_asv
            discr_asv_settings = {
                'model_name': self.model_name,
                'model_path': checkpoint,
                'input_dim': 704,
                'output_dim': 1160
            }
            self._save_single_model(model=D_asv, model_dir=models_dir / 'gan_discriminator_asv',
                                    settings=discr_asv_settings, checkpoint=checkpoint)


    def _save_single_model(self, model, model_dir, settings, checkpoint):
        model_dir = model_dir / self.model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        torch.save(model.cpu().state_dict(), model_dir / checkpoint)
        model.to(self.device)

        with open(model_dir / 'settings.json', 'w') as f:
            json.dump(settings, f)


    def train(self, data_loader, writer, models_dir):
        self.G.train()
        self.D_dist.train()

        for epoch in range(self.epochs):
            if self.gen_steps >= self.n_max_iterations:
                logger.info("Break epoch loop - Max iterations reached - Stop training")
                return
            self.current_epoch = epoch
            time_start_epoch = time.time()
            logger.info('Epoch: {:03d}'.format(epoch))
            self._train_epoch(data_loader, models_dir)

            D_loss_avg = np.average(self.losses['D'])
            wd_avg = np.average(self.losses['WD'])
            D_loss_asv_avg = np.average(self.losses['D_ASV'])

            writer.add_scalar('GAN/loss/critic', D_loss_avg, self.tensorboard_counter)
            writer.add_scalar('GAN/wasserstein-distance', wd_avg, self.tensorboard_counter)
            writer.add_scalar('GAN/loss/speaker_recognition', D_loss_asv_avg, self.tensorboard_counter)

            logger.info("Iterations: {:07d}".format(self.num_steps))
            logger.info("Critic loss: {:.9f}".format(D_loss_avg))
            logger.info("Wasserstein distance: {:.9f}".format(self.losses['WD'][-1]))
            logger.info(f'Speaker Recognition Loss: {round(D_loss_asv_avg, 9)}')

            time_end_epoch = time.time()
            logger.info('Epoch: {:03d} - time: {:.2f}s'.format(epoch, time_end_epoch - time_start_epoch))

        return self

    def sample_generator(self, num_samples, input_data, nograd=False):
        #self.G.eval()
        #latent_samples = self.G.sample_latent(num_samples)
        if self.use_cuda:
            input_data = input_data.to(self.device)
        if nograd:
            with torch.no_grad():
                generated_data = self.G(input_data)
        else:
            generated_data = self.G(input_data)
        #self.G.train()
        return generated_data.detach()

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]

    def save_model_checkpoint(self, model_path, model_parameters, timestampStr, mean, std):
            #dateTimeObj = datetime.now()
            #timestampStr = dateTimeObj.strftime("%d-%m-%Y-%H-%M-%S")
            name = '%s_%s'%(timestampStr, 'wgan')
            model_filename = os.path.join(model_path, name)
            torch.save({
                'generator_state_dict': self.G.state_dict(),
                'critic_state_dict': self.D_dist.state_dict(),
                'asv_discriminator_state_dict': self.D_asv.state_dict(),
                'gen_optimizer_state_dict': self.G_opt.state_dict(),
                'critic_optimizer_state_dict': self.D_dist_opt.state_dict(),
                'asv_optimizer_state_dict': self.D_asv_opt.state_dict(),
                'model_parameters': model_parameters,
                'iterations': self.num_steps,
                'dataset_mean': mean,
                'dataset_std': std
                }, model_filename)
                