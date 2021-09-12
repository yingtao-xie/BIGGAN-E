'''
boosting strategy
'''
from __future__ import print_function
import argparse
import numpy as np
import functools
import os
import random
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import math
import torch.nn.functional as F

import models.evaluation_utils as evaluation_utils
from models.evaluation_utils import prepare_inception_metrics_V2, prepare_z_y
from models.losses import  loss_dcgan_gen, loss_dcgan_dis, loss_dcgan_dis_real, loss_dcgan_dis_fake
from models.losses import loss_hinge_dis, loss_hinge_gen, cascade_generator_hinge_loss
from models.utils.utils import  get_dataset,save_images_for_evaluation
import models.utils.utils as utils

from models.MultiStage_BigGANV5 import Discriminator, Generator

cudnn.enabled = True
benchmark = True

def train(dataset_name, channel=64, batch_size=64, image_size=32, use_SN=False, epoch_of_updateG=1, 
                dataroot='./data/cifar', cuda=True, G_path='', D_path='', init_type="ortho", max_iterations=70000):
    
    workers = 4
    mini_batchSize = batch_size
    if batch_size>=256:
        mini_batchSize = 128
    batchSizeforEvaluation = 100
    num_inception_images = 5000
    nz = 128
    G_lr = 1e-4
    D_lr = 4e-4
    beta1 = 0.00
    prints_iterations = 500    
    
    outf = "MultiStage_BigGANV5"
    
    log_dir = os.path.join('./log/', outf, dataset_name, str(image_size))
    score_dir = os.path.join('./ImagesForScores/', outf, dataset_name, str(image_size))

    file_name = str(batch_size)+"x"+str(channel)
    log_dir = os.path.join(log_dir, file_name)
    score_dir = os.path.join(score_dir, file_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
        
    manualSeed = random.randint(1, 100000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataloader, num_classes = get_dataset(dataset_name, dataroot, image_size=image_size, 
                                        batch_size=batch_size, num_workers=workers)

    device = torch.device("cuda" if cuda else "cpu")
    
    netG = Generator(G_ch=channel, dim_z=nz, resolution=image_size, n_classes=num_classes)
    if G_path != '':
        netG.load_state_dict(torch.load(G_path))

    netD = Discriminator(D_ch=channel, resolution=image_size, n_classes=num_classes)
    if D_path != '':
        netD.load_state_dict(torch.load(D_path))
    
    if cuda:
        #netD = nn.DataParallel(netD, device_ids=[0,1,2,3])
        #netG = nn.DataParallel(netG, device_ids=[0,1,2,3])
        netD = netD.to(device)
        netG = netG.to(device)
    
    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=G_lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=D_lr, betas=(beta1, 0.999))

    z_, y_ = prepare_z_y(batch_size, nz, num_classes, device=device)
    z_for_sample, y_for_sample = prepare_z_y(batchSizeforEvaluation, nz, num_classes, device=device) 
    fixed_z, fixed_y = prepare_z_y(mini_batchSize, nz, num_classes, device=device)
    fixed_y_one = torch.ones_like(fixed_y)
    fixed_z.sample_()
    fixed_y.sample_()

    score_tr = np.zeros((1000,4))
    counter=0
    sample1 = functools.partial(utils.sample, G=netG, z_=z_for_sample, y_=y_for_sample)
    sample2 = functools.partial(utils.sample1, G=netG, z_=z_for_sample, y_=y_for_sample)
    
    
    
    iterations = 0
    best_IS = 0
    best_IS_std = 0
    best_FID = 1000000

    resolutions = [2**i for i in range(int(math.log(16,2)), int(math.log(image_size, 2)+1))]

    while iterations<max_iterations:
        for i, data in enumerate(dataloader, 0):
            if iterations>=max_iterations:
                break
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            real_x = data[0].to(device)
            real_y = data[1].to(device)
            real_x_splits = torch.split(real_x, mini_batchSize)
            real_y_splits = torch.split(real_y, mini_batchSize)
            z_.sample_()
            y_.sample_()
            z_splits = torch.split(z_, mini_batchSize)
            y_splits = torch.split(y_, mini_batchSize)

            errD_prints = 0
            
            for resolution in reversed(resolutions):
                optimizerD.zero_grad()   
                scale_size = image_size // resolution
                undersample = nn.AvgPool2d([scale_size, scale_size]) 
                #scale_factor = 1.0 / 2.0**(max_block_depth-block_depth)
                    
                for step_index in range(len(real_x_splits)):
                    #step 1: train net_D at resolution 2^(index+2) with real images
                    if scale_size>1:
                        real_x_copy = undersample(real_x_splits[step_index].detach())
                    else:
                        real_x_copy = real_x_splits[step_index].detach()
                    #real_x_copy = F.upsample(real_x_splits[step_index].detach(), scale_factor=scale_factor, mode='bilinear')
                    
                    
                    #step 2: generate images at resolution 2^(index+2)
                    fake = netG(z_splits[step_index], y_splits[step_index], training=True, resolution=resolution)
                    fak1 = netG(real_x_splits[step_index], real_y_splits[step_index], training=True, resolution=resolution, supervised=False)
                        
                    
                    
                    errD_fake = netD(fake.detach(), y_splits[step_index].detach(), resolution=resolution)
                    
                    errD_prints += loss_fake.detach().item() + loss_real.detach().item()
                    if iterations%prints_iterations==0:
                        print("loss_dis_real=", loss_real.detach().item(), "   loss_dis_fake=", loss_fake.detach().item())
                optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if i % epoch_of_updateG == 0:
                iterations += 1
                z_.sample_()
                y_.sample_()
                z_splits = torch.split(z_, mini_batchSize)
                y_splits = torch.split(y_, mini_batchSize)
                errG_prints = 0

                weights = None
                undersample2 = nn.AvgPool2d(2)

                for resolution in reversed(resolutions):
                    optimizerG.zero_grad()
                    if weights is not None:
                        weights_split = torch.split(weights.detach(), mini_batchSize)
                    this_weights = []

                    scale_size = image_size // resolution
                    undersample = nn.AvgPool2d([scale_size, scale_size]) 
                
                    for step_index in range(len(z_splits)):
                        #step 1: juedge the quality of generated images
                        fake = netG(z_splits[step_index], y_splits[step_index], training=True, resolution=resolution)
                        output = netD(fake, y_splits[step_index], resolution=resolution)
                        #errG_real = loss_hinge_gen(output)/float(len(z_splits))
                        if weights is not None:
                            errG_real, temp_weights = cascade_generator_hinge_loss(output, weights_split[step_index])
                        else:
                            errG_real, temp_weights = cascade_generator_hinge_loss(output)
                        this_weights += [temp_weights]
                        errG_real /= float(len(z_splits))
                        errG_real.backward()
                        errG_prints += errG_real.detach().item()
                        if iterations%prints_iterations==0:
                            print("loss_gen_real=", errG_real.detach().item())
                        
                        #step 2: take netG as autoencoder
                        if scale_size>1:
                            real_x_copy = undersample(real_x_splits[step_index].detach())
                        else:
                            real_x_copy = real_x_splits[step_index].detach()
                        real_x_input = undersample2(real_x_copy.detach())
                        
                        output_coder = netG(real_x_input, real_y_splits[step_index], training=True, resolution=resolution, supervised=False)
                        errG_coder = torch.mean(torch.abs(output_coder-real_x_copy))
                        errG_coder.backward()
                        if iterations%prints_iterations==0:
                            print("loss_coder=", errG_coder.detach().item())
                        

                    this_weights = torch.cat(this_weights, 0)
                    if weights is not None:
                        weights = torch.cat([weights, this_weights.detach()], 1)
                    else:
                        weights = this_weights.detach()
                    optimizerG.step()
                
            if iterations % prints_iterations == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f '
                    % (iterations, max_iterations, i, len(dataloader),
                        errD_prints, errG_prints))
            if iterations % prints_iterations == 0:
                vutils.save_image(real_x_splits[0],
                        '%s/real_samples.png' % log_dir,
                        normalize=True)
                fake = netG(fixed_z, fixed_y)
                vutils.save_image(fake.detach(),'%s/fake_samples_epoch_%03d.png' % (log_dir, iterations),
                                    normalize=True)
                fake1 = netG(fixed_z, fixed_y_one)
                vutils.save_image(fake1.detach(),'%s/fake_samples_one_epoch_%03d.png' % (log_dir, iterations),
                                normalize=True)
            
            if iterations%2000==0 and i%epoch_of_updateG==0:
                IS_mean, IS_std, FID_mean, FID_std = get_inception_metrics(sample1, sample2, batch_size=batchSizeforEvaluation, n=num_inception_images)
                score_tr[counter, 0] = IS_mean
                score_tr[counter, 1] = IS_std
                score_tr[counter, 2] = FID_mean
                score_tr[counter, 3] = FID_std
                counter+=1
                if IS_mean > best_IS:
                    best_IS = IS_mean
                    best_IS_std = IS_std
                    torch.save(netG.state_dict(), '%s/netG_best.pth' % (log_dir))
                    torch.save(netD.state_dict(), '%s/netD_best.pth' % (log_dir))
                if FID_mean<best_FID:
                    best_FID=FID_mean
                    best_FID_std = FID_mean
                
                torch.save(netG.state_dict(), '%s/netG_epoch_%d_.pth' % (log_dir, iterations))
                torch.save(netD.state_dict(), '%s/netD_poech_%d_.pth' % (log_dir, iterations))
                np.save('%s/score_tr_ep.npy' % log_dir, score_tr[0:counter])
                print('[Iterations=%d][IS=%.4f / Best_IS=%.4f] [FID=%.4f / Best_FID=%.4f] '
                            % (iterations, IS_mean, best_IS, FID_mean, best_FID))

    return best_IS, best_IS_std, best_FID, 1