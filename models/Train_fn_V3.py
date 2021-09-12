'''

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
from models.evaluation_utils import prepare_z_y
from models.losses import cascade_loss_dis_real, cascade_loss_dis_fake,cascade_gen_loss,loss_dcgan_dis
from models.utils.utils import  get_dataset,save_images_for_evaluation, save_checkpoint
import models.utils.utils as utils

from models.MultiStage_GANV3 import Discriminator, Generator

cudnn.enabled = True
benchmark = True

def train(dataset_name, channel=64, batch_size=64, image_size=32, use_SN=False, epoch_of_updateG=1, 
                dataroot='./data/cifar', cuda=True, model_path='', init_type="ortho", max_iterations=50005):
    workers = 4
    mini_batchSize = batch_size
    if batch_size>=128:
        mini_batchSize = 128
    batchSizeforEvaluation = 100
    num_inception_images = 5000
    nz = 128
    G_lr = 0.0005   #0.0001   #5e-5   1e-5
    D_lr = 0.0005      #0.0004   #2e-4   4e-5
    beta1 = 0.00
    prints_iterations = 500    
    attrIndex = 20     # the attribute index for the male attribute in CelebA
    start_percent = 1.0

    #outf = "DisV3+gamma1+attention+kernel=3"+str(epoch_of_updateG)
    outf = "Ensemble-loss_lr="+str(G_lr)
    log_dir = os.path.join('./log/', outf, dataset_name, str(image_size))
    score_dir = os.path.join('./ImagesForScores/', outf, dataset_name, str(image_size))

    file_name = str(batch_size)+"x"+str(channel)
    log_dir = os.path.join(log_dir, file_name)
    score_dir = os.path.join(score_dir, file_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
        
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataloader, num_classes = get_dataset(dataset_name, dataroot, image_size=image_size, 
                                        batch_size=batch_size, num_workers=workers)
    device = torch.device("cuda" if cuda else "cpu")
    
    netG = Generator(G_ch=channel, dim_z=nz, resolution=image_size, n_classes=num_classes)
    netD = Discriminator(D_ch=channel, resolution=image_size, n_classes=num_classes)
    optimizerG = optim.Adam(netG.parameters(), lr=G_lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=D_lr, betas=(beta1, 0.999))

    if model_path != '':
        stats = torch.load(model_path)
        start_iteration = stats['iterations']
        netG.load_state_dict(stats['gen_dict'])
        netD.load_state_dict(stats['dis_dict'])
        optimizerG.load_state_dict(stats['opti_gen'])
        optimizerD.load_state_dict(stats['opti_dis'])
        for state in optimizerG.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        for state in optimizerD.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    

    if cuda:
        netD = nn.DataParallel(netD)
        netG = nn.DataParallel(netG)
        netD = netD.to(device)
        netG = netG.to(device)
    
    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=G_lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=D_lr, betas=(beta1, 0.999))

    z_, y_ = prepare_z_y(batch_size, nz, num_classes, device=device)
    z_for_sample, y_for_sample = prepare_z_y(batchSizeforEvaluation, nz, num_classes, device=device) 
    fixed_z, fixed_y = prepare_z_y(64, nz, num_classes, device=device)
    fixed_y_one = torch.ones_like(fixed_y)
    fixed_z.sample_()
    fixed_y.sample_()

    sample = functools.partial(utils.sample, G=netG, z_=z_for_sample, y_=y_for_sample)
    
    iterations = 0
    epochs = 0
    max_epochs = 20
    while iterations<max_iterations:
        epochs += 1
        for i, data in enumerate(dataloader, 0):
            if iterations>=max_iterations:
                break
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            real_x = data[0].to(device)
            if dataset_name.lower() == "celeba":
                temp_y = data[1]
                real_y = temp_y[:, attrIndex].to(device)
            else:
                real_y = data[1].to(device)
            real_x_splits = torch.split(real_x, mini_batchSize)
            real_y_splits = torch.split(real_y, mini_batchSize)
            z_.sample_()
            y_.sample_()
            z_splits = torch.split(z_, mini_batchSize)
            y_splits = torch.split(y_, mini_batchSize)

            errD_prints = 0
            
            optimizerD.zero_grad()   
            #errD = 0 
            #topK = start_percent+(1-start_percent)*iterations/max_iterations
            topK = 1
            for step_index in range(len(real_x_splits)):
                errD_reals = netD(real_x_splits[step_index], real_y_splits[step_index])
                real_loss = 0
                real_weights = None
                for index in range(len(errD_reals)):
                    loss, temp_real_weights = cascade_loss_dis_real(errD_reals[index], real_weights)
                    real_loss += loss*(float)(index+1)/len(errD_reals)
                    #real_loss += loss
                    if real_weights is None:
                        real_weights = temp_real_weights.detach()
                    else:
                        real_weights = torch.cat((real_weights, temp_real_weights.detach()), 1)
                #step 2: train dis with fake images
                fake = netG(z_splits[step_index], y_splits[step_index])
                errD_fakes = netD(fake.detach(), y_splits[step_index].detach())
                fake_loss = 0
                fake_weights = None
                for index in range(len(errD_fakes)):
                    loss, temp_fake_weights = cascade_loss_dis_fake(errD_fakes[index], fake_weights, topK=topK)
                    fake_loss += loss*(float)(index+1)/len(errD_fakes)
                    #fake_loss += loss
                    if fake_weights is None:
                        fake_weights = temp_fake_weights.detach()
                    else:
                        fake_weights = torch.cat((fake_weights, temp_fake_weights.detach()), 1)

                real_loss /= float(len(real_x_splits))
                fake_loss /= float(len(real_x_splits))
                #errD_fake = loss_dcgan_dis_fake(output) / float(len(real_x_splits))
                errD = real_loss+fake_loss
                errD.backward()
                
                errD_prints += errD.item()
                if iterations%prints_iterations==0:
                    print("loss_dis_real=", real_loss.item(), "   loss_dis_fake=", fake_loss.item())
                
                del errD
                del real_loss
                del fake_loss
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

                optimizerG.zero_grad()
                for step_index in range(len(z_splits)):
                    fake = netG(z_splits[step_index], y_splits[step_index])
                    outputs = netD(fake, y_splits[step_index])

                    gen_loss = 0
                    gen_weights = None
                    for index in range(len(outputs)):
                        loss, temp_gen_weights = cascade_gen_loss(outputs[index], gen_weights, topK=topK)
                        gen_loss += loss * (float)(index+1)/len(outputs)
                        #gen_loss += loss
                        if gen_weights is None:
                            gen_weights = temp_gen_weights.detach()
                        else:
                            gen_weights = torch.cat((gen_weights, temp_gen_weights.detach()), 1)

                    gen_loss /= float(len(z_splits))
                    gen_loss.backward()
                    errG_prints += gen_loss.detach().item()
                    if iterations%prints_iterations==0:
                        print("loss_gen_real=", gen_loss.detach().item())
                optimizerG.step()
                del gen_loss

            if iterations % prints_iterations == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f '
                    % (iterations, max_iterations, i, len(dataloader),
                        errD_prints, errG_prints))
            if iterations % prints_iterations == 0:
                vutils.save_image(real_x_splits[0],
                        '%s/real_samples.png' % log_dir,
                        normalize=True)
                fake = netG(fixed_z, fixed_y)
                #fake = fake*0.5+0.5
                vutils.save_image(fake.detach(),'%s/fake_samples_epoch_%03d.png' % (log_dir, iterations),
                                    normalize=True)
                fake1 = netG(fixed_z, fixed_y_one)
                #fake1 = fake1*0.5+0.5
                vutils.save_image(fake1.detach(),'%s/fake_samples_one_epoch_%03d.png' % (log_dir, iterations),
                                normalize=True)
                
                del fake
                del fake1
            
            if iterations%2000==0 and i%epoch_of_updateG==0:
                
                states = {
                    'iterations':iterations,
                    'gen_dict':netG.state_dict(),
                    'dis_dict':netD.state_dict(),
                    'opti_gen':optimizerG.state_dict(),
                    'opti_dis':optimizerD.state_dict()
                }
                
                save_checkpoint(states, False, log_dir, iterations)
                this_score_dir = os.path.join(score_dir, str(iterations)+".npz")
                save_images_for_evaluation(sample, n=num_inception_images, batch_size=batchSizeforEvaluation, save_dir=this_score_dir)

            del real_x
            del real_y
            del data    
    