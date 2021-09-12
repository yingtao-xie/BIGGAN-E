import os
import torch
import dateutil.tz
from datetime import datetime
import time
import logging
import numpy as np
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from models.utils.CelebA import CelebA

def save_checkpoint(states, is_best, output_dir, iterations=100):
    filename =  "Epoch_%d_checkpoint.pth" %(iterations)
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def get_mix_up_index(batch_size, alpha=1.0, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    index = torch.randperm(batch_size)
    if use_cuda:
        index = index.cuda()
    return index, lam

def mix_up(x, y, index, lam):
    mix_x = lam * x + (1 - lam) * x[index,:]
    return mix_x, y, y[index]

def get_dataset(dataname, root, image_size=32, batch_size=64, num_workers=4):
    if dataname in ['imagenet', 'folder', 'lfw', 'I128']:
        # folder dataset
        dataset = dset.ImageFolder(root=root,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        num_classes=1000
    elif dataname == 'LSUN':
        dataset = dset.ImageFolder(root=root,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        num_classes=2

    elif dataname == "celeba":
        # folder dataset
        transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
        dataset = CelebA(root=root, split="train", target_type="attr", transform=transform, download=False)
        num_classes=2
        
    elif dataname == 'lsun':
        classes = [ c + '_train' for c in opt.classes.split(',')]
        dataset = dset.LSUN(root=root, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        num_classes=100
    elif dataname == 'C10':
        dataset = dset.CIFAR10(root=root, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    #transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        num_classes=10

    elif dataname == 'C100':
        dataset = dset.CIFAR100(root=root, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
        num_classes=100

    elif dataname == 'mnist':
        dataset = dset.MNIST(root=root, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ]))
        num_classes=10
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    return data_loader, num_classes

def sample(G, z_, y_):
    with torch.no_grad():
        z_.sample_()
        y_.sample_()
        G_z = G(z_, y_)
    return G_z, y_

def sample1(G, z_, y_):
    with torch.no_grad():
        z_.sample_()
        y_.sample_()
        G_z = G(z_, y_, supervised=False)
    return G_z, y_

def save_images_for_evaluation(sample, n=50000, batch_size=50, save_dir='./ImagesForScores'):
    n_batches = n//batch_size
    images = list()
    for i in range(n_batches):
        start = i*batch_size
        end = (i+1)*batch_size
        batch, labels = sample()
        gen_imgs = batch.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        
        images.extend(list(gen_imgs))
    
    images = np.array(images)
    np.savez_compressed(save_dir, images=images)

