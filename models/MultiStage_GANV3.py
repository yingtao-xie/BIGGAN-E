'''
V1+Attention
'''

import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from .layers import bn, identity, ccbn, SNEmbedding, SNConv2d, SNLinear, Attention, G_arch, D_arch

from .sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d

# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must 
# be preselected)
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None, upsample=None):
        super(GBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        #self.conv1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                        kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

        
    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        out = h+x
        return out

class Branch_Block(nn.Module):
    def __init__(self, in_channels, which_conv=SNConv2d, which_linear=SNLinear, n_classes=1000,
                which_embedding=SNEmbedding,preactivation=False, activation=None, downsample=None):
        super(Branch_Block, self).__init__()
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        self.conv1 = which_conv(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = which_conv(in_channels, in_channels)

        self.linear = which_linear(in_channels, 1)
        # Embedding for projection discrimination
        self.embed = which_embedding(n_classes, in_channels)
    
    def forward(self, x, y):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x    
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h) 
        
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out


    
# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, which_embedding=SNEmbedding,
                which_linear=SNLinear, n_classes=1000, wide=True,
                preactivation=False, activation=None, downsample=None, last_block=False):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
            
        # Conv layers
        if last_block==False:
            self.Branch = Branch_Block(out_channels, which_conv=which_conv, n_classes=n_classes,
                preactivation=preactivation, activation=activation, downsample=downsample)
        else:
            self.Branch = None
            
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                        kernel_size=1, padding=0)
    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x
        
    def forward(self, x, y=None):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it 
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x    
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)
        out = h+self.shortcut(x)
        if self.Branch is not None:
            branch_loss = self.Branch(out, y)
            return out, branch_loss
        else:
            return out
             

class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
                n_classes=1000, G_shared=True, shared_dim=0, hier=True,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=True),
               BN_eps=1e-5, SN_eps=1e-12, G_init='ortho', skip_init=False,
               G_param='SN', norm_style='bn',
               **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Architecture dict
        self.arch = G_arch(self.ch)[resolution]

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size *  self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                            kernel_size=3, padding=1,
                            eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                            eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
        
        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                    else self.which_embedding)
        self.which_bn = functools.partial(ccbn,
                            which_linear=bn_linear,
                            cross_replica=self.cross_replica,
                            mybn=self.mybn,
                            input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                        else self.n_classes),
                            norm_style=self.norm_style,
                            eps=self.BN_eps)


        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                        else identity())
        # First linear layer
        self.linear = self.which_linear(self.dim_z // self.num_slots,
                                        self.arch['in_channels'][0] * (self.bottom_width **2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        block_num = len(self.arch['out_channels'])
        for index in range(block_num):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                out_channels=self.arch['out_channels'][index],
                                which_conv=self.which_conv,
                                which_bn=self.which_bn,
                                activation=self.activation,
                                upsample=(functools.partial(F.interpolate, scale_factor=2)
                                        if self.arch['upsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

            
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                        self.activation,
                                        self.which_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

  # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) 
                or isinstance(module, nn.Linear) 
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y):
        # If hierarchical, concatenate zs and ys
        y = self.shared(y)   #modification
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)
      
        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])
        images = self.output_layer(h)
        images = torch.tanh(images)
        return images

class Discriminator(nn.Module):
    def __init__(self, D_ch=64, D_wide=True, resolution=128, n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=True),
               SN_eps=1e-12, output_dim=1,
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
        super(Discriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Architecture
        self.arch = D_arch(self.ch)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                kernel_size=3, padding=1,
                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                eps=self.SN_eps)
            self.which_embedding = functools.partial(SNEmbedding,
                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                    eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            self.which_embedding = functools.partial(SNEmbedding,
                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                    eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        block_num = len(self.arch['out_channels'])
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                        out_channels=self.arch['out_channels'][index],
                        which_conv=self.which_conv,
                        which_embedding=self.which_embedding,
                        which_linear=self.which_linear,
                        n_classes=n_classes,
                        wide=self.D_wide,
                        activation=self.activation,
                        preactivation=(index > 0),
                        downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None),
                        last_block=(index==block_num-1))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                                    self.which_conv)]
            
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        
  # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        losses = []
        for index, blocklist in enumerate(self.blocks):
            if index<len(self.blocks)-1:
                for block in blocklist:
                    if not isinstance(block, Attention):
                        h, loss = block(h, y)
                        losses += [loss]
                    else:
                        h = block(h, y)
            else:
                for block in blocklist:
                    h = block(h, y)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        losses += [out]
        return losses
