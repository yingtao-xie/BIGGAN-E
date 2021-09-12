import torch
import torch.nn.functional as F

def cascade_gen_loss(input, weights=None, gamma=2, topK=1.0):
  #logpt = -input
  logpt = F.softplus(-input)
  #logpt = F.relu(1.0-input)
  pt = torch.exp(-logpt)
  #pt = F.sigmoid(input)
  topk = (int)(input.size(0)*topK)
  topk_loss,preds = torch.topk(pt, topk, dim=0, largest=False)
  #loss = -input
  
  if weights is None:
    p = pt*1
    p = p.view(len(input), 1)
    p = (1-p)**gamma
    loss = p.clone().detach() * logpt
    weights = pt.detach()
    #loss = logpt
  else:
    weights = torch.cat([weights, pt], 1)
    #p,_ = torch.min(instance_weight, 1, keepdim=True)
    p = torch.mean(weights, 1)
    #p = weights[:, -1]
    p = p.view(len(input), 1)
    p = (1-p)**gamma
    loss = p.clone().detach()*logpt

  
  #loss = torch.mean(loss[preds])
  loss = torch.mean(logpt)
  return loss, weights

def cascade_loss_dis_real(dis_real, weights=None, gamma=2):
  logpt = F.softplus(-dis_real)
  #logpt = F.relu( - dis_real)
  pt = torch.exp(-logpt)
  #pt = F.sigmoid(dis_real)
  #loss = F.relu(1. - dis_real)
  
  if weights is None:
    p = pt*1
    p = p.view(len(dis_real), 1)
    p = (1-p)**gamma
    loss = p.clone().detach() * logpt
    weights = pt.detach()
    
    #loss = logpt
  else:
    weights = torch.cat([weights, pt], 1)
    #p,_ = torch.min(instance_weight, 1, keepdim=True)
    p = torch.mean(weights, 1)
    p = p.view(len(dis_real), 1)
    p = (1-p)**gamma
    loss = p.clone().detach()*logpt
  
  loss = torch.mean(logpt)
  return loss, weights

def cascade_loss_dis_fake(dis_fake, weights=None, gamma=2, topK=1.0):
  logpt = F.softplus(dis_fake)
  #logpt = F.relu( dis_fake)
  pt = torch.exp(-logpt)
  #pt = F.sigmoid(dis_fake)
  #pt = 1.0-pt
  topk = (int)(dis_fake.size(0)*topK)
  topk_loss, preds = torch.topk(pt, topk, dim=0, largest=False)
  #loss = F.relu(1. + dis_fake)
  
  if weights is None:
    p = pt*1
    p = p.view(len(dis_fake), 1)
    p = (1-p)**gamma
    loss = p.clone().detach() * logpt
    weights = pt.detach()
    #loss = logpt
  else:
    weights = torch.cat([weights, pt], 1)
    #p,_ = torch.min(instance_weight, 1, keepdim=True)
    p = torch.mean(weights, 1)
    p = p.view(len(dis_fake), 1)
    p = (1-p)**gamma
    loss = p.clone().detach()*logpt
  
  
  #loss = torch.mean(loss[preds])
  loss = torch.mean(logpt)
  return loss, weights


# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


def loss_hinge_dis_fake(dis_fake):
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  #loss_fake = torch.mean(F.softplus(dis_fake))
  return loss_fake

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  #loss = torch.mean(F.softplus(-dis_fake))
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis