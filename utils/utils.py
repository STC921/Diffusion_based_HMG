import torch.nn as nn
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def get_motion_shape(dataset="h36m"):
    if dataset == "h36m":
        data = (128, 125, 17, 3)
        num_frames = 240
        keep_ratio = 0.20
    elif dataset == "humaneva":
        data = (128, 75, 15, 3)
        num_frames = 75
        keep_ratio = 0.10
    else:
        data = None
        num_frames = None
        keep_ratio = None

    return data, num_frames, keep_ratio


def normal_init_(layer, mean_, sd_, bias, norm_bias=True):
    """Intialization of layers with normal distribution with mean and bias"""
    classname = layer.__class__.__name__
    # Only use the convolutional layers of the module
    #if (classname.find('Conv') != -1 ) or (classname.find('Linear')!=-1):
    if classname.find('Linear') != -1:
        print('[INFO] (normal_init) Initializing layer {}'.format(classname))
        layer.weight.data.normal_(mean_, sd_)
        if norm_bias:
            layer.bias.data.normal_(bias, 0.05)
        else:
            layer.bias.data.fill_(bias)


def weight_init(
    module, 
    mean_=0, 
    sd_=0.004, 
    bias=0.0, 
    norm_bias=False, 
    init_fn_=normal_init_):
  """Initialization of layers with normal distribution"""
  moduleclass = module.__class__.__name__
  try:
    for layer in module:
      if layer.__class__.__name__ == 'Sequential':
        for l in layer:
          init_fn_(l, mean_, sd_, bias, norm_bias)
      else:
        init_fn_(layer, mean_, sd_, bias, norm_bias)
  except TypeError:
    init_fn_(module, mean_, sd_, bias, norm_bias)



def xavier_init_(layer, mean_, sd_, bias, norm_bias=True):
    classname = layer.__class__.__name__
    if classname.find('Linear')!=-1:
        print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
        nn.init.xavier_uniform_(layer.weight.data)
        # nninit.xavier_normal(layer.bias.data)
        if norm_bias:
            layer.bias.data.normal_(0, 0.05)
        else:
            layer.bias.data.zero_()