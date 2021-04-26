import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
import traceback
from collections import OrderedDict
import VBMF

def decomp_model(model):
    
    for name, layer in model._modules.items():
        if len(list(layer.children())) > 0:
            # recurse
            model._modules[name] = decomp_model(layer)
        else:
            new_layer = decomp_layer(layer)
            model._modules[name] = new_layer

    return model

def decomp_layer(layer):
    if type(layer) == nn.Conv2d:
        #check rank
        ranks = tucker_ranks(layer)
        if (any(r <= 0 for r in ranks)):
            print(f"One or more of the estimated ranks are 0 or less -> cannot do tucker decomposition -> Skip {layer}")
            return layer
        # elif (np.prod(ranks) >= conv_layer.in_channels * conv_layer.out_channels):
        #     print(f"np.prod(ranks) = {np.prod(ranks)} >= conv_layer.in_channels * conv_layer.out_channels) -> Tucker provides no compression. Skip {layer}")
        #     return layer
        else:   
            #do tucker decomp
            decomped_layer = tucker_decomp_layer(layer,ranks)
            #todo add data reconstruction optimization
            # data_reconstruction_optimization(layer,decomped_layer)
            return decomped_layer

    elif type(layer) == nn.Linear:
        #in linear layer we use tucker1 decomposition
        rank = tucker1_rank(layer)
        if rank <= 0:
            print(f'rank <=0 cannot do tucker1 decompostition -> skip {layer}')
            return layer
        decomped_layer = svd_linear_layer(layer,rank)
        return decomped_layer
    else:
        return layer

    
def tucker_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data.numpy()

    unfold_0 = tl.base.unfold(weights, 0) 
    unfold_1 = tl.base.unfold(weights, 1)

    unfold_0 = torch.from_numpy(unfold_0)
    unfold_1 = torch.from_numpy(unfold_1)
    
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)

    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def tucker1_rank(layer):
    """
    Used for linear layer
    """
    weights = layer.weight.data

    _, diag, _, _ = VBMF.EVBMF(weights)

    rank = diag.shape[0]
    return rank

def tucker_decomp_layer(layer,ranks):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """
    #mode=[0,1] -> 
    core, [last, first] = \
        partial_tucker(layer.weight.data.numpy(), \
            modes=[0, 1], rank=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=True)

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    last = torch.from_numpy(last.copy()) #convert from numpy to torch tensor --if error occurs when you use torch.from_numpy(nd.array), use nd.array.copr() instead
    core = torch.from_numpy(core.copy())
    first = torch.from_numpy(first.copy())

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

def svd_linear_layer(layer,rank):
    #linear layer is split into 2 linear layer, [fc_a=core,fc_b=last]
    # core, [last] = tl.decomposition.partial_tucker(layer.weight.data.numpy(),modes=[0],rank = [rank],init='svd')
    # fc_a = torch.nn.Linear(in_features=core.shape[1], out_features=core.shape[0], bias=False)
    # fc_b = torch.nn.Linear(in_features=last.shape[1], out_features=last.shape[0], bias=True)
    
    # if layer.bias is not None:
    #     fc_b.bias.data = layer.bias.data

    # fc_b_weight = torch.from_numpy(last.copy()) #convert from numpy to torch tensor --if error occurs when you use torch.from_numpy(nd.array), use nd.array.copr() instead
    # fc_a_weight = torch.from_numpy(core.copy()) #convert from numpy to torch tensor
    # fc_b.weight.data = fc_b.weight.unsqueeze(-1).unsqueeze(-1)
    # fc_a.weight.data = fc_a.weight

    # new_layers = [fc_a, fc_b]
    # return nn.Sequential(*new_layers)
    [U, S, V] = tl.partial_svd(layer.weight.data.numpy(), rank)

    first_layer = torch.nn.Linear(in_features=V.shape[1], out_features=V.shape[0], bias=False)
    second_layer = torch.nn.Linear(in_features=U.shape[1], out_features=U.shape[0], bias=True)

    if layer.bias is not None:
        second_layer.bias.data = layer.bias.data
    #convert numpy to torch tensor
    V = torch.from_numpy(V.copy())
    S = torch.from_numpy(S.copy())
    U = torch.from_numpy(U.copy())

    first_layer.weight.data = (V.t() * S).t()
    second_layer.weight.data = U

    new_layers = [first_layer, second_layer]
    return nn.Sequential(*new_layers)