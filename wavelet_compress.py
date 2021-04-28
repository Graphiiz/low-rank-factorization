import tensorly as tl
import torch
import torch.nn as nn
from pywt import fswavedecn
import VBMF

def compress_model(model, model_name):
    
    for name, layer in model._modules.items():
        if len(list(layer.children())) > 0:
            # recurse
            model._modules[name] = compress_model(layer, model_name=model_name)
        else:
            new_layer = compress_layer(layer, name, model_name = model_name)
            model._modules[name] = new_layer

    return model

def compress_layer(layer, name, model_name):
    if type(layer) == nn.Conv2d:
        if model_name == "vgg11_bn":
            last_conv_layer = '25'
        if model_name == "vgg13_bn":
            last_conv_layer = '31'

        if name == '0':
            levels = [1, 0, 0, 0]
        elif name == last_conv_layer:
            levels = [0, 1, 0, 0]
        else:
            levels = [1, 1, 0, 0]
        decomp_layer = wavedec_conv_layer(layer, levels)
        return decomp_layer
    elif type(layer) == nn.BatchNorm2d:
        if name == '26' or name == '32':
            return layer
        decomp_layer = wavedec_bn_layer(layer)
        return decomp_layer
    elif type(layer) == nn.Linear:
        #in linear layer we use tucker1 decomposition
        rank = tucker_rank(layer)
        if rank <= 0:
            print(f'rank <=0 cannot do tucker1 decompostition -> skip {layer}')
            return layer
        decomped_layer = svd_linear_layer(layer,rank)
        return decomped_layer
    else:
        return layer

def tucker_rank(layer):
    """
    Used for linear layer
    """
    weights = layer.weight.data

    _, diag, _, _ = VBMF.EVBMF(weights)

    rank = diag.shape[0]
    return rank

def wavedec_conv_layer(layer, levels):
    w_approx = fswavedecn(layer.weight.data.numpy(), 'rbio6.8', levels=levels).approx

    hasBias = True if layer.bias is not None else False

    new_layer = nn.Conv2d(in_channels=w_approx.shape[1], out_channels=w_approx.shape[0],
                          kernel_size=w_approx.shape[2:], stride=layer.stride, padding=layer.padding,
                          dilation=layer.dilation, bias=hasBias)

    w_approx = torch.from_numpy(w_approx.copy())
    new_layer.weight.data = w_approx

    if hasBias:
        if levels == [0, 1, 0, 0]:
            new_layer.bias.data = layer.bias.data
        else:
            b_approx = fswavedecn(layer.bias.data.numpy(), 'rbio6.8', levels=1).approx
            b_approx = torch.from_numpy(b_approx.copy())
            new_layer.bias.data = b_approx

    return new_layer

def wavedec_bn_layer(layer):
    new_shape = fswavedecn(layer.running_mean.data.numpy(), 'rbio6.8', levels = 1).approx.shape
    new_layer = nn.BatchNorm2d(new_shape, eps=layer.eps, momentum=layer.momentum,
                               affine=layer.affine, track_running_stats=layer.track_running_stats)
    new_layer.weight.data = torch.ones(new_shape)
    new_layer.bias.data = torch.zeros(new_shape)
    return new_layer

def svd_linear_layer(layer,rank):
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
