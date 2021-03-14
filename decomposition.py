import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
import traceback
from collections import OrderedDict
import VBMF


class EnergyThreshold(object):

    def __init__(self, threshold, eidenval=True):
        """
        :param threshold: float, threshold to filter small valued sigma:
        :param eidenval: bool, if True, use eidenval as criterion, otherwise use singular
        """
        self.T = threshold
        assert self.T < 1.0 and self.T > 0.0
        self.eiden = eidenval

    def __call__(self, sigmas):
        """
        select proper numbers of singular values
        :param sigmas: numpy array obj which containing singular values
        :return: valid_idx: int, the number of sigmas left after filtering
        """
        if self.eiden:
            energy = sigmas**2
        else:
            energy = sigmas

        sum_e = torch.sum(energy)
        valid_idx = sigmas.size(0)
        for i in range(energy.size(0)):
            if energy[:(i+1)].sum()/sum_e >= self.T:
                valid_idx = i+1
                break

        return valid_idx

def decompose_model(model, type, config):
    config["criterion"] = None
    if config["rank"] is not None and config["threshold"] is not None:
        raise Exception("Either threshold or rank can be set. Not both.")
    elif config["rank"] is None:
        if type in ["tucker", "cp"]:
            config["criterion"] = VBMF
        # else:
        #     if config["threshold"] is None:
        #         config["threshold"] = 0.85
        #     config["criterion"] = EnergyThreshold(config["threshold"])

    layer_configs = get_per_layer_config(model, config, type)
    
    if type == 'tucker':
        return tucker_decompose_model(model, layer_configs)
    elif type == 'cp':
        return cp_decompose_model(model, exclude_first_conv=False, exclude_linears=True) #cp doesn't need layer_configs
    # elif type == 'channel':
    #     return channel_decompose_model(model, layer_configs)
    # elif type == 'depthwise':
    #     return depthwise_decompose_model(model, layer_configs)
    # elif type == 'spatial':
    #     return spatial_decompose_model(model, layer_configs)
    else:
        raise Exception(('Unsupported decomposition type passed: ' + type))

#tucker related function
def get_per_layer_config(model, config, decomp_type, passed_first_conv=False):
    layer_configs = {}

    # TODO: handle conflicts in settings
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            layer_configs.update(get_per_layer_config(module, config, decomp_type, passed_first_conv))
        elif type(module) == nn.Conv2d:
            conv_layer = module 

            # pop the mask list and check the value of current mask
            enable_current_conv = True
            if config["mask_conv_layers"] is not None:
                enable_current_conv = not mask_conv_layers.pop(0)

            if config["conv_ranks"] is not None:
                if decomp_type != "tucker" or passed_first_conv is False:
                    current_conv_rank = config["conv_ranks"].pop(0)
                else:
                    current_conv_rank = [config["conv_ranks"].pop(0), config["conv_ranks"].pop(0)]
            elif config["rank"] is not None:
                current_conv_rank = config["rank"]
            else:
                current_conv_rank = None
           
            if not passed_first_conv and config["exclude_first_conv"]:
                layer_configs.update({conv_layer: (None, None)}) #dict.update() = add key:value to the existing dict
            elif enable_current_conv is False:
                layer_configs.update({conv_layer: (None, None)})
            elif current_conv_rank is None:
                layer_configs.update({conv_layer: (None, config["criterion"])})
            elif current_conv_rank is not None:
                layer_configs.update({conv_layer: (current_conv_rank, None)})
                
            if passed_first_conv is False:
                passed_first_conv = True

        elif type(module) == nn.Linear:
            linear_layer = module

            if config["exclude_linears"] is True:
                layer_configs.update({linear_layer: (None, None)})
            else:
                layer_configs.update({linear_layer: (None, config["criterion"])})

    return layer_configs

def tucker_decompose_model(model, layer_configs):
    '''
    decompose filter NxCxHxW to 3 filters:
    R1xCx1x1 , R2xR1xHxW, and NxR2x1x1
    Unlike other decomposition methods, it requires 2 ranks
    '''
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = tucker_decompose_model(module, layer_configs)
        elif type(module) == nn.Conv2d:
            conv_layer = module 
            print(conv_layer)

            (set_ranks, criterion) = layer_configs[conv_layer]

            if set_ranks is not None and criterion is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                ranks = tucker_ranks(conv_layer)
            elif set_ranks is not None:
                ranks = set_ranks
            elif set_ranks is None and criterion is None:
                print("\tExcluding layer")
                continue
            print("\tRanks: ", ranks)

            # check if Tucker-1 rank or Tucker-2 ranks
            if np.isscalar(ranks):
                rank = ranks
                is_tucker2 = False
            else:
                is_tucker2 = True

            if (is_tucker2):
                if (np.prod(ranks) >= conv_layer.in_channels * conv_layer.out_channels):
                    print("np.prod(ranks) >= conv_layer.in_channels * conv_layer.out_channels)")
                    continue

                if (any(r <= 0 for r in ranks)):
                    print("One of the estimated ranks is 0 or less. Skipping layer")
                    continue

                decomposed = tucker_decomposition_conv_layer(conv_layer, ranks)
            else:
                if (rank <= 0):
                    print("The estimated rank is 0 or less. Skipping layer")
                    continue
                    
                decomposed = tucker1_decomposition_conv_layer(conv_layer, rank)

            model._modules[name] = decomposed
        elif type(module) == nn.Linear:
            linear_layer = module
            print(linear_layer)

            (set_rank, criterion) = layer_configs[linear_layer]

            if set_rank is not None and criterion is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                rank = tucker1_rank(linear_layer)

                print(linear_layer, "Tucker1 Estimated rank", rank)
                # hack to deal with the case when rank is very small (happened with ResNet56 on CIFAR10) and could deteriorate accuracy
                if rank < 2: 
                    rank = svd_rank_linear(linear_layer)
                    print("Will instead use SVD Rank (using 90% rule) of ", rank, "for layer: ", linear_layer)
            elif set_rank is not None:
                rank = min(set_rank, dim[1])
            elif set_rank is None and criterion is None:
                print("\tExcluding layer")
                continue

            decomposed = svd_decomposition_linear_layer(linear_layer, rank)

            model._modules[name] = decomposed

    return model

def tucker_decomposition_conv_layer(layer, ranks):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """
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

def tucker1_decomposition_conv_layer(layer, rank):
    core, [last] = partial_tucker(layer.weight.data.numpy(), modes=[0], rank=rank, init='svd')
    #don't forget to convert tensor to numpy for partial_tucker function
    '''
    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)
    '''

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
    core = torch.from_numpy(core.copy()) #convert from numpy to torch tensor
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [core_layer, last_layer]
    return nn.Sequential(*new_layers)

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
    weights = layer.weight.data

    _, diag, _, _ = VBMF.EVBMF(weights)

    rank = diag.shape[0]
    return rank

def svd_decomposition_linear_layer(layer, rank):
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

def svd_rank(weight, criterion):
    _, S, _ = torch.svd(weight, some=True) # tl.partial_svd(weight, min(weight.shape))

    return criterion(S)

def svd_rank_linear(layer, criterion=EnergyThreshold(0.85)):
    return svd_rank(layer.weight.data, criterion)


#cp related function
def cp_decompose_model(model, exclude_first_conv=False, exclude_linears=False, passed_first_conv=False):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = cp_decompose_model(module, exclude_first_conv, exclude_linears, passed_first_conv)
        elif type(module) == nn.Conv2d:
            if passed_first_conv is False:
                passed_first_conv = True
                if exclude_first_conv is True:
                    continue

            conv_layer = module
            rank = cp_rank(conv_layer)
            print(conv_layer, "CP Estimated rank", rank)

            if (rank**2 >= conv_layer.in_channels * conv_layer.out_channels):
                print("(rank**2 >= conv_layer.in_channels * conv_layer.out_channels")
                continue
            
            decomposed = cp_decomposition_conv_layer(conv_layer, rank)

            model._modules[name] = decomposed
        # TODO: Revisit this part to decide how to deal with linear layer in CP Decomposition  
          
        # elif type(module) == nn.Linear:
        #     if exclude_linears is True:
        #         continue
        #     linear_layer = module 
        #     rank = svd_rank_linear(linear_layer)
        #     print(linear_layer, "SVD Estimated Rank (using 90% rule): ", rank)

        #     decomposed = svd_decomposition_linear_layer(linear_layer, rank)
           
        #     model._modules[name] = decomposed

    return model

def cp_decomposition_conv_layer(layer, rank):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensorly. 
    #last, first, vertical, horizontal = \
        #parafac(layer.weight.data, rank=rank, init='svd')
    k = parafac(layer.weight.data.numpy(), rank=rank, init='svd')
    last = torch.from_numpy(k[1][0])
    first = torch.from_numpy(k[1][1])
    vertical = torch.from_numpy(k[1][2])
    horizontal = torch.from_numpy(k[1][3])

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, 
            dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1], 
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
            groups=vertical.shape[1], bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1], \
            out_channels=horizontal.shape[1], 
            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
            padding=(0, layer.padding[0]), 
            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
            out_channels=last.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=True)

    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    depthwise_horizontal_layer.weight.data = \
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]
    
    return nn.Sequential(*new_layers)

def cp_rank(layer):
    weights = layer.weight.data.numpy()

    # Method used in previous repo
    # rank = max(layer.weight.shape)//3
    # return rank

    unfold_0 = tl.base.unfold(weights, 0) 
    unfold_1 = tl.base.unfold(weights, 1)
    unfold_0 = torch.from_numpy(unfold_0)
    unfold_1 = torch.from_numpy(unfold_1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)

    rank = max([diag_0.shape[0], diag_1.shape[0]])
    return rank