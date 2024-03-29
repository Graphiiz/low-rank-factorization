# low-rank-factorization
## Pretrained weight 
- You can get the pretrained weight of VGG11, VGG13 and VGG16 from https://github.com/huyvnphan/PyTorch_CIFAR10. 
- Put the directory of pretrained weights and model from the [link](https://github.com/huyvnphan/PyTorch_CIFAR10) above in the same directory. Ex. /low-rank-factorization/cifar10_models

## How to run the code (tucker decomposition)
### Compress model with fine-tuning

`python pretrained_main.py --model vgg16_bn --dataset CIFAR10 --batch_size 128 --fine_tune --epoch 30 --lr 0.0001 --save`

### Compress model without fine-tuning

`python pretrained_main.py --model vgg16_bn --dataset CIFAR10 --batch_size 128 --save`

-You can change the argument as follows(with available options):
 * --model | vgg11_bn, vgg13_bn, vgg16_bn
 * --batch_size
 * --epoch 
 * --lr
 * --epoch
### Wavelet decomposition code
- See wavelet_compress.py
## Reference
- Tucker Decomposition for convolutional layers is described here: https://arxiv.org/abs/1511.06530
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
- VBMF code was taken from here: https://github.com/CasvandenBogaard/VBMF
- Tensorly: https://github.com/tensorly/tensorly
- For Tucker-Decomposition, we use the code from https://github.com/jacobgil/pytorch-tensor-decompositions 
