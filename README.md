# low-rank-factorization
## Pretrained weight 
- You can get the pretrained weight of VGG11, VGG13 and VGG16 from https://github.com/huyvnphan/PyTorch_CIFAR10. 
## How to run the code
`python pretrained_main.py --model vgg16_bn --dataset CIFAR10 --batch_size 128 --fine_tune --epoch 30 --lr 0.0001 --save
-You can change following arguments:
  - 
## Reference
- Tucker Decomposition for convolutional layers is described here: https://arxiv.org/abs/1511.06530
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
- VBMF code was taken from here: https://github.com/CasvandenBogaard/VBMF
- Tensorly: https://github.com/tensorly/tensorly
