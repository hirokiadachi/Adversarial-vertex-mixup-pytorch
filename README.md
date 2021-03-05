# Adversarial-vertex-mixup-pytorch
Pytorch implementation of "Adversarial Vertex Mixup: Toward Better Adversarially Robust Generalization"

Official implementation (tensorflow): https://github.com/Saehyung-Lee/cifar10_challenge \
Paper: https://arxiv.org/abs/2003.02484

## Accuracy and Robustness
* Network: WideResNet32-10
* Batch size: 128
* Num of iters: 80000 iterations(about 200 epochs)
* Dataset: CIFAR-10
* Optimizer (learning rate): Momentum SGD (0.1)

|     |Clean|FGSM |PGD-10|PGD-20|CW-20|
|:---:|:---:|:---:|:----:|:----:|:---:|
|Paper results|93.24| 78.25| 62.67 | 58.23 | 53.63|
|Impremented results|94.97|83.69|62.63|57.54|40.5|
