import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import argparse
import multiprocessing
import numpy as np
from utils import *
from models.wideresnet import *

parser = argparse.ArgumentParser(description='PyTorch Adversarial Vertex Mixup')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=2e-4, type=float)
parser.add_argument('--epsilon', default=8.0/255, type=float)
parser.add_argument('--alpha', default=2.0/255, type=float)
parser.add_argument('--gamma', default=2.0, type=float)
parser.add_argument('--lambda1', default=0.5, type=float)
parser.add_argument('--lambda2', default=0.7, type=float)
parser.add_argument('--m', default=8, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=27)
parser.add_argument('--pgd_repeat', type=int, default=10)
parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count())
parser.add_argument('-g', '--gpu', nargs='*', type=int, required=True)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'svhn', 'tiny_imagenet'])
parser.add_argument('--resume', '-r', type=str, default=None, help='resume from checkpoint')
parser.add_argument('--result_dir', type=str, default='checkpoint')
parser.add_argument('--tb', type=str, default='logs')
args = parser.parse_args()
device = torch.device("cuda:%d" % args.gpu[0] if torch.cuda.is_available() else "cpu")
iters = 0
best_acc_clean = 0
best_acc_adv = 0

def main():
    global best_acc_clean, best_acc_adv, model_type
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()])
    
    trainloader, testloader, net, model_type = get_dataset(transform_train, transform_test)
    
    print('==> Building model..')
    net = net.to(device)
    if len(args.gpu) > 1:
        net = nn.DataParallel(net, device_ids=args.gpu)
        
    if args.resume is not None:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = CustomLossFunction()
    
    print('='*50)
    print('  -->\tepsilon: {}\n'
          '  -->\talpha: {}\n'
          '  -->\tdataset: {}\n'
          '  -->\tGPU number: {}\n'
          '  -->\tEpochs (start/end): ({}/{})\n'
          '  -->\tBatch size: {}\n'
          '  -->\tAdv iterations: {}\n'
          '  -->\tPGD repeat (eval): {}'.format(args.epsilon, args.alpha, args.dataset, args.gpu, args.start_epoch, args.end_epoch, args.batch_size, args.m, args.pgd_repeat))
    print('='*50)
    
    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc = train(epoch, trainloader, net, criterion, optimizer)
        test_clean_loss, test_clean_acc = validation_normal(epoch, testloader, net, criterion)
        test_adv_loss, test_adv_acc = validation_pgd(epoch, testloader, net, criterion, n_repeat=args.pgd_repeat)
        
        if test_clean_acc > best_acc_clean and test_adv_acc > best_acc_adv:
            print('==> Updating the best model..')
            best_acc_clean = test_clean_acc
            best_acc_adv = test_adv_acc
            torch.save(net.state_dict(), os.path.join(args.result_dir, 'best_model_at'+'_'+args.dataset+'_'+model_type))
    
    
def get_dataset(transform_train, transform_test):
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/root/work/Datasets/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/root/work/Datasets/data', train=False, download=True, transform=transform_test)
        net = WideResNet_34_10_CIFAR10()
        model_type = 'WRN-34-10'
        print('==> Model type: Wide-ResNet-34-10')
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='/root/work/Datasets/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='/root/work/Datasets/data', train=False, download=True, transform=transform_test)
        net = WideResNet_34_10_CIFAR100()
        model_type = 'WRN-34-10'
        print('==> Model type: Wide-ResNet-34-10')
    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='/root/work/Datasets/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root='/root/work/Datasets/data', train=False, download=True, transform=transform_test)
        net = WideResNet_16_8_SVHN()
        model_type = 'WRN-16-8'
        print('==> Model type: Wide-ResNet-16-8')
    else:
        assert 0, 'Dataset %s is not supported.' % args.dataset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.workers)
    return trainloader, testloader, net, model_type

def train(epoch, trainloader, net, criterion, optimizer, random_start=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    global iters
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (img, tgt) in enumerate(trainloader):
        img, tgt = img.to(device), tgt.to(device)
        onehot = torch.eye(10)[tgt].to(device)
        
        if random_start:
            x = img + torch.FloatTensor(img.shape).uniform_(-args.epsilon, args.epsilon).to(device)
            x = x.clamp(min=0., max=1.)
        else:
            x = img.clone()
        x.requires_grad_()
        
        for i in range(args.m):
            out = net(x)
            loss = criterion.softlabel_ce(out, onehot)
            grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True)[0]
            
            x = x + args.alpha*torch.sign(grads)
            
            max_x = img + args.epsilon
            min_x = img - args.epsilon
            x = torch.max(torch.min(x, max_x), min_x)
            x = x.clamp(min=0., max=1.)
            
        pert = (x - img) * args.gamma
        x_av = img + pert
        x_av = x_av.clamp(min=0., max=1.)
        y_nat = label_smoothing(onehot, 10, args.lambda1)
        y_ver = label_smoothing(onehot, 10, args.lambda2)
        policy = np.random.beta(1.0, 1.0)
        x = policy * img + (1 - policy) * x_av
        y = policy * y_nat + (1 - policy) * y_ver
        
        out = net(x)
        loss = criterion.softlabel_ce(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iters += 1
        if batch_idx % 100 == 0:
            with torch.no_grad():
                clean_out = net(img)
                clean_loss = criterion.softlabel_ce(clean_out, onehot)
            prec1, prec5 = accuracy(clean_out, tgt, topk=(1,5))
            losses.update(clean_loss.item(), img.size(0))
            top1.update(prec1[0], img.size(0))
            top5.update(prec5[0], img.size(0))
                
            print('  Loss: {loss.val: .4f} ({loss.avg: .4f})'
                  '  Acc (top1): {top1.val: .4f} ({top1.avg: .4f})'
                  '  Acc (top5): {top5.val: .4f} ({top5.avg: .4f})'.format(loss=losses, top1=top1, top5=top5))
            #tb.add_scalars('Acc_train', {'top1': top1.avg, 'top5': top5.avg}, iters)
    
    print('Saving..')
    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    torch.save(net.state_dict(), os.path.join(result_dir, args.model+'_'+args.dataset+'_'+model_type))
    return losses.avg, top1.avg
            
def validation_normal(epoch, testloader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()
    for idx, (img, tgt) in enumerate(testloader):
        img, tgt = img.to(device), tgt.to(device)
        onehot = torch.eye(10)[tgt].to(device)
        b = img.size(0)
        out = net(img)
        loss = criterion.softlabel_ce(out, onehot)
            
        prec1, prec5 = accuracy(out, tgt, topk=(1,5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1[0], img.size(0))
        top5.update(prec5[0], img.size(0))
            
    print('  Top1 Acc (clean): {top1.val: .4f} ({top1.avg: .4f})'
          '  Top5 Acc (clean): {top5.val: .4f} ({top5.avg: .4f})'.format(top1=top1, top5=top5))
    #tb.add_scalars('Acc_clean', {'top1': top1.avg, 'top5': top5.avg}, epoch)
    return losses.avg, top1.avg

def validation_pgd(epoch, testloader, net, criterion, n_repeat=10):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()
    for idx, (img, tgt) in enumerate(testloader):
        img, tgt = img.to(device), tgt.to(device)
        b = img.size(0)
        onehot = torch.eye(10)[tgt].to(device)
        org_img = img.clone()
            
        randn = torch.FloatTensor(img.size()).uniform_(-args.epsilon, args.epsilon).to(device)
        img += randn
        img.clamp_(0, 1.0)
        for _ in range(n_repeat):
            img1 = torch.autograd.Variable(img, requires_grad=True)
            out = net(img1)
            ascend_loss = criterion.softlabel_ce(out, onehot)
            grads = torch.autograd.grad(ascend_loss, img1, grad_outputs=None, only_inputs=True)[0]
            pert = args.alpha * torch.sign(grads)
                
            # adversarial examples: linf norm
            img += pert.data
            img = torch.max(org_img - args.epsilon, img)
            img = torch.min(org_img + args.epsilon, img)
            img.clamp_(0, 1.0)
                
        with torch.no_grad():
            out = net(img)
            loss = criterion.softlabel_ce(out, onehot)
        prec1, prec5 = accuracy(out, tgt, topk=(1,5))
                
        losses.update(loss.item(), img.size(0))
        top1.update(prec1[0], img.size(0))
        top5.update(prec5[0], img.size(0))
        
    print('  Top1 Acc (AT): {top1.val: .4f} ({top1.avg: .4f})'
          '  Top5 Acc (AT): {top5.val: .4f} ({top5.avg: .4f})'.format(top1=top1, top5=top5))
    #tb.add_scalars('Acc_adv', {'top1': top1.avg, 'top5': top5.avg}, epoch)
    return losses.avg, top1.avg
        
def adjust_learning_rate(optimizer, epoch):
    if epoch < 12:
        lr = 0.1
    elif epoch >= 12 and epoch < 22:
        lr = 0.01
    elif epoch >= 22: 
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
if __name__ == '__main__':
    main()
        