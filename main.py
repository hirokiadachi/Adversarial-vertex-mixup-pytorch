import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import argparse
import multiprocessing
import numpy as np
from utils import *
from models.wideresnet import *
from org_dataloader import *

parser = argparse.ArgumentParser(description='PyTorch Adversarial Vertex Mixup')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=2e-4, type=float)
parser.add_argument('--epsilon', default=8.0, type=float)
parser.add_argument('--alpha', default=2.0, type=float)
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
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'cifar100_p', 'svhn', 'tiny_imagenet'])
parser.add_argument('--resume', '-r', type=str, default=None, help='resume from checkpoint')
parser.add_argument('--result_dir', type=str, default='checkpoint')
parser.add_argument('--tb', type=str, default='logs')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--n_cls', type=int, default=10)
parser.add_argument('--seed', type=int, default=np.random.randint(100000))
args = parser.parse_args()

if not os.path.isdir(args.result_dir):
    mkdir_p(args.result_dir)
    
tb_filename = os.path.join(args.result_dir, args.tb)
if os.path.exists(tb_filename):    shutil.rmtree(tb_filename)
tb = SummaryWriter(log_dir=tb_filename)

device = torch.device("cuda:%d" % args.gpu[0] if torch.cuda.is_available() else "cpu")
train_iters = 0
best_acc_clean = 0
best_acc_adv = 0

def main():
    global best_acc_clean, best_acc_adv, model_type
    print('==> Preparing data..')
    
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform_trainlist = [transforms.RandomCrop(32, padding=4),
                           transforms.ToTensor()]
    
    transform_testlist = [transforms.ToTensor()]
    
    trainloader, testloader, net, model_type = get_dataset(transform_trainlist, transform_testlist)
    
    print('==> Building model..')
    net = net.to(device)
    if len(args.gpu) > 1:
        net = nn.DataParallel(net, device_ids=args.gpu)
        
    if args.resume is not None:
        print('==> Resuming from checkpoint..')
        net.load_state_dict(torch.load(args.resume))

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = CustomLossFunction()
    scheduler = [int(args.end_epoch*0.5), int(args.end_epoch*0.75)]
    
    print('='*50)
    print('  -->\tepsilon: {}\n'
          '  -->\talpha: {}\n'
          '  -->\tadjust trigger: {}\n'
          '  -->\tdataset: {}\n'
          '  -->\tGPU number: {}\n'
          '  -->\tEpochs (start/end): ({}/{})\n'
          '  -->\tBatch size: {}\n'
          '  -->\tAdv iterations: {}\n'
          '  -->\tPGD repeat (eval): {}\n'
          '  -->\tSeed value: {}'.format(args.epsilon, args.alpha, scheduler, args.dataset, args.gpu, args.start_epoch, args.end_epoch, args.batch_size, args.m, args.pgd_repeat, args.seed))
    print('='*50)
    
    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, scheduler, epoch)
        train_loss, train_acc = train(epoch, trainloader, net, criterion, optimizer, args.n_cls)
        test_clean_loss, test_clean_acc = validation_normal(epoch, testloader, net, args.n_cls, criterion)
        test_adv_loss, test_adv_acc = validation_pgd(epoch, testloader, net, criterion, args.n_cls, n_repeat=args.pgd_repeat)
        tb.add_scalars('Loss', {'val/clean': test_clean_loss, 'val/robustness': test_adv_loss}, epoch)
        tb.add_scalars('Accuracy', {'val/clean': test_clean_acc, 'val/robustness': test_adv_acc}, epoch)
        
        if test_clean_acc > best_acc_clean and test_adv_acc > best_acc_adv:
            print('==> Updating the best model..')
            best_acc_clean = test_clean_acc
            best_acc_adv = test_adv_acc
            torch.save(net.state_dict(), os.path.join(args.result_dir, 'best_model_at'+'_'+args.dataset+'_'+model_type+'_'+str(args.seed)))
    
    
def get_dataset(transform_trainlist, transform_testlist):
    if args.dataset == 'cifar100_p':    use_dataset = args.dataset.upper()
    else:    use_dataset = datasets.__dict__[args.dataset.upper()]
    if args.dataset == 'svhn':
        trainset = use_dataset(root='/root/work/Datasets/data', split='train', download=True, transform=transforms.Compose(transform_trainlist))
        testset = use_dataset(root='/root/work/Datasets/data', split='test', download=True, transform=transforms.Compose(transform_testlist))
        depth, widen_factor = 16, 8
        model_type = 'WRN-16-8'
    elif args.dataset == 'tiny_imagenet':
        transform_trainlist.insert(1, transforms.RandomHorizontalFlip())
        common_path = '/root/work/Datasets/tiny-imagenet-200'
        trainset = datasets.ImageFolder(os.path.join(common_path, 'train'), transforms.Compose(transform_trainlist))
        testset = datasets.ImageFolder(os.path.join(common_path, 'test'), transforms.Compose(transform_testlist))
        model_type = 'PreActResNet-18'
    elif args.dataset == 'cifar100_p':
        transform_trainlist.insert(1, transforms.RandomHorizontalFlip())
        datapath = '/root/work/Datasets/cifar100'
        trainset = CIFAR100_P(datapath=datapath, n_use_cls=args.n_cls, train=True, transform=transforms.Compose(transform_trainlist), seed=args.seed)
        testset = CIFAR100_P(datapath=datapath, n_use_cls=args.n_cls, train=False, transform=transforms.Compose(transform_testlist), seed=args.seed)
        depth, widen_factor = 34, 10
        model_type = 'WRN-34-10'
    else:
        transform_trainlist.insert(1, transforms.RandomHorizontalFlip())
        trainset = use_dataset(root='/root/work/Datasets/data', train=True, download=True, transform=transforms.Compose(transform_trainlist))
        testset = use_dataset(root='/root/work/Datasets/data', train=False, download=True, transform=transforms.Compose(transform_testlist))
        depth, widen_factor = 34, 10
        model_type = 'WRN-34-10'
    print('==> Model type: %s' % model_type)

    if args.dataset == 'tiny_imagenet':    net=resnet18(pretrained=True, num_classes=args.n_cls)
    else:    net=WideResNet(depth=depth, num_classes=args.n_cls, widen_factor=widen_factor)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return trainloader, testloader, net, model_type

def train(epoch, trainloader, net, criterion, optimizer, n_classes, random_start=True):
    global train_iters
    print('\nEpoch: %d' % epoch)
    net.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    epsilon = args.epsilon / 255
    alpha = args.alpha / 255
    
    for batch_idx, (img, tgt) in enumerate(trainloader):
        img, tgt = img.to(device), tgt.to(device)
        onehot = torch.eye(n_classes)[tgt].to(device)
        
        if random_start:
            noise = torch.FloatTensor(img.shape).uniform_(-epsilon, epsilon).to(device)
            x = img + noise
            x = x.clamp(min=0., max=1.)
        else:
            x = img.clone()
        x.requires_grad_()
        
        for i in range(args.m):
            x.requires_grad_()
            out = net(x)
            loss = criterion.softlabel_ce(out, onehot)
            grads = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
            
            x = x.detach() + alpha * torch.sign(grads.detach())
            x = torch.min(torch.max(x, img - epsilon), img + epsilon)
            x = torch.clamp(x, min=0.0, max=1.0)
          
        pert = (x - img) * args.gamma
        x_av = img + pert
        x_av = x_av.clamp(min=0., max=1.)
        y_nat = label_smoothing(onehot, 10, args.lambda1)
        y_ver = label_smoothing(onehot, 10, args.lambda2)
        #policy = np.random.beta(1.0, 1.0)
        policy_x = torch.from_numpy(np.random.beta(1, 1, [x.size(0), 1, 1, 1])).float()
        policy_y = policy_x.view(x.size(0), -1)
        x = policy_x * img + (1 - policy_x) * x_av
        y = policy_y * y_nat + (1 - policy_y) * y_ver
        
        out = net(x)
        loss = criterion.softlabel_ce(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_iters += 1
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
            tb.add_scalars('train_loss', {'loss': losses.avg}, train_iters)
            tb.add_scalars('train_acc', {'top1': top1.avg, 'top5': top5.avg}, train_iters)
    
    print('Saving..')
    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    torch.save(net.state_dict(), os.path.join(result_dir, args.model+'_'+args.dataset+'_'+model_type+'_'+str(args.seed)))
    return losses.avg, top1.avg
            
def validation_normal(epoch, testloader, net, n_classes, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()
    for idx, (img, tgt) in enumerate(testloader):
        img, tgt = img.to(device), tgt.to(device)
        onehot = torch.eye(n_classes)[tgt].to(device)
        b = img.size(0)
        out = net(img)
        loss = criterion.softlabel_ce(out, onehot)
            
        prec1, prec5 = accuracy(out, tgt, topk=(1,5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1[0], img.size(0))
        top5.update(prec5[0], img.size(0))
            
    print('  Top1 Acc (clean): {top1.val: .4f} ({top1.avg: .4f})'
          '  Top5 Acc (clean): {top5.val: .4f} ({top5.avg: .4f})'.format(top1=top1, top5=top5))
    return losses.avg, top1.avg

def validation_pgd(epoch, testloader, net, criterion, n_cls, n_repeat=10):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    alpha = args.alpha/255.
    epsilon = args.epsilon/255.
    
    net.eval()
    for idx, (img, tgt) in enumerate(testloader):
        img, tgt = img.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        b = img.size(0)
        onehot = torch.eye(n_cls)[tgt].to(device, non_blocking=True)
        org_img = img.clone()
            
        randn = torch.FloatTensor(img.size()).uniform_(-epsilon, epsilon).to(device)
        img += randn
        img.clamp_(0., 1.0)
        for _ in range(n_repeat):
            img1 = torch.autograd.Variable(img, requires_grad=True)
            out = net(img1)
            ascend_loss = criterion.softlabel_ce(out, onehot)
            grads = torch.autograd.grad(ascend_loss, img1, grad_outputs=None, only_inputs=True)[0]
            pert = alpha * torch.sign(grads)
                
            # adversarial examples: linf norm
            img += pert.data
            img = torch.max(org_img - epsilon, img)
            img = torch.min(org_img + epsilon, img)
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
    tb.add_scalars('Acc_adv', {'top1': top1.avg, 'top5': top5.avg}, epoch)
    return losses.avg, top1.avg
        
def adjust_learning_rate(optimizer, scheduler, epoch):
    if epoch in scheduler:
        for param_group in optimizer.param_groups:
            lr = param_group['lr'] * 0.1
            param_group['lr'] = lr
        
if __name__ == '__main__':
    main()
        