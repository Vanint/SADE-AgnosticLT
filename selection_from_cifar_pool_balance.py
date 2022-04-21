'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler 
import models.cifar as models
import torchvision.models
import clip
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import PIL

from sklearn.model_selection import train_test_split
from randaugment import rand_augment_transform 


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])) 
model_names.append('resnet50')
model_names.append('CLIP-VIT-B32')
model_names.append('CLIP-VIT-L14')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data_save_dir', default='data', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--expanded_number', default=50000, type=int)
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[41, 81],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
#parser.add_argument('--augdata_dir', default='data/augment_data', type=str, metavar='PATH',
#                    help='path to save new data (default: checkpoint)')

# RandAugment
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--m', default=10, type=int)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='CLIP-ViT-B32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--accumulate',  type=int, default= 0)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

 
def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
         os.makedirs(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
 
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
    ])
     
    
    #trainset_pool = datasets.ImageFolder(args.data_save_dir+'/noselection',  transform) 
    trainset_pool = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    class_names = trainset_pool.classes
    data_number = len(trainset_pool)
    print("total", len(trainset_pool))
    #trainset = train_dataloader(root='./data', train=True, download=True, transform=transform_train)
    #indices = torch.randperm(len(trainset_pool)).numpy()
    indices = np.arange(0, len(trainset_pool))
    saved_indices, out_indices = train_test_split(indices, train_size=args.expanded_number, stratify=trainset_pool.targets) 
    saved_subset = torch.utils.data.Subset(trainset_pool, saved_indices)
    print("saved_subset", len(saved_subset))
    
    saved_loader = data.DataLoader(saved_subset, batch_size=1, shuffle=False, num_workers=args.workers)
    #train_loader_batch1 = data.DataLoader(trainset_pool, batch_size=1, shuffle=False, num_workers=args.workers)
    
   
    # Model
    print("==> creating model '{}'".format(args.arch))
    # create model
    num_classes = len(class_names)
    dim_feature = 2048    
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    elif args.arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True) 
        model.fc = nn.Linear(dim_feature, num_classes)
    elif args.arch == 'CLIP-VIT-B32':     
        model, preprocess = clip.load("ViT-B/32")       
        text_descriptions = [f"This is a photo of a {label}" for label in class_names]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_classifier = text_features
    elif args.arch == 'CLIP-VIT-L14':     
        model, preprocess = clip.load("ViT-L/14")       
        text_descriptions = [f"This is a photo of a {label}" for label in class_names]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_classifier = text_features        
        
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        
        
    model = model.cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Resume
    title = 'cifar-100-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

 

    print("Start random selection from the pool:")
 
    class_save_list = [0]*100 
  
    for inputs, targets in saved_loader:
        inputs = inputs[0]
        convert_image = transform_convert(inputs,transforms.Compose([transforms.ToTensor()])) 
        path = args.data_save_dir + "/{}".format(class_names[targets])
        if not os.path.isdir(path): 
            os.makedirs(path)
        convert_image.save(args.data_save_dir + "/{}/expanded_{}.png".format(class_names[targets], class_save_list[targets])) 
        class_save_list[targets] += 1      
    
    
    print("class_save_list", class_save_list)  
 
def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])
        
    img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
    
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().cpu().numpy()*255
    
    if isinstance(img_tensor, torch.Tensor):
    	img_tensor = img_tensor.cpu().numpy()
    
    if img_tensor.shape[2] == 3:
        img = PIL.Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = PIL.Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2])) 
    return img 


def save_checkpoint(state, is_best, best_acc, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        print("The best performance:", best_acc)
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
 
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

