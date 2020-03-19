import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from visdom import Visdom

import numpy as np
import argparse
import json
import cv2
import dataset
import time
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('fish_train', metavar='TRAIN',
                    help='path to train.txt')
parser.add_argument('fish_val', metavar='VALUE',
                    help='path to test.txt')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task', metavar='TASK', type=str,
                    help='task id to use.')



def main():
    global args, best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.original_lr = 1e-5
    args.lr = 1e-5
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 100
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    with open(args.fish_train, 'r') as outfile:
        train_list = outfile.read().split(',')
        # print(train_list)
    with open(args.fish_val, 'r') as outfile:
        val_list = outfile.read().split(',')
        # print(val_list)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    #加载模型
    model = CSRNet()

    # model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()
    # criterion = nn.MSELoss(size_average=False)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    # 加载断点保存的模型
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    #可视化
    viz = Visdom()
    viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
    viz.line([0.], [0], win='val_acc', opts=dict(title='val_loss'))
    global_step = 0
    losses=0
    accuracy=0
    Loss_list = []
    Accuracy_list=[]
    # 保存模型
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_list, model, criterion, optimizer, epoch)
        # 绘制实时损失函数曲线
        viz.line([losses], [global_step], win = 'loss', update = 'append')
        Loss_list.append(losses)

        # 验证集
        prec1 = validate(val_list, model, criterion)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task)

        #画准确率曲线
        viz.line([accuracy], [global_step], win='accuracy', update='append')
        Accuracy_list.append(accuracy)
        global_step += 1

    csv_save(Loss_list)
    csv_save(Accuracy_list)

def train(train_list, model, criterion, optimizer, epoch):
    #初始化loss和accuracy

    # 查找这个是什么函数
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            seen=model.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)
        # 给指定位置添加维度
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)

        loss = criterion(output, target)
        #为什么要更新loss
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # 打印训练情况
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    return losses

#保存文件
def csv_save(mylist):
    df= pd.DataFrame(data=[mylist])
    df.to_csv("train_loss.csv",header=False,index=False)


def validate(val_list, model, criterion):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=args.batch_size)
    # 计算评价
    model.eval()

    mae  = 0
    accuracy = 0
    total = 0

    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
        total += target.sum().type(torch.FloatTensor).cuda()

    mae_average = mae / len(test_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae_average))
    accuracy = 1-(mae/total)

    return mae_average, accuracy

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()