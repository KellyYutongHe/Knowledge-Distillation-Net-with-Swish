import argparse
import logging
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import student_swish
import resnet
from util import AverageMeter, accuracy
import time


# Helper function: get [batch_idx, teacher_outputs] list by running teacher model once
def fetch_teacher_outputs(teacher_model, dataloader):
    # set teacher_model to evaluation mode
    teacher_model.eval()
    teacher_outputs = []
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        output_teacher_batch = teacher_model(data_batch).data.cpu().numpy()
        teacher_outputs.append(output_teacher_batch)

    return teacher_outputs

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def train_kd_on_the_fly(args, model, teacher, device, optimizer, dataloader, epoch):
    # set model to training mode
    model.train()
    teacher.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        # compute model output, fetch teacher output, and compute KD loss
        output_batch = model(train_batch)

        # get one batch output from teacher_outputs list
        output_teacher_batch = teacher(train_batch)

        loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, args)
        prec1, prec5 = accuracy(output_batch.data, labels_batch.data, topk=(1, 5))
        # losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], train_batch.size(0))
        top5.update(prec5[0], train_batch.size(0))

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}, Top1:{:.3f}, Top5:{:.3f}'.format(
                epoch, i * len(train_batch), len(dataloader.dataset),
                100. * i / len(dataloader), loss.item(), top1.avg, top5.avg))

def test_kd_on_the_fly(args, model, device, teacher, test_loader):
    model.eval()
    test_loss = 0
    # correct = 0
    top1t = AverageMeter()
    top5t = AverageMeter()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            output_teacher_batch = teacher(data)
            test_loss = loss_fn_kd(output, target, output_teacher_batch, args)

            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            top1t.update(prec1[0], data.size(0))
            top5t.update(prec5[0], data.size(0))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Top1:{:.3f}, Top5:{:.3f}\n'.format(
        test_loss, top1t.avg, top5t.avg))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIIFAR100 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=35, metavar='N',
                        help='number of epochs to train (default: 35)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--prerun', action='store_true', default=True,
                        help='KD disk caching or not')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha for KD')
    parser.add_argument('--temperature', type=float, default= 5,
                        help='temperature for KD')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # print(torch.cuda.is_available())

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    student_model = student_swish.Net().to(device)
    # model.load_state_dict(torch.load("student_relu.pt"))
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum)

    teacher_model = resnet.resnet18().to(device)
    teacher_model.load_state_dict(torch.load("resnet.pt"))

    start_time = time.time()
    if args.prerun:
        # teacher_outputs = fetch_teacher_outputs(teacher_model, train_loader)
        # teacher_outputs_test = fetch_teacher_outputs(teacher_model, test_loader)
        print("Finish teacher outputs:", time.time()-start_time, "s")
        train_start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            train_kd_on_the_fly(args, student_model, teacher_model, device, optimizer, train_loader, epoch)
            train_time = time.time()
            print("Training Time:", train_time-train_start_time)
            test_kd_on_the_fly(args, student_model, device, teacher_model, test_loader)
            print("Testing Time:", time.time()-train_time)
        # if epoch % 10 == 0:
        #     torch.save(student_model.state_dict(),"kd_prerun_05_5.pt")
        # torch.save(student_model.state_dict(),"kd_prerun_05_5.pt")

if __name__ == '__main__':
    main()
