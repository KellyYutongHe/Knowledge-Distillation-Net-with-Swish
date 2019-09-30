"""
Classification main file
"""
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from util import AverageMeter, accuracy, save_checkpoint
from data_util import CarDataset
from model import res_car


def train_one_epoch(model, train_loader, optimizer, criterion, print_freq=100):
    """
    Args:
        model (torch): Torch model
        train_loader (torch): Data generator
        optimizer (torch): Optimizer for model
        criterion (torch): Loss functions
        print_freq (int) : print frequency
    """
    # model on training mode
    model.train()

    # set trackable meter
    losses = AverageMeter()
    Accuracy = AverageMeter()

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate accuracy metric
        c_pred = outputs.data.cpu().numpy().argmax(axis=1)
        c_true = labels.data.cpu().numpy()
        accurate = (np.sum(c_pred == c_true, dtype=float) / c_true.shape)[0]
        Accuracy.update(accurate, inputs.size(0))

        # print loss
        if i % print_freq == 0:
            print(
                '===> Train:({}/{}):\t'
                'Loss {:.4f}\t'
                'Acc {:.4f}'.format(
                    i,
                    len(train_loader),
                    losses.avg,
                    Accuracy.avg))


def eval_one_epoch(model, val_loader, criterion):
    """
    Args:
        model (torch): Torch model
        val_loader (torch): Data generator
        criterion (torch): Loss functions
        print_freq (int) : print frequency

    Returns:
        losses.avg (float): average loss
        accuracy.avg (float): average accuracy
        gt (numpy ndarray)
        pred
    """
    # model on training mode
    model.eval()

    # set trackable meter
    losses = AverageMeter()
    Accuracy = AverageMeter()

    # store prediction and ground truth
    gt = np.array([])
    pred = np.array([])

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            # calculate accuracy metric
            c_pred = outputs.data.cpu().numpy().argmax(axis=1)
            c_true = labels.data.cpu().numpy()
            accurate = (np.sum(c_pred == c_true, dtype=float) / c_true.shape)[0]
            Accuracy.update(accurate, inputs.size(0))

            gt = np.append(gt, c_true)
            pred = np.append(pred, c_pred)

    # print loss
    print(
        '===> evaluate:\t'
        'Loss {:.4f}\t'
        'Acc {:.4f}'.format(
            losses.avg,
            Accuracy.avg))

    return losses.avg, Accuracy.avg, gt, pred


if __name__ == '__main__':
    """
    main func
    """
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=50, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--output', default=False, type=bool, help='whether to output the prediction and ground truth')
    parser.add_argument('--save', default=True, type=bool, help='whether to save model')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH', help='path to pretrain model')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='if evaluate only')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()
    print(args)

    # Evaluation metric
    best_loss = float('Inf')
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    # Hyperparameters
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    print_freq = args.print_freq

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_set = CarDataset(
        csv_file='../../car_data/train.csv',
        transform=transforms.Compose([
            # transforms.Resize((256, 256)),
            # transforms.RandomCrop((224, 224)),
            transforms.Resize((128, 128)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    eval_set = CarDataset(
        csv_file='../../car_data/test.csv',
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    # eval_set = ImageFolder(
    #     root='../../car_data/result',
    #     transform=transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]))

    eval_loader = DataLoader(
        eval_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load model
    model = res_car(num_classes=296).cuda()

    # Loss & Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    # Load pretrain model
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrained_dict = torch.load(args.pretrain)['model_state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print(("=> no pretrained file found at '{}'".format(args.pretrain)))

    # Training and Evaluation
    if not args.evaluate:
        for e in range(epoch):
            print('{}/{} epoch starts'.format(e, epoch))
            # scheduler.step()
            train_one_epoch(model, train_loader, optimizer, criterion, print_freq=print_freq)
            loss, accuracy, gt, pred = eval_one_epoch(model, eval_loader, criterion)

            # advanced metric
            precision = precision_score(gt, pred, average='micro')
            recall = recall_score(gt, pred, average='micro')
            f1 = f1_score(gt, pred, average='micro')

            # remember the best loss and save checkpoint
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if is_best:
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall
                best_f1 = f1

            if args.save is True:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                    }, is_best=is_best, filename='./saved_models/weights_{}epoch.pth.tar'.format(e))

        print(
            'The best loss is {:.4f}\t'
            'The best accuracy is {:.4f}\t'
            'The best precision is {:.4f}\t'
            'The best recall is {:.4f}\t'
            'The best f1 is {:.4f}'.format(best_loss, best_accuracy, best_precision, best_recall, best_f1))

    # Evaluatioin only
    else:
        loss, accuracy, gt, pred = eval_one_epoch(model, eval_loader, criterion)

        # advanced metric
        precision = precision_score(gt, pred, average='micro')
        recall = recall_score(gt, pred, average='micro')
        f1 = f1_score(gt, pred, average='micro')

        if args.output is True:
            np.savez('./data/result.npz', gt=gt, pred=pred)

        print(
            'The loss is {:.4f}\t'
            'The accuracy is {:.4f}\t'
            'The precision is {:.4f}\t'
            'The recall is {:.4f}\t'
            'The f1 is {:.4f}'.format(loss, accuracy, precision, recall, f1))
