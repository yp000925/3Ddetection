import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from retinanet import model
import time
from torch.autograd import Variable
import lbtoolbox as lb
from signal import SIGINT, SIGTERM
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
from retinanet import model
from retinanet import coco_eval
from retinanet.dataloader import CocoDataset,collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
from retinanet.utils import format_time
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', default=2, type=int, help='the training batch size')
parser.add_argument('--n_epochs', default=15, type=int, help='training epoches')
parser.add_argument('--depth', default=18, type=int, help='model depth')

args = parser.parse_args()

if not os.path.isdir('summary'):
    os.mkdir('summary')
writer = SummaryWriter('./summary/log')


# assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
trainset = CocoDataset(root_dir='./data', file_name='annotations_train.json',
                            transform=transforms.Compose([Resizer()])
                            )
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=1, collate_fn=collater)


testset = CocoDataset(root_dir='./data', file_name='annotations_test.json',
                            transform=transforms.Compose([Resizer()])
                            )
testloader = DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=1, collate_fn=collater)

# Model
if args.depth == 18:
    net = model.resnet18(num_classes=trainset.num_classes(), pretrained=False)
elif args.depth == 34:
    net = model.resnet34(num_classes=trainset.num_classes(), pretrained=False)
elif args.depth == 50:
    net = model.resnet50(num_classes=trainset.num_classes(), pretrained=False)
elif args.depth == 101:
    net = model.resnet101(num_classes=trainset.num_classes(), pretrained=False)
elif args.depth == 152:
    net = model.resnet152(num_classes=trainset.num_classes(), pretrained=False)
else:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

# net.load_state_dict(torch.load('./model/net.pth'))

# args.resume = 1
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/net_epoch0.pt')
    state_dict = checkpoint['net']
    new_state_dict = OrderedDict([(key.split('module.')[-1],state_dict[key]) for key in state_dict])
    net.load_state_dict(new_state_dict)
    start_epoch = checkpoint['epoch']
    print('loaded')
use_gpu = True

if use_gpu:
    if torch.cuda.is_available():
        net = net.cuda()

if torch.cuda.is_available():
    net = torch.nn.DataParallel(net).cuda()
else:
    net = torch.nn.DataParallel(net)


# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

train_losses = []
valid_losses = []
InterrutedFlag = False



# Training
def train(epoch):
    global train_losses
    global InterrutedFlag

    print('\nEpoch: %d' % epoch)
    print('==================================================================')
    net.train()
    # net.module.freeze_bn()

    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:

        for iter_num, data in enumerate(trainloader):

            try:
                t1 = time.time()
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = net([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    classification_loss, regression_loss = net([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)

                optimizer.step()

                train_losses.append(float(loss))

                t2 = time.time()
                time_elp = t2 - t1
                total_time = time_elp * len(trainloader)
                time_pass = (iter_num + 1) * time_elp

                print('\rIteration: {}/{} | Classification loss: {:1.5f} | '
                      'Regression loss: {:1.5f} | Running loss average: {:1.5f}     Time : %s / %s in in epoch'.format(
                    iter_num, len(trainloader), float(classification_loss), float(regression_loss), np.mean(train_losses),
                    format_time(time_pass), format_time(total_time)), end='', flush=True)

                writer.add_scalar('iteration_train_loss', float(loss), iter_num + epoch * len(trainloader))

                del classification_loss
                del regression_loss

                if u.interrupted:
                    print("Interrupted on request!")
                    InterrutedFlag = True
                    break

            except Exception as e:
                print(e)
                continue

    if  u.interrupted:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'avg_loss': np.mean(train_losses),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_interrupted.pt')
        print('\nSaved')



def eval(epoch_num):
    global valid_losses
    print('\nValidation==================================================================')
    net.train()
    test_loss = 0
    with torch.no_grad():
        for iter_num, data in enumerate(testloader):
            print('Batch: %3d/50' % (iter_num))
            if torch.cuda.is_available():
                classification_loss, regression_loss = net([data['img'].cuda().float(), data['annot']])
            else:
                classification_loss, regression_loss = net([data['img'].float(), data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            test_loss += float(loss)
            # for quick check only use 50*batch_size samples
            if iter_num == 50:
                break

    valid_losses.append(test_loss/50)
    print('validation loss at epoch {:d} is {:.4f}'.format(epoch_num,test_loss))



def test(epoch):
    print('Evaluating dataset')
    coco_eval.evaluate_coco(testset, net)


avg_train_losses = []
avg_valid_losses = []
early_stopping = EarlyStopping(patience=3, verbose=True)



for epoch in range(start_epoch, start_epoch+args.n_epochs):
    train(epoch)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join('./checkpoint', 'net_epoch{}.pt'.format(epoch)))
    # test(epoch)
    eval(epoch)
    writer.add_scalar('train_loss_per_epoch', np.average(train_losses), epoch)
    writer.add_scalar('validation_loss_per_epoch', np.average(valid_losses), epoch)

    if InterrutedFlag:
        print("Interupt ")
        break