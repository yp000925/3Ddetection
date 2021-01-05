import os
import argparse
from encoder import DataEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import CoCoDataset
import time
from torch.autograd import Variable
from utils import format_time
import lbtoolbox as lb
from signal import SIGINT, SIGTERM
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', default=2, type=int, help='the training batch size')
parser.add_argument('--n_epochs',default=15,type=int,help='training epoches')
args = parser.parse_args()

if not os.path.isdir('summary'):
    os.mkdir('summary')
writer = SummaryWriter('./summary/log')


# assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = CoCoDataset(root='./data/hologram',
                       json_file='./data/annotations/annotations_train.json',
                       train=True, transform=transform, input_size=1024)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=1, collate_fn=trainset.collate_fn)

testset = CoCoDataset(root='./data/hologram',
                      json_file='./data/annotations/annotations_val.json',
                      train=False, transform=transform, input_size=1024)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=1, collate_fn=testset.collate_fn)

# Model
net = RetinaNet()
# net.load_state_dict(torch.load('./model/net.pth'))

# args.resume = 1
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt_interrupted.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
if torch.cuda.is_available():
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

criterion = FocalLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(),lr=args.lr)

train_losses = []
valid_losses = []

# Training
def train(epoch):
    global train_losses
    print('\nEpoch: %d' % epoch)
    print('==================================================================')
    net.train()
    # iter_p_epoch = int(len(trainloader)/ args.batch_size)
    # net.module.freeze_bn()
    train_loss = 0

    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:

        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
            print('Iter: %4d / %d ' % (batch_idx+1, len(trainloader)))
            t1 = time.time()
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                loc_targets = Variable(loc_targets.cuda())
                cls_targets = Variable(cls_targets.cuda())
            else:
                inputs = Variable(inputs)
                loc_targets = Variable(loc_targets)
                cls_targets = Variable(cls_targets)
            optimizer.zero_grad()
            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            t2 = time.time()
            time_elp = t2-t1
            total_time = time_elp*len(trainloader)
            time_pass = (batch_idx+1)*time_elp
            print('train_loss: %.3f | avg_loss: %.3f' % (loss.data.item(), train_loss/(batch_idx+1)))
            print('Time : %s / %s' % (format_time(time_pass), format_time(total_time)))
            print('\n')
            train_losses.append(loss.detach())
            writer.add_scalar('train_loss_per_batch',loss.data.item(),batch_idx+epoch*len(trainloader))

            if u.interrupted:
                print("Interrupted on request!")
                break

    if  u.interrupted:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': train_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_interrupted.pt')
        print('\nSaved')


# Test
def test(epoch):
    global valid_losses
    print('\nTest==================================================================')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        print('Batch: %3d/%3d' % (batch_idx, int(len(testloader)/args.batch_size)))
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda(), volatile=True)
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())
        else:
            inputs = Variable(inputs, volatile=True)
            loc_targets = Variable(loc_targets)
            cls_targets = Variable(cls_targets)

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.item()
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.item(), test_loss/(batch_idx+1)))
    valid_losses.append(test_loss/len(testloader))

    # # Save checkpoint
    # global best_loss
    # test_loss /= len(testloader)
    # if test_loss < best_loss:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'loss': test_loss,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_loss = test_loss
    #     print('\nSavd')

avg_train_losses = []
avg_valid_losses = []
early_stopping = EarlyStopping(patience=3, verbose=True)

for epoch in range(start_epoch,start_epoch+args.n_epochs):
    train(epoch)
    test(epoch)
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    epoch_len = len(str(args.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{args.n_epochs:>{epoch_len}}] ' +
             f'train_loss: {train_loss:.5f} ' +
             f'valid_loss: {valid_loss:.5f}')
    print(print_msg)
    writer.add_scalar('train_loss_per_epoch', train_loss, epoch)
    writer.add_scalar('val_loss_per_epoch', valid_loss, epoch)
    train_losses = []
    valid_losses = []
    early_stopping(valid_loss, net, epoch)

    if early_stopping.early_stop:
        print("Early stopping")
        break

