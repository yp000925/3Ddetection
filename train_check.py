import os
import argparse
from encoder import DataEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
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

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', default=1, type=int, help='the training batch size')
args = parser.parse_args()


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

args.resume = 1
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

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    iter_p_epoch = int(len(trainloader)/ args.batch_size)
    # net.module.freeze_bn()
    train_loss = 0

    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:

        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
            print('Iter: %4d / %d ' % (batch_idx+1, iter_p_epoch))
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
            total_time = time_elp*iter_p_epoch
            time_pass = (batch_idx+1)*time_elp
            print('train_loss: %.3f | avg_loss: %.3f' % (loss.data.item(), train_loss/(batch_idx+1)))
            print('Time : %s / %s' % (format_time(time_pass), format_time(total_time)))

            if u.interrupted:
                print("Interrupted on request!")
                break

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'loss': train_loss,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_interrupted.pth')
    print('\nSavd')


# Test
def test(epoch):
    print('\nTest')
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

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss
        print('\nSavd')


def visualization():
    import numpy as np
    from PIL import Image,ImageDraw
    print('visulization\n')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        # inputs = Variable(inputs.cuda(), volatile=True)
        # loc_targets = Variable(loc_targets.cuda())
        # cls_targets = Variable(cls_targets.cuda())
        print('Batch: %3d/%3d' % (batch_idx+1, int(len(testloader)/args.batch_size)))
        inputs = Variable(inputs,volatile=True)
        loc_targets = Variable(loc_targets)
        cls_targets = Variable(cls_targets)
        # loc_preds, cls_preds = net(inputs)
        cls_preds = torch.rand([loc_targets.shape[1],256], dtype=torch.float)
        encoder = DataEncoder()
        boxes, labels = encoder.decode(loc_targets.squeeze(), cls_preds, (1024, 1024))
        images = inputs.numpy()[0].transpose(1, 2, 0)
        images = images*255
        image = Image.fromarray(images.astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for bbox in boxes.numpy():
            x,y,x_,y_ = bbox
            draw.rectangle([int(x), int(y), int(x_), int(y_)], width=2, outline='yellow')
        image.show()
        break

for epoch in range(start_epoch, start_epoch+15):
    # visualization()
    train(epoch)
    test(epoch)
