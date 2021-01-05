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
from PIL import Image,ImageDraw,ImageFont

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Visualization')
parser.add_argument('--depth', default=18, type=int, help='model depth')

args = parser.parse_args()

#prepare the data
print('==> Preparing data..')
dataset = CocoDataset(root_dir='./data', file_name='annotations_test.json',
                            transform=transforms.Compose([Resizer()])
                            )
dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=1, collate_fn=collater)

print('==> Test data loaded')

# Model
if args.depth == 18:
    net = model.resnet18(num_classes=dataset.num_classes(), pretrained=False)
elif args.depth == 34:
    net = model.resnet34(num_classes=dataset.num_classes(), pretrained=False)
elif args.depth == 50:
    net = model.resnet50(num_classes=dataset.num_classes(), pretrained=False)
elif args.depth == 101:
    net = model.resnet101(num_classes=dataset.num_classes(), pretrained=False)
elif args.depth == 152:
    net = model.resnet152(num_classes=dataset.num_classes(), pretrained=False)
else:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')


#resume from the checkpoint
RESUME_Epoch = 2
print('==> Resuming from checkpoint%d'%(RESUME_Epoch))
checkpoint = torch.load('./checkpoint/net_epoch%d.pt'%(RESUME_Epoch),map_location=torch.device('cpu'))
state_dict = checkpoint['net']
new_state_dict = OrderedDict([(key.split('module.')[-1], state_dict[key]) for key in state_dict])
net.load_state_dict(new_state_dict)
start_epoch = checkpoint['epoch']
print('loaded')

if torch.cuda.is_available():
    net = net.cuda()
    net = torch.nn.DataParallel(net).cuda()
else:
    net = torch.nn.DataParallel(net)

net.eval()

font = ImageFont.truetype('arial.ttf', 10)

for idx, data in enumerate(dataloader):
    with torch.no_grad():
        if torch.cuda.is_available():
            scores, classification, transformed_anchors = net(data['img'].cuda().float())
        else:
            scores, classification, transformed_anchors = net(data['img'].float())

        idxs = np.where(scores.cpu() > 0.2)
        img = np.array(255 * data['img'][0, :, :, :]).copy()
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset.labels[int(classification[idxs[0][j]])]
            draw.rectangle([x1, y1, x2, y2], outline='red')
            draw.text((x1, y1), label_name, (0, 255, 255), font=font)
        img.show()
    if idx == 10:
        break








