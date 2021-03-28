'''
evaluate the network and write each into the csv file
then compare them to calculate the xy-error,z-error and the accuracy

'''

import os
import argparse
import torch

import torchvision.transforms as transforms

import numpy as np

from retinanet import model
from retinanet import coco_eval
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from collections import OrderedDict
from PIL import Image,ImageDraw,ImageFont
import pandas as pd
from function import group_refine,process_hist,compute_IOU,evaluation_df ,result_analysis


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
RESUME_Epoch = 15
print('==> Resuming from checkpoint%d'%(RESUME_Epoch))
checkpoint = torch.load('./experiment/checkpoint/net_epoch%d.pt'%(RESUME_Epoch),map_location=torch.device('cpu'))
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
xy_errors = []
z_errors=[]
accs = []

for idx, data in enumerate(dataloader):

    with torch.no_grad():
        if torch.cuda.is_available():
            scores, classification, transformed_anchors = net(data['img'].cuda().float())
        else:
            scores, classification, transformed_anchors = net(data['img'].float())

        idxs = np.where(scores.cpu() > 0.2)

        scores_ = scores[idxs]
        classification_ = classification[idxs]
        transformed_anchors_ = transformed_anchors[idxs]
        Group_Refined = group_refine(transformed_anchors_, scores_, classification_, thred=0.7)
        scores_p, classification_p, anchors_p = process_hist(Group_Refined) #final prediction after post-process

        # get the ground truth
        anchors_gt, classification_gt = data['annot'][0, :, 0:4], data['annot'][0, :, 4]

        match_result = evaluation_df(np.array(anchors_gt), np.array(anchors_p), np.array(classification_gt), np.array(classification_p))
        match_result.to_csv("experiment/prediction/pred%d.csv"%(idx),index=False)

        xy_error, z_error, acc = result_analysis(match_result)
        xy_errors.append(xy_error)
        z_errors.append(z_error)
        accs.append(acc)

        # print(idx,len(dataloader))



