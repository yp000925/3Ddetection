import os
import argparse
import torch

import torchvision.transforms as transforms

import numpy as np

from retinanet import model
from retinanet import coco_eval
from retinanet.dataloader import CocoDataset,collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from collections import OrderedDict
from PIL import Image,ImageDraw,ImageFont
import pandas as pd
from function import group_bbox_iou, group_refine,visualization,group_bbox,process_hist



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
RESUME_Epoch = 10
print('==> Resuming from checkpoint%d'%(RESUME_Epoch))
checkpoint = torch.load('./experiment/checkpoint/net_epoch%d.pt'%(RESUME_Epoch), map_location=torch.device('cpu'))
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
        if idx != 12:
            continue

        if torch.cuda.is_available():
            scores, classification, transformed_anchors = net(data['img'].cuda().float())
        else:
            scores, classification, transformed_anchors = net(data['img'].float())

        idxs = np.where(scores.cpu() > 0.5)
        img = np.array(255 * data['img'][0, :, :, :]).copy()
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        # img = Image.fromarray(img)
        # draw = ImageDraw.Draw(img)
        scores_ = scores[idxs]
        classification_ = classification[idxs]
        transformed_anchors_ = transformed_anchors[idxs]

        # group = group_bbox_iou(transformed_anchors_, scores_, classification_, thred=0.6)
        # scores_p, classification_p, anchors_p = process_hist(group)
        # group_Refine = group_refine(anchors_p, scores_p, classification_p, thred=0.7)

        group_Refine = group_refine(transformed_anchors_, scores_, classification_, thred=0.7)
        scores_p2, classification_p2, anchors_p2 = process_hist(group_Refine)

        # img1 = visualization(img, anchors_p, classification_p,color='red')
        # img1.show()
        img3 = visualization(img, anchors_p2, classification_p2, color='green')
        img3.show()

        img2 = visualization(img, data['annot'][0, :, 0:4], data['annot'][0, :, 4], color='yellow')
        img2.show()
        if idx == 12:
            break





        # for j in range(len(scores_p)):
        #     bbox = anchors_p[j, :]
        #     x1 = int(bbox[0])
        #     y1 = int(bbox[1])
        #     x2 = int(bbox[2])
        #     y2 = int(bbox[3])
        #     label_name = classification_p[j]
        #     draw.rectangle([x1, y1, x2, y2], outline='red')
        #     draw.text((x1, y1), str(label_name), (0, 255, 255), font=font)
        #
        # for i in range(data['annot'].shape[1]):
        #     bbox = data['annot'][0,i,0:4]
        #     x1 = int(bbox[0])
        #     y1 = int(bbox[1])
        #     x2 = int(bbox[2])
        #     y2 = int(bbox[3])
        #     label_name = int(data['annot'][0,i,4])
        #     draw.rectangle([x1, y1, x2, y2], outline='yellow')
        #     draw.text((x2, y2), str(label_name), (255, 255, 0), font=font)
        # img.show()









