'''Load image/labels/boxes from an annotation file.

The annotations.json is like:
'images': [
    {
        'file_name': '0.jpg',
        'height': 1024,
        'width': 1024,
        'id': 0
    },
    ...
],
'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # if you have mask labels
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
    },
    ...
],

'categories': [
    {'id': 0, 'name': '0'},
 ]
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image,ImageDraw
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop
import json

class CoCoDataset(data.Dataset):
    def __init__(self,root, json_file, train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''

        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        with open(json_file) as jf:
            f = json.load(jf)
            images = f['images']
            annos = f['annotations']
            categories = f['categories']
            self.num_samples = len(images)

        for image in images:
            img_id = image['id']
            self.fnames.append(image['file_name'])
            box = []
            label = []
            for anno in annos:
                if anno['image_id'] != img_id:
                    continue
                [bbox_x, bbox_y, width, height] = anno['bbox']
                box.append([float(bbox_x), float(bbox_y), float(bbox_x+width), float(bbox_y+height)])
                label.append(int(anno['category_id']))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.Tensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load images and boxes
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation
        if self.train:
            img, boxes = random_flip(img,boxes)
            # img, boxes = random_crop(img,boxes)
            img, boxes = resize(img, boxes, (size, size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size, size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''

        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            # loc_target, cls_target = (boxes[i], labels[i]) #just for checking
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    import torchvision
    import numpy as np
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Converts a PIL Image or numpy.ndarray (H x W x C) in
        # the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    dataset = CoCoDataset(root='/Users/zhangyunping/PycharmProjects/Holo_synthetic/hologram',
                          json_file='/Users/zhangyunping/PycharmProjects/Holo_synthetic/annotations/annotations.json',
                          train=True, transform=transform, input_size=1024)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=1, collate_fn=dataset.collate_fn)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for images, loc_targets, cls_targets in dataloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break


    ## check for whether the bbox is correctly loaded  NOTE: the encoder in dataloader should be commented when do the
    ## checking work

    # for images, loc_targets, cls_targets in dataloader:
    #     images = images.numpy()[0].transpose(1,2,0)
    #     images = images*255
    #     image = Image.fromarray(images.astype(np.uint8))
    #     draw = ImageDraw.Draw(image)
    #     loc_target = loc_targets.numpy()[0]
    #     cls_target = cls_targets.numpy()[0]
    #     for bbox in loc_target:
    #         x,y,x_,y_ = bbox
    #         draw.rectangle([int(x), int(y), int(x_), int(y_)], width=2, outline='yellow')
    #     image.show()
    #     break








