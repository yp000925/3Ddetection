from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, num_classes=256):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,classes+1]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,#classes]

        # p = x.sigmoid()
        p = x
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def sigmoid_focal_loss(self, inputs, targets, reduction= "sum", gamma=2, alpha=0.25):
        t = one_hot_embedding(targets.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        if torch.cuda.is_available():
            t = Variable(t).cuda()
        else:
            t = Variable(t)
        p = torch.sigmoid(inputs)

        ce_loss = F.binary_cross_entropy_with_logits(inputs, t, reduction="none")# according to the official documentation, this operation contains sigmoid inside
        p_t = p * t + (1 - p) * (1 - t)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * t + (1 - alpha) * (1 - t)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return  loss


    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        if torch.cuda.is_available():
            t = Variable(t).cuda()
        else:
            t = Variable(t)

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''
        Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N, #anchors]
        num_pos = pos.data.long().sum() # calculate all the value larger than 0


        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N, #anchors, 4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)      # [#pos, 4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos, 4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        # print('loc_loss:%.3f'% (loc_loss.data.item()))

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################

        pos_neg = cls_targets > -1  # [N, #anchors] # only exclude ignored anchors

        num_peg = pos_neg.data.long().sum()

        mask = pos_neg.unsqueeze(2).expand_as(cls_preds) # [N, #anchors, #classes]
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)# [#pos_neg, #classes]
        masked_cls_targets = cls_targets[pos_neg] # [#pos_neg]
        # print('masked_cls_preds size: %s' % (str(masked_cls_preds.shape)))
        # print('masked_cls_targets size: %s'%(str(masked_cls_targets.shape)))
        # cls_loss = self.focal_loss_alt(masked_cls_preds, masked_cls_targets)
        cls_loss = self.sigmoid_focal_loss(masked_cls_preds,masked_cls_targets)
        # print('cls_loss:%.3f' % (cls_loss.data.item()))
        # num_pos = max(1.0, num_pos.item())

        overlapped_anchors = cls_targets == -1
        overlapped_anchors_number =overlapped_anchors.data.long().sum()

        print('|Obj|Background|Overlapped anchors : %d| %d| %d' % (num_pos,num_peg-num_pos,overlapped_anchors_number))
        # print(' background+obj anchors : %d/ %d' % (num_peg, num_boxes * batch_size))
        # print(' overlapped anchors : %d / %d'% (overlapped_anchors_number,num_boxes*batch_size))

        # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data.item()/num_pos, cls_loss.data.item()/num_pos), end=' | ')
        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data.item()/num_pos.type(torch.float),
                                                   cls_loss.data.item()/num_peg.type(torch.float)), end=' | ')
        # loss = loc_loss/num_pos + cls_loss/num_peg
        loss = (loc_loss+cls_loss)/num_pos #since the loss came from the large volume of background should be negligible
        return loss

if __name__ == '__main__':
    loc_preds = torch.ones([2,3,4])
    loc_targets = torch.ones([2,3,4])
    cls_preds = torch.ones([2,3,256])*(-100)
    cls_preds[:,:,0] = 1
    cls_targets = torch.ones([2,3])
    criterion = FocalLoss()
    loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
