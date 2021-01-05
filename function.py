import numpy as np
import pandas as pd
import re
from PIL import ImageDraw,Image,ImageFont
from retinanet.dataloader import CocoDataset,collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms

def group_bbox(bbox,scores,labels,thred = 5):
    info = pd.DataFrame()
    info['x1'] = np.round(np.array(bbox)[:, 0])
    info['y1'] = np.round(np.array(bbox)[:, 1])
    info['x2'] = np.round(np.array(bbox)[:, 2])
    info['y2'] = np.round(np.array(bbox)[:, 3])
    info['score'] = np.array(scores)
    info['label'] = np.array(labels)


    info = info.sort_values(by=['x1', 'y1', 'x2', 'y2'])
    hist = {}
    groupid = 0
    hist[groupid] = []
    for i in range(len(info) - 1):
        hist[groupid].append(info.index[i])
        c_loc = info.iloc[i, 0:4]
        n_loc = info.iloc[i + 1, 0:4]
        gap = np.mean(abs(n_loc - c_loc))
        if gap > thred:
            # next grouping process
            groupid += 1
            hist[groupid] = []
    hist[groupid].append(info.index[-1])
    group = {}
    for j in range(len(hist)):
        anno = info.loc[hist[j]]
        group[j] = anno.to_dict(orient='list')
    return group

def process_hist(group):
    s = []
    classification = []
    anchors = []
    for i in range(len(group)):
        info = group[i]
        x1s = info['x1']
        y1s = info['y1']
        x2s = info['x2']
        y2s = info['y2']
        scores = info['score']
        labels = info['label']
        x1 = np.mean(x1s)
        y1 = np.mean(y1s)
        x2 = np.mean(x2s)
        y2 = np.mean(y2s)
        label = labels[np.argmax(scores)]
        score = scores[np.argmax(scores)]
        anchors.append((x1,y1,x2,y2))
        s.append(score)
        classification.append(label)
    return np.asarray(s), np.asarray(classification),np.asarray(anchors)

def evaluation_df(anchors_gt, anchors_p, classification_gt, classification_p, thred=0.7):
    '''
    :param anchors_gt: list : groud truth for anchor bbox
    :param anchors_p: prediction for anchor bbox
    :param classification_gt: groud truth for label
    :param classification_p: prediction for label
    :param thred: thred for IOU
    :return: dataframe 1--successful pair -1--missed predition 0--wrong prediction
    '''

    if len(anchors_gt)!=len(classification_gt) or len(anchors_p) != len(classification_p):
        raise ValueError

    result = pd.DataFrame()
    IOU_array = np.zeros((len(anchors_gt), len(anchors_p)))
    for i in range(len(anchors_gt)):
        for j in range(len(anchors_p)):
            IOU_array[i, j] = compute_IOU(anchors_gt[i], anchors_p[j])

    flag = []
    gt_anchor = []
    pred_anchor = []
    gt_label = []
    pred_label = []
    visited = []

    for i in range(len(anchors_gt)):
        idx_max_iou = np.argmax(IOU_array[i, :])
        visited.append(idx_max_iou)
        if IOU_array[i, idx_max_iou] > thred:
            gt_anchor.append(anchors_gt[i,:])
            pred_anchor.append(anchors_p[idx_max_iou,:])
            gt_label.append(classification_gt[i])
            pred_label.append(classification_p[idx_max_iou])
            flag.append(1)
        else:
            flag.append(-1)  # gt bbox missed
            gt_anchor.append(anchors_gt[i,:])
            pred_anchor.append(anchors_p[idx_max_iou,:])
            gt_label.append(classification_gt[i])
            pred_label.append(classification_p[idx_max_iou])

    wrong_proposal = [id for id in range(len(anchors_p)) if id not in visited]
    for idx in wrong_proposal:
        flag.append(0)  # proposal is wrong
        gt_anchor.append([0, 0, 0, 0])
        pred_anchor.append(anchors_p[idx,:])
        gt_label.append(0)
        pred_label.append(classification_p[idx])
    gt_anchor = np.array(gt_anchor)
    pred_anchor = np.array(pred_anchor)
    result['x1_gt'] = gt_anchor[:,0]
    result['y1_gt'] = gt_anchor[:,1]
    result['x2_gt'] = gt_anchor[:,2]
    result['y2_gt'] = gt_anchor[:,3]
    result['x1_pred'] = pred_anchor[:,0]
    result['y1_pred'] = pred_anchor[:,1]
    result['x2_pred'] = pred_anchor[:,2]
    result['y2_pred'] = pred_anchor[:,3]
    result['labels_gt'] = gt_label
    result['labels_pred'] = pred_label
    result['flag'] = flag

    return result

def result_analysis(result, depth_res=0.02 / 256, xy_res =0.01/1024):
    xy_error = []
    z_error = []
    TP_TN = 0
    FN = 0
    FP = 0
    for i in range(len(result)):
        xy_gt = np.array(
            [result.loc[i]['x1_gt'], result.loc[i]['y1_gt'], result.loc[i]['x2_gt'], result.loc[i]['y2_gt']])
        xy_pred = np.array(
            [result.loc[i]['x1_pred'], result.loc[i]['y1_pred'], result.loc[i]['x2_pred'], result.loc[i]['y2_pred']])
        label_gt = result.loc[i]['labels_gt']
        label_pred = result.loc[i]['labels_pred']
        if result.loc[i]['flag'] == 1:
            xy_error.append(np.mean(np.abs(xy_gt - xy_pred)))
            z_error.append(np.abs(label_pred - label_gt))
            TP_TN += 1
        elif result.loc[i]['flag'] == -1:
            FN += 1
        elif result.loc[i]['flag'] == 0:
            FP += 1
        else:
            raise ValueError

    acc = TP_TN / (TP_TN + FN + FP)

    return np.mean(xy_error)*xy_res, np.mean(z_error)*depth_res, acc

def group_refine(bbox,scores,labels,thred = 0.7):
    info = pd.DataFrame()
    info['x1'] = np.round(np.array(bbox)[:, 0])
    info['y1'] = np.round(np.array(bbox)[:, 1])
    info['x2'] = np.round(np.array(bbox)[:, 2])
    info['y2'] = np.round(np.array(bbox)[:, 3])
    info['score'] = np.array(scores)
    info['label'] = np.array(labels)


    IOU_array = np.zeros((len(bbox),len(bbox)))
    for i in range(len(bbox)):
        for j in range(len(bbox)):
            IOU_array[i,j] = compute_IOU(bbox[i],bbox[j])

    group = {}
    visited = []
    groupid = 0
    for rowidx in range(len(bbox)):
        col =IOU_array[rowidx, :]
        colidxs = np.where(col > thred)
        g = [rowidx]
        if rowidx in visited:
            continue
        else:
            visited.append(rowidx)
            for id in colidxs[0]:
                if id in visited:
                    continue
                else:
                    g.append(id)
                    visited.append(id)
            group[groupid] = info.iloc[g].to_dict(orient='list')
            groupid += 1
    return group

def group_bbox_iou(bbox,scores,labels,thred = 0.7):
    info = pd.DataFrame()
    info['x1'] = np.round(np.array(bbox)[:, 0])
    info['y1'] = np.round(np.array(bbox)[:, 1])
    info['x2'] = np.round(np.array(bbox)[:, 2])
    info['y2'] = np.round(np.array(bbox)[:, 3])
    info['score'] = np.array(scores)
    info['label'] = np.array(labels)

    info = info.sort_values(by=['x1', 'y1', 'x2', 'y2'])
    hist = {}
    groupid = 0
    hist[groupid] = []
    for i in range(len(info) - 1):
        hist[groupid].append(info.index[i])
        c_loc = info.iloc[i, 0:4]
        n_loc = info.iloc[i + 1, 0:4]
        # gap = np.mean(abs(n_loc - c_loc))
        iou = compute_IOU(c_loc,n_loc)
        if iou < thred:
            # next grouping process
            groupid += 1
            hist[groupid] = []
    hist[groupid].append(info.index[-1])
    group = {}
    for j in range(len(hist)):
        anno = info.loc[hist[j]]
        group[j] = anno.to_dict(orient='list')
    return group

def compute_IOU(bbox1,bbox2):
    '''

    :param bbox1: [x1,y1,x2,y2]
    :param bbox2: [x1,y1,x2,y2]
    :return: scala value of IOU
    '''
    # compute each area
    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1])

    # compute the sum area
    sum_area = area1 + area2

    # find the each edge of intersect rectangle
    up_left = (max(bbox1[0],bbox2[0]),max(bbox1[1],bbox2[1])) #new x1,y1
    bottom_right = (min(bbox1[2],bbox2[2]),min(bbox1[3],bbox2[3])) #new x2,y2

    if up_left[0] >= bottom_right[0] or up_left[1] >= bottom_right[1]:
        return 0
    else:
        intersect = (bottom_right[0]-up_left[0])*(bottom_right[1]-up_left[1])
        return float(intersect/(sum_area-intersect))

def group_bbox_iou_refine(bbox,scores,labels,thred = 0.7):
    info = pd.DataFrame()
    info['x1'] = np.round(np.array(bbox)[:, 0])
    info['y1'] = np.round(np.array(bbox)[:, 1])
    info['x2'] = np.round(np.array(bbox)[:, 2])
    info['y2'] = np.round(np.array(bbox)[:, 3])
    info['score'] = np.array(scores)
    info['label'] = np.array(labels)
    info['mid_x'] = 0.5 * (info['x2'] - info['x1'])
    info['mid_y'] = 0.5 * (info['y2'] - info['y1'])

    info = info.sort_values(by=['mid_x', 'mid_y'])

    hist = {}
    groupid = 0
    hist[groupid] = []
    for i in range(len(info) - 1):
        hist[groupid].append(info.index[i])
        c_loc = info.iloc[i, 0:4]
        n_loc = info.iloc[i + 1, 0:4]
        # gap = np.mean(abs(n_loc - c_loc))
        iou = compute_IOU(c_loc,n_loc)
        if iou < thred:
            # next grouping process
            groupid += 1
            hist[groupid] = []
    hist[groupid].append(info.index[-1])
    group = {}
    for j in range(len(hist)):
        anno = info.loc[hist[j]]
        group[j] = anno.to_dict(orient='list')
    return group

def show_comparision(file_path, dataloader):
    results = pd.read_csv(file_path)
    pairs=results.loc[results['flag'] == 1]
    bbox_gt = np.array(pairs.iloc[:,0:4])
    bbox_pred = np.array(pairs.iloc[:,4:8])
    label_gt = np.array(pairs.iloc[:,8])
    label_pred = np.array(pairs.iloc[:,9])
    false_1 = results.loc[results['flag'] == -1] # missed
    false_2 = results.loc[results['flag'] == 0] # wrong
    bbox_missed = np.array(false_1.iloc[:, 0:4])
    bbox_false = np.array(false_2.iloc[:,4:8])
    label_missed = np.array(false_1.iloc[:,8])
    label_false = np.array(false_2.iloc[:,9])

    idx = int(re.findall(r'\d+',file_path)[0])
    for i, data in enumerate(dataloader):
        if i != idx:
            continue
        img = np.array(255 * data['img'][0, :, :, :]).copy()
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

    img = visualization(img, bbox_pred, label_pred, color='green')
    img = visualization(img, bbox_false, label_false, color='red')
    img = visualization(img, bbox_missed, label_missed, color='yellow')

    img.show()

    return


def visualization(img,boxes,labels, color = 'red'):
    font = ImageFont.truetype('arial.ttf', 13)
    if isinstance(img,np.ndarray):
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    if len(boxes) != len(labels):
        raise ValueError("BBoxes and labels mismatch")
    for i in range(len(boxes)):
        bbox = boxes[i, :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        label_name = int(labels[i])
        draw.rectangle([x1, y1, x2, y2], outline= color, width=2)
        draw.text((x1, y1), str(label_name), color = color, font=font)
    return img



if __name__=='__main__':
    # test the evaluation dataframe function
    # anchors_gt = np.array([[0,0,1,1],[0,0,2,2],[1,1,2,2]])
    # anchors_p = np.array([[0,0,1,1],[0,0,0,0],[0,0,2,2],[0,0,0,0]])
    # classification_gt = np.ones(3)
    # classification_p = np.ones(4)
    # df = evaluation_df(anchors_gt,anchors_p,classification_gt,classification_p)
    #
    # test the comparison img show function

    # prepare the data
    print('==> Preparing data..')
    dataset = CocoDataset(root_dir='./data', file_name='annotations_test.json',
                          transform=transforms.Compose([Resizer()])
                          )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=1, collate_fn=collater)

    print('==> Test data loaded')
    show_comparision('./experiment/prediction/pred4.csv', dataloader)



