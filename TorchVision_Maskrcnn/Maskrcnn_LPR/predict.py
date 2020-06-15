import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import sys
sys.path.append("./detection")
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import cv2
import cv2_util

import random
import time
import datetime

def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
 
    return (b,g,r)


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256

def PredictImg( image, model,device):
    #img, _ = dataset_test[0] 
    img = cv2.imread(image)
    result = img.copy()
    dst=img.copy()
    img=toTensor(img)

    names = {'0': 'background', '1': 'car'}
    # put the model in evaluati
    # on mode

    prediction = model([img.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']

    m_bOK=False;
    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.8:
            m_bOK=True
            color=random_color()
            mask=masks[idx, 0].mul(255).byte().cpu().numpy()
            thresh = mask
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(dst, contours, -1, color, -1)

            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(result,(x1,y1),(x2,y2),color,thickness=2)
            cv2.putText(result, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)

            dst1=cv2.addWeighted(result,0.7,dst,0.5,0)

            
    if m_bOK:
        cv2.imshow('result',dst1)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,num_classes=num_classes)
    model.to(device)
    model.eval()
    save = torch.load('model.pth')
    model.load_state_dict(save['model'])
    start_time = time.time()
    PredictImg('test/3.jpg',model,device)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(total_time)