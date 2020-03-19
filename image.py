import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('fish_process','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if False:
        crop_size = (img.size[0]/2,img.size[1]/2)

        if random.randint(0,9)<= -1:
            # 这种情况不会发生
            # randint(a,b)产生[a,b]之间的随机整数
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            # random.random()产生 0 到 1 之间的随机浮点数
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)

        # im1 = im.crop((left, top, right, bottom))
        # img.crop()按矩形框来裁剪图像
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 缩小缩放时通常采用cv2.INTER_CUBIC插值
    # 缩小以后为什么要乘以64
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    # target = cv2.resize(target, (int(target.shape[1] / 4), int(target.shape[0] / 4)),
    #                     interpolation=cv2.INTER_CUBIC)*16
    
    return img,target

