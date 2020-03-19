
# coding: utf-8

# In[1]:


import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
# import json
from matplotlib import cm as CM
import pandas as pd
# from image import *
# from model import CSRNet
# import torchns
# get_ipython().magic('matplotlib inline')


# In[ ]:


#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
#定义高斯卷积核函数
def gaussian_filter_density(gt):
    print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)
    print(distances)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros( gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density


# In[2]:

#now generate the fish_image ground truth

with open('fish.txt','r') as file:
    fish_img=file.read().split('\n')
    print('fish_img:{}'.format(fish_img))


for img_path in fish_img:
    print ('img_path:{}'.format(img_path))
    csv_path=img_path.replace('\f','\\f').replace('.jpg.jpg','.csv').replace('fish2019.9.16','fish_annotation')
    print('csv_path:', csv_path)
    gt_Dataframe =pd.read_csv(csv_path)
    print('gt_Dataframe: {}'.format(gt_Dataframe))
    gt = gt_Dataframe.values.tolist()
    print('gt:{}'.format(gt))
    # gt = io.loadmat(img_path.replace('.jpg.jpg','.csv').replace('fish2019.9.16','fish_annotation'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter_density(k)
    h5_path=img_path.replace('\f','\\f').replace('.jpg.jpg','.h5').replace('fish2019.9.16','ground_truth')
    with h5py.File(h5_path, 'w') as hf:
        hf['density'] = k


# In[ ]:


#now see a sample from Fish_count
plt.imshow(Image.open(fish_img[0]))


# In[ ]:


gt_file = h5py.File(h5_path[0],'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
plt.show()

# In[ ]:


print('real_number: {0}'.format(np.sum(groundtruth)))# don't mind this slight variation
