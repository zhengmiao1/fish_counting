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
from image import *
# from model import CSRNet
# import torchns
# get_ipython().magic('matplotlib inline')

with open('fish1.txt','r') as file:
    fish_img=file.read().split('\n')
print(fish_img[0])

# In[ ]:
# img=plt.imread(fish_img[0])
# print(img)
# plt.imshow(img)


# plt.imshow(Image.open(fish_img[0]))

# In[ ]:
gt_file = h5py.File(fish_img[0].replace('.jpg.jpg','.h5').replace('fish2019.9.16','ground_truth'),'r')
print(gt_file)
groundtruth = np.asarray(gt_file['density'])
# plt.imshow(fish_img[0])
plt.imshow(groundtruth,cmap=CM.jet)
plt.show()

# img_add = cv2.addWeighted(fish_img[0], 0.3, groundtruth, 0.7, 0)
# plt.show(img_add)
# plt.show()