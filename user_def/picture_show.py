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

with open('fish_train.txt','r') as file:
    fish_img=file.read().split('\n')

plt.imshow(Image.open(fish_img[0]))

# In[ ]:

plt.imshow(Image.open(fish_img[0]))


# In[ ]:


gt_file = h5py.File(fish_img[0].replace('.jpg.jpg','.h5').replace('mark','ground_truth'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
plt.show()