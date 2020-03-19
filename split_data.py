# 划分数据集
from sklearn.model_selection import train_test_split
import os
import glob

# root=r'D:\Postgraudate\learning\experiment\fish\fish_data1'
# root ='/root/miaozheng/fish/fish_data1'
root='/content/drive/My Drive/fish_counting/fish_data1'
fish_path= os.path.join(root,'fish_process')

#读入数据集图片
data_list=[]

for path in glob.glob(os.path.join(fish_path,'*.jpg')):
    print(path)
    data_list.append(path)

#读入数据集的标签
label_list=[]
for path in glob.glob(os.path.join(fish_path.replace('fish_process','csv_modified'),'*.csv')):
    print(path)
    label_list.append(path)

#划分数据集
fish_train,fish_test1,label_train,label_test1= train_test_split(data_list,label_list,test_size=0.4,random_state=0)
fish_val,fish_test,label_val,label_test= train_test_split(fish_test1,label_test1,test_size=0.5,random_state=0)
with open('fish_train.txt','w') as outfile:
    outfile.write(','.join(fish_train))
with open('fish_val.txt','w') as outfile:
    outfile.write(','.join(fish_val))
with open('fish_test.txt','w') as outfile:
    outfile.write(','.join(fish_test))