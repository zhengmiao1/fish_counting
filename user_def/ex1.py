#
# img_path= 'E:\Postgraduate\data\fish_data1\fish2019.9.16\IMG_0001.jpg.jpg'
# img_path1=img_path.replace('\f','\\f').replace('.jpg.jpg', '.h5').replace('fish2019.9.16', 'ground_truth')
# print('img_path1:{}'.format(img_path1))
# img_path2= img_path.replace('.jpg.jpg', '.h5').replace('fish2019.9.16', 'ground_truth').replace('\f','\\f')
# print('img_path2:{}'.format(img_path2))
# # with h5py.File(img_path1, 'w') as hf:
# #     hf['density'] = 1
# import h5py
# f=h5py.File(img_path2)
# f.keys()
# # for key in f.keys():
# #     print(f[key].name)
# #     print(f[key].shape)
# #     print(f[key].value)

# file=open('fish.txt')
# print(file.read())

with open('fish_train.txt', 'r') as outfile:
    train_list = outfile.read().split()
    print(train_list)