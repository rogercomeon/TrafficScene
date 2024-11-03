import os
import torch
import numpy as np
import imageio
import scipy.io
from torch.utils import data
import random
from  glob import glob
import fnmatch
from PIL import Image
import re
import cv2

class LFdatasetLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        augmentations=None,
        test_mode=False,
        model_name=None
    ):
        self.root = root       
        self.split = split     
        self.augmentations = augmentations  
        self.test_mode=test_mode   
        self.model_name=model_name  
        self.n_classes = 15
        # self.files = [] 
        self.files = []
        if self.split in ['train','val','test']: #
            self.image_dir = os.path.join(self.root,  self.split)
            self.label_dir = os.path.join(self.root, self.split)
        for img_class in os.listdir(self.image_dir):
            img_all =[]
            class_name = os.path.join(self.image_dir,img_class)
            if(os.path.isdir(class_name)):
                img_all = glob(class_name+'/*_*.png')
                # img_all.remove(class_name+'/label.png')
                oneimg_dict={}
                oneimg_dict["image"] = class_name
                oneimg_dict["label"] = glob(class_name+'/2_2_label.png')[0]
                self.files.append(oneimg_dict)


    def __len__(self):     
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):  
        """__getitem__

        :param index:
        """

        img_all=[]
        labels=[]
        data_dict = self.files[index]
        img_list = data_dict["image"]
        # print(img_list)
        label_path = data_dict["label"]

        label_map = {
            0: 255, 1: 255, 2: 0, 3: 255, 4: 1, 5: 2, 6: 3,
            7: 255, 8: 255, 9: 4, 10: 5, 11: 255, 12: 6, 13: 7, 14: 8, 15: 9,
            16: 10, 17: 11, 18: 12, 19: 255, 20: 13, 21: 14
        }

        dict_file ={0:['1_2','2_2','3_2']
                    ,1:['2_1','2_2','2_3']
                    ,2:['1_1','2_2','3_3']
            ,3:['1_3','2_3','3_1']}
        
        label = cv2.imread(label_path,0)
        # label= np.array(label,dtype=np.uint8)

                # 创建一个新的数组，根据映射关系重排标签
        mapped_labels = np.vectorize(label_map.get)(label)

        # # 将标签值为255的部分改回原始值（保持不变）
        # mapped_labels[mapped_labels == 255] = label[mapped_labels == 255]

        img_id = re.findall(r'\d+', label_path.split('/')[-2])[0]
        img_all.append(imageio.imread(os.path.join(img_list,"1_1.png")))  
        img_all.append(imageio.imread(os.path.join(img_list,"1_2.png")))
        img_all.append(imageio.imread(os.path.join(img_list,"1_3.png")))
        img_all.append(imageio.imread(os.path.join(img_list,"2_1.png")))
        img_all.append(imageio.imread(os.path.join(img_list,"2_2.png")))
        img_all.append(imageio.imread(os.path.join(img_list,"2_3.png")))
        img_all.append(imageio.imread(os.path.join(img_list,"3_1.png")))
        img_all.append(imageio.imread(os.path.join(img_list,"3_2.png")))
        img_all.append(imageio.imread(os.path.join(img_list,"3_3.png")))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"1_1_label.png"),0)))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"1_2_label.png"),0)))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"1_3_label.png"),0)))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"2_1_label.png"),0)))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"2_2_label.png"),0)))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"2_3_label.png"),0)))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"3_1_label.png"),0)))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"3_2_label.png"),0)))
        labels.append(np.vectorize(label_map.get)(cv2.imread(os.path.join(img_list,"3_3_label.png"),0)))
        # for i in range(1,5):
        #     for name in dict_file[i-1]:
        #         img_all[i].append(imageio.imread(os.path.join(img_list,name+'.png')))

        if self.augmentations is not None:
            img_all, labels = self.augmentations(img_all, labels)

        # img = img.float()
        for i in range(len(img_all)):
            img_all[i]= img_all[i].float()
        label = [torch.from_numpy(i).long()  for i in labels]
        return img_all, label,img_id
            
            # return img,lbl,img_path

