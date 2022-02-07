import os
import glob
import torch
import torch.utils.data as data
from PIL import Image
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from ST_CGAN import TernaryTanh
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def DatasetSplit(metadata_file, dataset_dir = 'dataset', val_rate = 0.2, test_rate = 0.2):

    with open(dataset_dir + '/' + metadata_file) as f:
            metadata = json.load(f)

    print(len(metadata))
    train = []
    val = []
    test = []
    train_rate = 1 - val_rate - test_rate

    num_songs = len(metadata)
    num_train = int(train_rate * num_songs)
    num_val = int(val_rate * num_songs)
    num_test = int(test_rate * num_songs)

    print(num_train, num_val, num_test)
    
    for i, song in enumerate(metadata):
        if(i < num_train):
            train += song['images']
            for shift in song['shifts']:
                for img in song['images']:
                    train.append(img.rstrip('.png') + '_s' + str(shift) + '.png')

        elif(i < num_train + num_val):
            val += song['images']
            for shift in song['shifts']:
                for img in song['images']:
                    val.append(img.strip('.png') + '_s' + str(shift) + '.png')
        elif(i < num_train + num_val + num_test):
            test += song['images']
            for shift in song['shifts']:
                for img in song['images']:
                    test.append(img.strip('.png') + '_s' + str(shift) + '.png')
        else:
            train += song['images']
            for j, shift in enumerate(song['shifts']):
                for img in song['images']:
                    if(j%3 == 0):
                        val.append(img.strip('.png') + '_s' + str(shift) + '.png')
                    elif(j%3 == 1):
                        test.append(img.strip('.png') + '_s' + str(shift) + '.png')
                    else:
                        train.append(img.strip('.png') + '_s' + str(shift) + '.png')
    return train, val, test




class ImageDataset(data.Dataset):
    """
    Dataset class. Inherit Dataset class from PyTrorch.
    """
    def __init__(self, img_list, dir):
        self.img_list = img_list
        self.dir = dir

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        get tensor type preprocessed Image
        '''
        convert_tensor = transforms.ToTensor()

        img = Image.open(self.dir + '/' + self.img_list[index])
        img = convert_tensor(img)
        if(self.img_list[index].strip('.png').split('_')[-1] == '0' or self.img_list[index].strip('.png').split('_')[-2] == '0' ):
            preSeg = np.random.rand(img.shape[1], img.shape[2])
            preSeg = Image.fromarray(preSeg)
            preSeg = convert_tensor(preSeg)
            preSeg =  TernaryTanh(preSeg)
            preSeg = preSeg / 127.5
        else:
            preSeg = Image.open(self.dir + '/' +  self.img_list[index - 1])
            preSeg = convert_tensor(preSeg)


        return img*127.5, preSeg*127.5

if __name__ == '__main__':
    '''img = Image.open('../dataset/train/train_A/test.png').convert('RGB')
    gt_shadow = Image.open('../dataset/train/train_B/test.png')
    gt = Image.open('../dataset/train/train_C/test.png').convert('RGB')

    print(img.size)
    print(gt_shadow.size)
    print(gt.size)

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(img)
    f.add_subplot(1, 3, 2)
    plt.imshow(gt_shadow, cmap='gray')
    f.add_subplot(1, 3, 3)
    plt.imshow(gt)

    img_transforms = ImageTransform(size=286, crop_size=256, mean=(0.5, ), std=(0.5, ))
    img, gt_shadow, gt = img_transforms([img, gt_shadow, gt])

    print(img.shape)
    print(gt_shadow.shape)
    print(gt.shape)


    f.add_subplot(2, 3, 4)
    plt.imshow(transforms.ToPILImage()(img).convert('RGB'))
    f.add_subplot(2, 3, 5)
    plt.imshow(transforms.ToPILImage()(gt_shadow).convert('L'), cmap='gray')
    f.add_subplot(2, 3, 6)
    plt.imshow(transforms.ToPILImage()(gt).convert('RGB'))
    f.tight_layout()
    plt.show()'''
    train, val, test = DatasetSplit('dbMetadata.json', test_rate = 0.21)
    '''plt.imshow(Image.open('dataset/alb_esp1_0.png'))
    plt.show()
    for img, preImg in trainDataSet:
        print(img)
        print(preImg)
        plt.imshow(img[0,:,:])
        plt.show()
        plt.imshow(preImg[0,:,:])
        plt.show()'''