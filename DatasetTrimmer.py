import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import json

def FindBadImgs(dir):
    badImgs = []
    for img in os.listdir(dir + '/'):
        if(img == 'dbMetadata.json'):
            continue
        im = cv2.imread(dir + '/' + img)/2
        im = im.astype('float32')
        if np.sum(im) < 20:
            img.strip('.png')
            f = img.split('_')
            if 's' in f[-1]:
                img.strip(f[-1])
            badImgs.append(img)
    return badImgs

if __name__ == '__main__':
    dir = 'dataset'
    badImgs = FindBadImgs(dir)
    with open(dir + '/' + 'dbmetadata.json') as f:
        metadata = json.load(f)

    for i, song in enumerate(metadata):
        for img in badImgs:
            if(img + '.mid' == song['filename']):
                print('Removing', img)
                metadata[i]['images'].remove(img)