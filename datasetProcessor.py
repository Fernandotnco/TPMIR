from MIDIProcessor import ProcessMIDI
import os
import cv2
import json
import matplotlib.pyplot as plt

def CreateDataset(datasetDir):
    segments = []
    metaData = []
    dir = ''
    metaNames = []
    for dir in os.listdir(datasetDir):  
        print(dir)      
        for file in os.listdir(datasetDir + '/' + dir):
            print(file)
            segs = ProcessMIDI(datasetDir + '/' + dir + '/' + file)
            segments += segs
            imgNames = []
            for i in range(len(segs)):
                imgName = file[:-4] + '_' + str(i) + '.png'
                imgNames.append(imgName)
                metaNames.append(imgName)
            metaData.append({'composer': dir,'filename': file, 'segNumber': len(imgNames), 'images': imgNames})

    return segments, metaData, metaNames


if __name__ == '__main__':
    segments, metadata, metaNames = CreateDataset('./ClassicPianoMIDI')
    print(len(segments))

    for i in range(len(segments)):
        cv2.imwrite('dataset/' + metaNames[i], segments[i]*2)
    with open('dataset/dbMetadata.json', 'w') as fout:
        json.dump(metadata, fout)
