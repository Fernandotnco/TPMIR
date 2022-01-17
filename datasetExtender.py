import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

def shiftPianoRoll(matrix, n):
    if(n > 0):
        if(np.sum(matrix[:n]) == 0):
            reduced = np.delete(matrix, list(range(n)), 0)
            newMatrix = np.concatenate((reduced, np.zeros((n, 128))), axis = 0)
            return newMatrix
        else:
            return None
    else:
        if(np.sum(matrix[n:]) == 0):
            reduced = np.delete(matrix, list(range(87, 87 + n, -1)), 0)
            newMatrix = np.concatenate((np.zeros((-n, 128)), reduced), axis = 0)
            return newMatrix
        else:
            return None




if __name__ == '__main__':
    datasetName = 'dataset/'
    with open(datasetName + 'dbmetadata.json') as f:
        metadata = json.load(f)
    for i in range(len(metadata)):
        obj = metadata[i]
        print(obj['filename'])
        shifts = list(range(-6, 7, 1))
        for imgName in obj['images']:
            for shift in range(-6, 7, 1):
                img = cv2.imread(datasetName + imgName, cv2.IMREAD_GRAYSCALE)
                shifted = shiftPianoRoll(img, shift)
                if(shifted is not None):
                    cv2.imwrite(datasetName + imgName[:-4] + "_s" + str(shift) + '.png', shifted)
                else:
                    if shift in shifts:
                        shifts.remove(shift)
        print(shifts)
        metadata[i]['shifts'] = shifts
    with open(datasetName + 'dbMetadata.json', 'w') as fout:
        json.dump(metadata, fout)