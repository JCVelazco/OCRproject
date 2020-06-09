from __future__ import division, print_function

import sys
import os
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if os.path.exists(ros_path):
    if ros_path in sys.path:
        sys.path.remove(ros_path)
import random
import numpy as np
import cv2
from utils import preprocess

class Sample:
    "Sample from the dataset"
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath

class Batch:
    "Batch containing images and gorund truth texts"
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class DataLoader:
    "Loads data in IAM format."
    def __init__(self, filePath, batchSize, imgSize, maxTxtLen):
        "Loads dataset at given location. Preprocess images using previous stages from project"

        assert filePath[-1]=='/' # Checks for valid directory

        self.dataAugmentation = False # This becomes true for training. Otherwise, it's False
        self.index = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f = open(filePath + "words.txt")
        chars=set()
        bad_samples=[]
        bad_samples_reference=['a01-117-05-02.png', 'r06-022-03-05.png']
        for line in f:
            # ignore comments
            if not line or line[0] == "#":
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> actual path (part1/part1-part2/part1-part2-part3)
            fileNameSplit = lineSplit[0].split('-')
            fileName = filePath + "words/" + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

            # method truncateLabel() prevents infinite gradient in ctc_loss
            gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTxtLen)
            chars = chars.union(set(list(gtText)))

            # checks for size != 0
            if not os.path.getsize(fileName):
                bad_samples.append(fileName)
                continue

            # gets sample into list
            self.samples.append(Sample(gtText, fileName))

        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found: ", bad_samples)
            print("Expected damaged images: ", bad_samples_reference)

        # split set to training and validation subsets: 95% - 5%
        splitIndex = int(0.95*len(self.samples))
        self.trainSamples = self.samples[:splitIndex]
        self.validationSamples = self.samples[splitIndex:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 25000

        #start to train set

        self.trainSet()

        self.charList = sorted(list(chars))

    def truncateLabel(self, text, maxTxtLen):
        # ctc loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeated letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost +=1
            if cost > maxTxtLen:
                return(text[:i])
        return text

    def trainSet(self):
        "Switch to randomly chosen subset of training set"
        self.dataAugmentation = True
        self.index = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    def validationSet(self):
        self.dataAugmentation = False
        self.index = 0
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        "Current batch indez and overall number of batches"
        return (self.index // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        "Iterator"
        return self.index + self.batchSize <= len(self.samples)

    def getNext(self):
        "Iterator"
        batchRange = range(self.index, self.index + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        # preprocessing of images
        imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
        self.index += self.batchSize
        return Batch(gtTexts, imgs)
