# -*- coding: utf-8 -*-
from typing import List
from scipy.misc import imread, imresize
from skimage.color import rgb2gray
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy
import os
import matplotlib.pyplot as plt

def image2Digit(image: numpy.array) -> numpy.array:
    """
    return a numpy.array
    """
    # resize the image
    imResized = imresize(image, (32, 32))
    # transfer the 3d image to 1d
    imGray = rgb2gray(image)

    return imGray.astype(numpy.float)

def getImageDataSet() -> (List[List[int]], List[str]):
    """
    return a list of data set that each element is a list of a sample and a list of file names
    """
    myDataSet = []
    myFiles = []
    pathBase = "data\\"
    for root, dirs, files in os.walk(pathBase):
        for file in files:
            try:
                path = pathBase + file
                image = imread(path)
                digit = numpy.resize(image2Digit(image), (1, 32 * 32))
                myDataSet.extend(digit.tolist())
                myFiles.append(file)
            except Exception:
                pass
    return myDataSet, myFiles

def prepDataSet(myDataSet: List[List[int]], myFiles: List[str]) -> (List[List], List[List], List[str], List[str], List[List]):
    """
    return training data set: list
           testing data set: list
           training files: list
           testing files: list
           featured data set: list
    """
    # descending the dimension of the data set to 16
    trainData, testData, trainFiles, testFiles = train_test_split(myDataSet, myFiles, test_size = 0.25)
    pca = PCA(n_components = 16, svd_solver = 'auto', whiten = False).fit(train)
    return pca.transform(train), pca.transform(test), trainFiles, testFiles, pca.components_.reshape((16, 32, 32))

def plotGallery(images: List[List], titles: List[str]):
    plt.figure(figsize = (16, 8))
    plt.subplots_adjust(bottom = 0, left = .01, right = .99, top = .90, hspace = .35)
    for i in range(len(images)):
        plt.subplot(2, 8, i + 1)
        plt.imshow(images[i].reshape(32, 32), cmap = plt.cm.gray)
        plt.title(titles[i], size = 8)
        plt.xticks(())
        plt.yticks(())
    plt.show()
    return

def test():
    myDataSet = getImageDataSet()
    trainDataSet, testDataSet, trainFiles, testFiles, eigenFaceSet = prepDataSet(myDataSet)
    eigenFaceTitleSet = ["eigenFace %d" % i for i in range(eigenFaceSet.shape[0])]
    plotGallery(eigenFaceSet, eigenFaceTitleSet)
