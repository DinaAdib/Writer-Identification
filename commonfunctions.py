## For signal processing
import csv

from scipy import signal
from scipy.signal import filter_design as fd
from scipy import fftpack

# import imutils
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram

# from sklearn.cr import StratifiedKFold
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, rgb2hsv

from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
import re

import matplotlib.pyplot as plt
from matplotlib import cm
import math

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb
from skimage.feature import greycomatrix
from sklearn import cluster
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture
from skimage import feature

# Edges
from skimage.filters import sobel_h, sobel, sobel_v, roberts, prewitt
import glob
import skimage.io as io
import cv2
import os
from skimage.morphology import skeletonize, thin
from matplotlib.pyplot import bar
import math
from sklearn.metrics import accuracy_score
import random


# import the necessary packages
# def local_binary_pattern(greyImage , binaryImage):
#     greyImage=np.array(greyImage)
#     [h , w]=greyImage
#     histogram=np.zeros([1,265])
#     for i in range(2,h-1):
#         for j in range(2,w-1):
#             if binaryImage[i][j]==255:
#


# This function splits a sequence of numbers on a given value (smallest)
def splitz(seq, smallest):
    group = []
    for i in range(len(seq)):
        if (seq[i] >= (smallest)):
            group.append(i)
        elif group:
            yield group
            group = []


def splitOnArray(seq, smallest):
    group = []
    index = 0
    for i in range(len(seq)):
        if (seq[i] > smallest[index] + 5000):
            group.append(i)
        elif group:
            yield group
            group = []
            index += 1


## This function scales all images to height 120 pixels
def verticalScaling(img):
    img = np.array(img)
    original = np.copy(img)
    ## get histogram of horizontal projection
    [r, c] = img.shape

    horizontalProjection = np.sum(img, axis=1)
    gradient1 = np.gradient(horizontalProjection, edge_order=1)
    peak = np.argmax(horizontalProjection)
    p = horizontalProjection[peak]

    # get lb
    lb = peak + 1
    while lb < r and horizontalProjection[lb] > p / 2:
        lb = lb + 1
    # print(" Peak Value= ",horizontalProjection[peak])
    # print(" Sqrt Peak Value= ",math.sqrt(p) ," Peak/2= " , p/2 )
    # print("Peak Index= ", peak , " LB= ",lb , " r is ", r)

    if lb != r:
        img[lb, :] = np.ones([1, c])

    ##get ub
    ub = peak - 1
    while ub >= 0 and horizontalProjection[ub] > p / 2:
        ub = ub - 1
    if ub != 0:
        img[ub, :] = np.ones([1, c])
    # print(" UB= ",ub)
    if lb >= (r - 2):
        return None
    ## Plot Image and Histogram
    # plt.figure()
    bar(range(r), horizontalProjection, width=0.8, align='center')
    ###
    ascender = np.zeros((30, c))
    middleRegion = np.zeros((30, c))
    descender = np.zeros((30, c))
    ascender = cv2.resize(original[0:ub, :], dsize=(c, 30))
    middleRegion = cv2.resize(original[ub + 1:lb, :], dsize=(c, 30))
    descender = cv2.resize(original[lb + 1: r - 1, :], dsize=(c, 30))
    ##
    scaledImg = np.vstack((ascender, middleRegion, descender))

    # show_images([img , scaledImg] ,["LINE" , "SCALED"])

    return scaledImg


def getBounds(img):
    img = np.array(img)
    original = np.copy(img)
    ## get histogram of horizontal projection
    [r, c] = img.shape

    horizontalProjection = np.sum(img, axis=1)
    gradient1 = np.gradient(horizontalProjection, edge_order=1)
    peak = np.argmax(horizontalProjection)
    p = horizontalProjection[peak]

    # get lb
    lb = peak + 1
    while lb < r and horizontalProjection[lb] > p / 2:
        lb = lb + 1
    # print(" Peak Value= ",horizontalProjection[peak])
    # print(" Sqrt Peak Value= ",math.sqrt(p) ," Peak/2= " , p/2 )
    # print("Peak Index= ", peak , " LB= ",lb , " r is ", r)

    if lb != r:
        img[lb, :] = np.ones([1, c])

    ##get ub
    ub = peak - 1
    while ub >= 0 and horizontalProjection[ub] > p / 2:
        ub = ub - 1
    if ub != 0:
        img[ub, :] = np.ones([1, c])

    return ub, lb


# Show the figures / plots inside the notebook
def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        # plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    # plt.show()


# This function shows histogram of a given image
def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


# This function gets contours of words of a form image
def getContours(img):
    threshold = threshold_otsu(img)
    img[img > threshold] = 0
    img[img != 0] = 255

    image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 5)

    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    img = 255 - img
    return img


def interwordDistance(thresh):
    #line_grey = greyImage(lineRGB)
    #ret, thresh = cv2.threshold(line_grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   # # cv2.imshow("Bin thresh", thresh)
   #  #cv2.waitKey(0)
   #
   #  #showLines(thresh,line_grey,"Threshold")
   #  _, contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   #
   #

   #  for c in contours:
   #      # get the bounding rect
   #      x, y, w, h = cv2.boundingRect(c)
   #      # draw a white rectangle to visualize the bounding rect
   #      cv2.rectangle(lineRGB, (x, y), (x + w, y + h), 255, 5)
   #      wordCount+=1
   #      if wordCount==5:
   #          break
   #
   #  cv2.imshow("contour image", lineRGB)
   #  cv2.waitKey(0)
   #
   #  thresh=np.array(thresh)/255
   #  cv2.imshow("threshold" ,thresh)
   #  cv2.waitKey(0)
    scale_percent = 25
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    dim = (width, height)
    resizedLine = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)/255
    verticalSum=np.sum(resizedLine,axis=0)
    # plt.plot(range(len(verticalSum)),verticalSum)
    # plt.show()
    # showLines(resizedLine,resizedLine,"resized")
    zeroIndicesX=np.where(verticalSum==0)
    diffIndices=np.diff(zeroIndicesX)
    diffIndices[diffIndices<20]=0
    avgInterwordDistance=np.average(diffIndices)
    return avgInterwordDistance



    return avgInterwordDistance


def showLines(greyImage, image2, title):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    ax = axes.ravel()
    ax[0].imshow(greyImage, cmap=cm.gray)
    ax[0].set_title(title)
    ax[0].set_axis_off()

    ax[1].imshow(image2, cmap=cm.gray)
    ax[1].set_xlim((0, image2.shape[1]))
    ax[1].set_ylim((image2.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Image 2')
    plt.tight_layout()
    # plt.show()


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    [w, h] = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized



def getHandwrittenPart(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    y_list = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if w > img.shape[1] / 2 and w < img.shape[1] * 5 / 6:
     #       print((x, y, w, h), img.shape[1])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
            y_list.append(y)
    y_list = np.sort(y_list)
    #print(y_list)
    #show_images([img])

    return y_list[-2], y_list[-1]



# def getHandwrittenPart(greyImage):
#     # greyImage = cv2.GaussianBlur(greyImage,(5,5),0)
#     kernel = np.array([[1, 1, 1],
#                        [1, 1, 1],
#                        [1, 1, 1]])
#     kernel = kernel / 9
#     greyImage = convolve2d(greyImage, kernel)
#     # show_images([greyImage], ["BLURRING"])
#
#     greyImage = np.uint8(greyImage)
#     # print(np.min(greyImage))
#     greyImage[greyImage < 200] = 0
#     cannyImage = cv2.Canny(greyImage, 100, 150)
#     #    show_images([edges],["EDGES"])
#
#     hspace, angles, dists = hough_line(cannyImage)
#     hspace, angles, dists = hough_line_peaks(hspace, angles, dists, threshold=0.18 * np.max(hspace))
#
#     #     minLineLength = 2
#     #     maxLineGap = 2
#     #     lines = cv2.HoughLinesP(cannyImage, 5, np.pi / 180, threshold=250)
#
#     # fig, axes = plt.subplots(1, 2, figsize=(20, 6))
#     # ax = axes.ravel()
#     # ax[0].imshow(greyImage, cmap=cm.gray)
#     # ax[0].set_title('Input image')
#     # ax[0].set_axis_off()
#     # ax[1].imshow(greyImage, cmap=cm.gray)
#
#     yprevious = 0
#     yoptimum = []
#     for angle, dist in zip(angles, dists):  # This line draws the line in red
#         x1 = 0
#         y1 = dist / math.cos(angle)
#
#         y2 = 0
#         x2 = dist / math.sin(angle)
#         angleDegree = angle * 180 / np.pi
#         # print(angleDegree)
#
#         if (angleDegree <= 100 and angleDegree >= 80):
#             yoptimum.append(x2)
#         #
#         # if yprevious-y >200:
#         #     print(yprevious , y )
#         #     yoptimum.append(yprevious)
#         #     yoptimum.append(y)
#         #     yprevious=y
#         # else:
#         #     yprevious=y
#
#     #     ax[1].plot((x1, y1), (x2, y2), '-r')
#     #
#     # ax[1].set_xlim((0, greyImage.shape[1]))
#     # ax[1].set_ylim((greyImage.shape[0], 0))
#     # ax[1].set_axis_off()
#     # ax[1].set_title('Detected lines')
#     # plt.tight_layout()
#     # plt.show()
#     height = greyImage.shape[0]
#
#     yoptimum = np.array(yoptimum).astype(int)
#     # print("BEFORE:" ,yoptimum)
#     yoptimum = yoptimum[yoptimum > int(height / 8)]
#     yoptimum = yoptimum[yoptimum < int(7 * height / 8)]
#
#     yoptimum = np.sort(yoptimum)
#     ystart = 0
#     yend = 0
#
#     # print("After" , yoptimum)
#
#     for i, y in enumerate(yoptimum):
#         if i != 0 and y - yoptimum[i - 1] > 1000:
#             # show_images([greyImage[yoptimum[i-1]:y]], ["Lines"])
#             return greyImage[yoptimum[i - 1]:y]
#
#     return None


def greyImage(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(greyImg)
    # print(threshold)
    maxValue = np.array(greyImg).max()
    if maxValue <= 1:
        print("hhhhhh")
        greyImg = np.array(greyImg) * 255
        greyImg = greyImg.astype('uint8')

    return greyImg


# This function binarizes a given image to values 0 and 255
def binarizeImage(img):
    grayImg = np.copy(img)
    if len(img.shape) == 3:
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(grayImg)
    # print(threshold)
    grayImg[grayImg > threshold] = 0
    grayImg[grayImg != 0] = 255
    return grayImg


# This function reads forms.txt and gets form name and corresponding writer (labeled data)
def labelData():
    Writers = {}
    reader = open('forms.txt')
    inputs = list(reader)
    for line in inputs:
        formInfo = line.split(' ')
        formID = formInfo[0]
        writerID = formInfo[1]
        Writers[formID] = writerID

    return Writers


# This function reads forms.txt and gets form name and corresponding writer (labeled data)
def labelForms():
    Writers = {}
    reader = open('forms.txt')
    inputs = list(reader)
    for line in inputs:
        formInfo = line.split(' ')
        formID = formInfo[0]
        writerID = formInfo[1]
        if writerID not in Writers.keys():
            Writers[writerID] = [formID]
        else:
            Writers[writerID].append(formID)
    return Writers


def eightDirections(window):
    x = 2
    y = 2
    windowHist = np.zeros((1, 9))
    if window[x, y] == 0:
        return windowHist
    windowHist[0][0] = window[x + 1, y] & window[x + 2, y]
    windowHist[0][1] = window[x + 1, y - 1] & window[x + 2, y - 1]
    windowHist[0][2] = window[x + 1, y - 1] & window[x + 2, y - 2]
    windowHist[0][3] = window[x, y - 1] & window[x + 1, y - 2]
    windowHist[0][4] = window[x, y - 1] & window[x, y - 2]
    windowHist[0][5] = window[x, y - 1] & window[x - 1, y - 2]
    windowHist[0][6] = window[x - 1, y - 1] & window[x - 2, y - 2]
    windowHist[0][7] = window[x - 1, y - 1] & window[x - 2, y - 1]
    windowHist[0][8] = window[x - 1, y] & window[x - 2, y]
    return windowHist


def computeSlantHistogram(line, ub, lb):
    line = np.array(line)
    scale_percent = 25
    width = int(line.shape[1] * scale_percent / 100)
    height = int(line.shape[0] * scale_percent / 100)
    dim = (width, height)
    lineResize = cv2.resize(line, dim, interpolation=cv2.INTER_AREA)
    # showLines(line,lineResize,"after resize")
    edgeX = sobel_h(lineResize)
    edgeY = sobel_v(lineResize)
    edgeMagnitude = np.sqrt(np.square(edgeX) + np.square(edgeY))
    # showLines(edgeMagnitude, line, "in slant")
    # edgeAngle=np.arctan(edgeY/edgeX)
    # edgeAngle8=np.mod(edgeAngle,8*(np.ones(edgeAngle.shape)))
    histogram = np.zeros((1, 9))
    h, w = lineResize.shape
    lineResize[lineResize == 255] = 1
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            window = lineResize[i - 2:i + 3, j - 2:j + 3]
            windowHistogram = eightDirections(window)
            histogram = histogram + windowHistogram
    return histogram


# This function preprocesses a form
def preprocessForm(filename):
    # Read image and convert it to binary
    img = cv2.imread(filename)
    y1,y2= getHandwrittenPart(img)

    image=binarizeImage(img)
    image=image[y1+20:y2-20,:]

    #
    # kernel = np.array([[1, 1, 1],
    #                    [1, 1, 1],
    #                    [1, 1, 1]])
    # kernel = kernel / 9
    # image = convolve2d(image, kernel)
    #
    # image=np.array(image).astype('uint8')

    # fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    # ax = axes.ravel()
    # ax[0].imshow(image, cmap=cm.gray)
    # ax[0].set_title('Input image')
    # ax[0].set_axis_off()
    # plt.tight_layout()
    # plt.show()
    verticalHistogram = image.sum(axis=1)
    # box = np.ones(5) / 5
    # verticalHistogram = np.convolve(verticalHistogram, box, mode='same')
    # ys = verticalHistogram
    # localMinima = [y for i, y in enumerate(ys) if
    #                ((i == 0) or (ys[i - 1] >= y)) and ((i == len(ys) - 1) or (y < ys[i + 1]))]
    smallest = int(np.average(verticalHistogram) - np.min(verticalHistogram)) / 4
    # print(smallest)

    linesArrays = splitz(verticalHistogram, int(smallest))
    # plt.plot(verticalHistogram)
    # plt.show()
    horizontalHistogram = image.sum(axis=0)
    # linesArrays = (list(splitz(verticalHistogram, 2000)))

    smallest = int(np.average(horizontalHistogram) - np.min(horizontalHistogram)) / 4
    marginsArrays = (list(splitz(horizontalHistogram[30:], smallest)))
    # plt.plot(horizontalHistogram)
    # plt.show()
    counter = 0
    extractedLines = []

    # create folder for this form to insert preprocessed lines images
    filename = filename[0:-4]  ##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
    if not os.path.exists(filename):
        os.makedirs(filename)
    cv2.imwrite(filename + "/AfterExtraction.png", image)

    # For each array (representing a line) extracted, perform some preprocessing operations
    for arr in (linesArrays):
        if (arr[-1] - arr[0] > 30):
            # print( " marginsArrays[0][0] is " ,  marginsArrays[0][0], " and marginsArrays[-1][-1] ", marginsArrays[-1][-1])
            # print( "image shape is " ,  len(image))
            # print( "arr[0] is " ,  arr[0], " and arr[-1] ", arr[-1])
            line = image[arr[0]:arr[-1], marginsArrays[0][0]:marginsArrays[-1][-1]]
            line[line != 0] = 255
            # thinned = thin(line)
            # thinned = np.array(thinned).astype('uint8')
            # thinned[thinned != 0] = 255
            # print("Vscale image unique values ", np.unique(thinned))
            extractedLines.append(line)
            # show_images([thinned],["hydrb hena"])
            cv2.imwrite(filename + "/output" + str(counter) + ".png", line)
            counter += 1
            # else:
            #     extractedLines.append(line)
            #     cv2.imwrite(filename + "/VScaledEmpty" + str(counter) + ".png", line)
            #     counter += 1

    return extractedLines


def normalizeFeatures(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0))
    # else:
    #     print("features ", features, " has std = 0")

    return features, mean, std


# This function sets features vectors for a specific form image using sliding window technique
def getFeaturesVectors(extractedLines, windowWidth):
    featuresCount = 14 ##chafeaturesCountnge here
    imageFeaturesVectors = np.empty([0, featuresCount])
    histogram = np.zeros((1, 9))
    for index, img in enumerate(extractedLines):

        if len(img[img == 255]) > 10:
            lineFeaturesVector = []
            indices = np.where(img == [255])
            topContour = indices[1][0]
            bottomContour = indices[1][-1]
            ub, lb = getBounds(img)
            f1 = math.fabs(topContour - ub)
            f2 = math.fabs(ub - lb)
            f3 = math.fabs(lb - bottomContour)
            f4 = f1 / f2
            avgDist=interwordDistance(img)
            # f5 = f2 / f3
            lineFeaturesVector.append(f1)
            lineFeaturesVector.append(f2)
            lineFeaturesVector.append(f3)
            lineFeaturesVector.append(f4)
            lineFeaturesVector.append(avgDist)
            histogram = computeSlantHistogram(img, ub, lb)
            sumHistogram=np.sum(histogram)
            if sumHistogram!=0:
                histogram = histogram /sumHistogram
            lineFeaturesVector = np.reshape(lineFeaturesVector, (1, featuresCount-9))  ##change
            histogram = np.reshape(histogram, (1, 9))
            allFeatures = np.hstack((lineFeaturesVector, histogram))
            imageFeaturesVectors = np.vstack((imageFeaturesVectors, allFeatures))  ##change
        else:
            continue

    if imageFeaturesVectors.shape[0] > 0:
        return imageFeaturesVectors
    else:
        print("Failed to extract features , shape ", imageFeaturesVectors.shape)
        return None


def pickRandomForms(candidateWriters):
    pickedWriters = random.sample(candidateWriters.keys(), 3)
    testingLabel = random.randint(0, 2)
    trainingForms = []
    testingForm = ""
    trainingLabels = []

    for writerIndex, writer in enumerate(pickedWriters):
        candidateForms = random.sample(candidateWriters[writer], len(candidateWriters[writer]))
        trainingForms += candidateForms[:2]
        trainingLabels += [writerIndex, writerIndex]
        if writerIndex == testingLabel:
            testingForm = candidateForms[2]
    return trainingForms, testingForm, trainingLabels, testingLabel


def getLabeledData(filename, windowWidth, labelVal, formsFeaturesVectors, labels):
    print("current filename is ", filename)
    extractedLines = preprocessForm(filename)
    processedSuccessfully = True

    if extractedLines is not None:
        if len(extractedLines) != 0:
            featuresVectors = np.array(getFeaturesVectors(extractedLines, windowWidth))
            if featuresVectors is not None:
                formsFeaturesVectors = np.vstack((formsFeaturesVectors, featuresVectors))
                for i in range(featuresVectors.shape[0]):
                    labels.append(labelVal)
                with open("Features" + str(labelVal) + ".csv", 'a') as filedata:
                    writer = csv.writer(filedata, delimiter=',')
                    for featuresVector in featuresVectors:
                        writer.writerow(featuresVector)
            else:
                print("Zero extracted lines")
                processedSuccessfully = False
    else:
        print("handwritten returned none")
        processedSuccessfully = False

    return formsFeaturesVectors, labels, processedSuccessfully


def MinimumDistanceClassifier(testFeatures, features, classesCount, yTrain):
    dist = []
    means = []
    for i in range(classesCount):
        indices = np.where(yTrain == i)
        means.append(np.mean(features[indices], axis=0))

    for mean in means:
        dist.append(np.linalg.norm(mean - np.mean(testFeatures, axis=0)))

    index = np.argmin(np.array(dist), axis=0)

    classification = index + 1
    return classification


def KNN(testFeatures, features, k, labels):
    dist = []
    points = np.array(features)

    for feature in features:
        dist.append(np.linalg.norm(feature - np.mean(testFeatures, axis=0)))

    idx = np.argpartition(dist, k, axis=0)

    numbers, numCount = np.unique(labels[idx[:k]], return_counts=True)

    classification = numbers[np.argmax(numCount)]

    return classification


# TESTING CONNECTED COMPONENTS
#foldername="Lines"
# for filename in glob.glob('Lines/*.png'):
#     img = cv2.imread(filename)
#     connected_components(img)
