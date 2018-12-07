## For signal processing
import csv

from scipy import signal
from scipy.signal import filter_design as fd
from scipy import fftpack

#import imutils
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


def showLines(greyImage, image2):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    ax = axes.ravel()
    ax[0].imshow(greyImage, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    ax[1].imshow(greyImage, cmap=cm.gray)

    ax[1].set_xlim((0, image2.shape[1]))
    ax[1].set_ylim((image2.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')
    plt.tight_layout()
    #plt.show()


def getHandwrittenPart(greyImage):
    # greyImage = cv2.GaussianBlur(greyImage,(5,5),0)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    kernel = kernel / 9
    greyImage = convolve2d(greyImage, kernel)
    # show_images([greyImage], ["BLURRING"])

    greyImage = np.uint8(greyImage)
    print(np.min(greyImage))
    greyImage[greyImage < 200] = 0
    cannyImage = cv2.Canny(greyImage, 100, 150)
    #    show_images([edges],["EDGES"])

    hspace, angles, dists = hough_line(cannyImage)
    hspace, angles, dists = hough_line_peaks(hspace, angles, dists, threshold=0.18 * np.max(hspace))

    #     minLineLength = 2
    #     maxLineGap = 2
    #     lines = cv2.HoughLinesP(cannyImage, 5, np.pi / 180, threshold=250)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    ax = axes.ravel()
    ax[0].imshow(greyImage, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    ax[1].imshow(greyImage, cmap=cm.gray)

    yprevious = 0
    yoptimum = []
    for angle, dist in zip(angles, dists):  # This line draws the line in red
        x1 = 0
        y1 = dist / math.cos(angle)

        y2 = 0
        x2 = dist / math.sin(angle)
        angleDegree = angle * 180 / np.pi
        # print(angleDegree)

        if (angleDegree <= 100 and angleDegree >= 80):
            yoptimum.append(x2)
        #
        # if yprevious-y >200:
        #     print(yprevious , y )
        #     yoptimum.append(yprevious)
        #     yoptimum.append(y)
        #     yprevious=y
        # else:
        #     yprevious=y

        ax[1].plot((x1, y1), (x2, y2), '-r')

    ax[1].set_xlim((0, greyImage.shape[1]))
    ax[1].set_ylim((greyImage.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')
    plt.tight_layout()
    # plt.show()
    height = greyImage.shape[0]

    yoptimum = np.array(yoptimum).astype(int)
    # print("BEFORE:" ,yoptimum)
    yoptimum = yoptimum[yoptimum > int(height / 8)]
    yoptimum = yoptimum[yoptimum < int(7 * height / 8)]

    yoptimum = np.sort(yoptimum)
    ystart = 0
    yend = 0

    # print("After" , yoptimum)

    for i, y in enumerate(yoptimum):
        if i != 0 and y - yoptimum[i - 1] > 1000:
            # show_images([greyImage[yoptimum[i-1]:y]], ["Lines"])
            return greyImage[yoptimum[i - 1]:y]

    return None
def greyImage(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(greyImg)
    print(threshold)
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



# This function preprocesses a form
def preprocessForm(filename):
    # Read image and convert it to binary
    grayImage = greyImage(cv2.imread(filename))
    # image = binarizeImage(cv2.imread(filename))
    # io.imshow(image)

    image = (getHandwrittenPart(grayImage))
    if image is None:
        print("A problem occured in file ", filename)
        return None
    image[image != 0] = 255
    image = 255 - image
    #
    # kernel = np.array([[1, 1, 1],
    #                    [1, 1, 1],
    #                    [1, 1, 1]])
    # kernel = kernel / 9
    # image = convolve2d(image, kernel)
    #
    # image=np.array(image).astype('uint8')

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    ax = axes.ravel()
    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    plt.tight_layout()
    # plt.show()
    verticalHistogram = image.sum(axis=1)
    # box = np.ones(5) / 5
    # verticalHistogram = np.convolve(verticalHistogram, box, mode='same')
    # ys = verticalHistogram
    # localMinima = [y for i, y in enumerate(ys) if
    #                ((i == 0) or (ys[i - 1] >= y)) and ((i == len(ys) - 1) or (y < ys[i + 1]))]
    smallest=int(np.average(verticalHistogram)-np.min(verticalHistogram))/4
    print(smallest)
    linesArrays = splitz(verticalHistogram, int(smallest))
    plt.plot(verticalHistogram)
    # plt.show()
    horizontalHistogram = image.sum(axis=0)
    # linesArrays = (list(splitz(verticalHistogram, 2000)))

    marginsArrays = (list(splitz(horizontalHistogram[30:], 100)))
    # plt.plot(horizontalHistogram)
    # plt.show()
    counter = 0
    extractedLines = []

    # create folder for this form to insert preprocessed lines images
    filename = filename[0:-4]##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
    if not os.path.exists(filename):
        os.makedirs(filename)

    # For each array (representing a line) extracted, perform some preprocessing operations
    for arr in (linesArrays):
        if (arr[-1] - arr[0] > 30):
            line = image[arr[0]:arr[-1], marginsArrays[0][0]:marginsArrays[-1][-1]]
            # show_images([line],["Line"])
            ## Vertical Scaling
            # if len(line) != 0:
            #     # while self.horizontal_hist_cropped[peaks[i]] > min(self.horizontal_hist_cropped[max(peaks[i] - window, 0):min(peaks[i] + window, len(self.horizontal_hist_cropped))]):
            #     vScaleImg = verticalScaling(line)
            #     # showLines(line, vScaleImg)
            # if vScaleImg is not None:
                # showLines(line, vScaleImg)
                # fig, axes = plt.subplots(1, 2, figsize=(20, 6))
                # ax = axes.ravel()
                # ax[0].imshow(vScaleImg, cmap=cm.gray)
                # ax[0].set_title('Input image')
                # ax[0].set_axis_off()
                # plt.tight_layout()
                # plt.show()
                # show_images([vScaleImg], ["Vertical Scaling"])
                ## Thinning
                # vScaleImg[vScaleImg != 0] = 255
                # thinned = thin(vScaleImg)
            line[line != 0] = 255
            thinned = thin(line)
            thinned = np.array(thinned).astype('uint8')
            thinned[thinned != 0] = 255
                # print("Vscale image unique values ", np.unique(thinned))
            extractedLines.append(thinned)
                #show_images([thinned],["hydrb hena"])
            cv2.imwrite(filename + "/output" + str(counter) + ".png", thinned)
            counter += 1
            # else:
            #     extractedLines.append(line)
            #     cv2.imwrite(filename + "/VScaledEmpty" + str(counter) + ".png", line)
            #     counter += 1

    return extractedLines

def normalizeFeatures(features):

    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0))
    # else:
    #     print("features ", features, " has std = 0")
    return features


# This function sets features vectors for a specific form image using sliding window technique
def getFeaturesVectors(extractedLines, windowWidth):
    imageFeaturesVectors = [[] for y in range(len(extractedLines))]
    for index, img in enumerate(extractedLines):
        # img[img==255] = 1  ##dark values are 1 bec they are the foreground
        # show_images([img],["AFTER OSTU"])

        # features to be extracted
        # centerOfGravityXLine = []
        # centerOfGravityYLine = []
        # blackPixelsLine = []
        # MomentXLine = []
        # MomentYLine = []
        # upperContourPosition = []
        # lowerContourPosition = []
        # upperContourOrientation = []
        # lowerContourOrientation = []
        # horizontalBlackToWhiteTrans = []
        # verticalBlackToWhiteTrans = []
        # blackPixelsBetweenContours = []
        #
        # h, w = img.shape
        # # print("h=", h ," w=",w)
        #
        # for i in range(w - windowWidth):
        #     slidingWindow = img[:h, i:i + windowWidth]
        #     # print(np.unique(slidingWindow))
        #     # ignore sliding windows that contain only background pixels
        #     checkwriting=slidingWindow[slidingWindow==255]
        #   #  checkwriting=int(checkwriting)
        #    # check=checkwriting.sum()
        #     if len(checkwriting) >= 10:
        #         contouredWindow = slidingWindow.copy()
        #
        #         indices = np.where(contouredWindow == [255])
        #         # show_images([slidingWindow])
        #         # NOTE: To get contours, sliding window must have values of 0 and 255
        #         # contours = cv2.findContours(contouredWindow, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #
        #         # contours = contours[0] if imutils.is_cv2() else contours[1]
        #         # c = np.concatenate(contours)
        #         # Fourth Feature: Upper Contour Position
        #
        #         topContour = [indices[1][0], indices[0][0]]
        #         upperContourPosition.append(np.array(topContour))
        #
        #         # Fifth Feature: Lower Contour Position
        #         bottomContour = [indices[1][-1], indices[0][-1]]
        #         lowerContourPosition.append(np.array(bottomContour))
        #
        #         # gradient = np.gradient(slidingWindow)
        #
        #         # Sixth Feature: Upper Contour Orientation
        #         # if gradient[0][topContour[1]][topContour[0]] != 0:
        #         #     upperContourOrientation.append(math.atan(
        #         #         gradient[1][topContour[1]][topContour[0]] / gradient[0][topContour[1]][topContour[0]]))
        #         # else:
        #         #     upperContourOrientation.append(math.pi / 2)
        #         #
        #         # # Seventh Feature: Lower Contour Orientation
        #         # if gradient[0][bottomContour[1]][bottomContour[0]] != 0:
        #         #     lowerContourOrientation.append(math.atan(
        #         #         gradient[1][bottomContour[1]][bottomContour[0]] / gradient[0][bottomContour[1]][
        #         #             bottomContour[0]]))
        #         # else:
        #         #     lowerContourOrientation.append(math.pi / 2)
        #
        #         minY = min(topContour[1], bottomContour[1])
        #         maxY = max(topContour[1], bottomContour[1])
        #         minX = min(topContour[0], bottomContour[0])
        #         maxX = max(topContour[0], bottomContour[0])
        #         slidingWindow = np.array(slidingWindow)
        #         contourRectangle = slidingWindow[minY:(maxY + 1), minX:(maxX + 1)]
        #         contourArea = (maxY - minY + 1) * (maxX - minX + 1)
        #         blackPixelsBetweenContours.append(np.sum(contourRectangle) / contourArea)
        #
        #         slidingWindow[slidingWindow == 255] = 1
        #
        #         rightTransitions, bottomTransitions = greycomatrix(slidingWindow, [1], [0, 3 * math.pi / 2], levels=2)
        #         # Eighth Feature: Black to white transition in both horizontal (to the right), and vertical (bottom) directions
        #         horizontalBlackToWhiteTrans.append(rightTransitions[:, 0][1][0])
        #         verticalBlackToWhiteTrans.append(bottomTransitions[:, 0][1][0])
        #
        #         slidingWindow = np.array(slidingWindow)
        #
        #         # first feature: number of pixels with text (white since image is inverted)
        #         blackPixelsLine.append(np.sum(slidingWindow[slidingWindow == 1]))
        #
        #         # second feature: center of gravity
        #         centerOfGravityX, centerOfGravityY = np.where(slidingWindow == 1)
        #         if len(centerOfGravityY) != 0 and len(centerOfGravityX) != 0:
        #             centerOfGravityXLine.append(round(np.average(centerOfGravityX)))
        #             centerOfGravityYLine.append(round(np.average(centerOfGravityY)))
        #             # print("At i= ",i,"Center of Gravity X: ",int(np.average(centerOfGravityX)) , "Center of Gravity Y: ", int(np.average(centerOfGravityY)))
        #
        #             # third feature: second order moment
        #             MomentXLine.append(int(np.sum(np.square(centerOfGravityX)) / (slidingWindow.shape[1] ** 2)))
        #             MomentYLine.append(int(np.sum(np.square(centerOfGravityY)) / (slidingWindow.shape[0] ** 2)))

                # append averaged features vector
        # imageFeaturesVectors[index]['Black Pixels Count']=np.average(blackPixelsLine)
        # imageFeaturesVectors[index]['Center of Gravity'] = [np.average(centerOfGravityXLine), np.average(centerOfGravityYLine)]
        # imageFeaturesVectors[index]['Moment'] = [np.average(MomentXLine), np.average(MomentYLine)]
        # imageFeaturesVectors[index]['Upper Contour Position'] = np.average(upperContourPosition, axis=0)
        # imageFeaturesVectors[index]['Lower Contour Position'] = np.average(lowerContourPosition, axis=0)
        # imageFeaturesVectors[index]['Upper Contour Direction'] = np.average(upperContourOrientation)
        # imageFeaturesVectors[index]['Lower Contour Direction'] =  np.average(lowerContourOrientation)
        # imageFeaturesVectors[index]['Horizontal Black to White'] =  np.average(horizontalBlackToWhiteTrans)
        # imageFeaturesVectors[index]['Vertical Black to White'] =  np.average(verticalBlackToWhiteTrans)
        # imageFeaturesVectors[index]['Black Pixels Fraction'] =  np.average(blackPixelsBetweenContours)

        # imageFeaturesVectors[index].append(np.average(blackPixelsLine))
        # imageFeaturesVectors[index].append(np.average(centerOfGravityXLine))
        # imageFeaturesVectors[index].append(np.average(centerOfGravityYLine))
        # imageFeaturesVectors[index].append(np.average(MomentXLine))
        # imageFeaturesVectors[index].append(np.average(MomentYLine))
        # imageFeaturesVectors[index].append(np.average(upperContourPosition, axis=0)[0])
        # imageFeaturesVectors[index].append(np.average(upperContourPosition, axis=0)[1])
        # imageFeaturesVectors[index].append(np.average(lowerContourPosition, axis=0)[0])
        # imageFeaturesVectors[index].append(np.average(lowerContourPosition, axis=0)[1])
        # imageFeaturesVectors[index].append(np.average(upperContourOrientation))
        # imageFeaturesVectors[index].append(np.average(lowerContourOrientation))
        # imageFeaturesVectors[index].append(np.average(horizontalBlackToWhiteTrans))
        ub, lb = getBounds(img)
        imageFeaturesVectors[index].append(ub)
        imageFeaturesVectors[index].append(lb)
        # imageFeaturesVectors[index].append(ub/lb)
        # imageFeaturesVectors[index].append(ub-lb)
        # imageFeaturesVectors[index].append(np.average(verticalBlackToWhiteTrans))
        # imageFeaturesVectors[index].append(np.average(blackPixelsBetweenContours))
    return imageFeaturesVectors


def pickRandomForms(candidateWriters):
    pickedWriters = random.sample(candidateWriters.keys(), 3)
    testingLabel = random.randint(0,2)
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


def getLabeledData(filename, windowWidth, labelVal,formsFeaturesVectors, labels):
    extractedLines = preprocessForm(filename)
    if extractedLines is not None:
        if len(extractedLines) != 0:
            featuresVectors = np.array(getFeaturesVectors(extractedLines, windowWidth))
            if featuresVectors is not None:
                formsFeaturesVectors = np.concatenate((formsFeaturesVectors, featuresVectors))
                for i in range(len(extractedLines)):
                    labels.append(labelVal)
                with open("Features" + str(labelVal) + ".csv", 'a') as filedata:
                    writer = csv.writer(filedata, delimiter=',')
                    for featuresVector in featuresVectors:
                        writer.writerow(featuresVector)
        else:
            print("Zero extracted lines")
    else:
        print("handwritten returned none")

    return formsFeaturesVectors, labels