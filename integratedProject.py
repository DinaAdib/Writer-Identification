import csv

from commonfunctions import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture

##########Global Variables############
formsFolderName = "forms"
windowWidth=14
featuresCount = 14##change here
iterationsCount = 100
accuracy = 0



def plot(x, y, z,  title='', xlabel='', ylabel='', zlabel='',color_style_str='', label_str='', figure=None, axis=None):
    if figure is None:
        fig = plt.figure()
    else:
        fig = figure
    ax = axis

    #TODO: Add title, x_label, y_label, z_label to ax.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    #TODO: Scatter plot of data points with coordinates (x, y, z) with the corresponding color and label.
    ax.scatter(x, y, z, c=color_style_str)
    handles, labels = ax.get_legend_handles_labels()

    unique = list(set(labels))
    handles = [handles[labels.index(u)] for u in unique]
    labels = [labels[labels.index(u)] for u in unique]

    ax.legend(handles, labels)

################Code##################
if glob.glob(formsFolderName + '/*.png') != True:
    print("Please make sure that this folder exists")

writers = labelData()
writerForms = labelForms()
candidateWriters = {k:v for k, v in writerForms.items() if len(v) > 2}

total = 0
while total < iterationsCount:

    trainingFormIDs, testingFormID, trainingLabels, testingLabel = pickRandomForms(candidateWriters)
    # trainingFormIDs = ['g06-026a', 'g06-011a', 'b04-181', 'b04-175', 'a04-085', 'a04-081']
    # testingFormID = 'a04-089'

    xTrain = np.empty([0, featuresCount])
    yTrain = []
    xTest = np.empty([0, featuresCount])
    yTest = []
    featuresVectors = []
    formsFeaturesVectors = np.empty([0, featuresCount])
    testFeaturesVectors = np.empty([0, featuresCount])

    ##########Processing Training form#################
    for trainingIndex, formID in enumerate(trainingFormIDs):
        filename = formsFolderName + "/" + formID + ".png"##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
        formsFeaturesVectors, labels, processedSuccessfully = getLabeledData(filename, windowWidth, trainingLabels[trainingIndex], formsFeaturesVectors, yTrain)
        if processedSuccessfully == False:
            continue
    if processedSuccessfully == False:
        continue

    xTrain = formsFeaturesVectors
    xTrain, mean, std = normalizeFeatures(xTrain)


    ########Processing Testing form###############
    filename = formsFolderName + "/" + testingFormID + ".png"  ##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
    xTest, yTest, processedSuccessfully = getLabeledData(filename, windowWidth, testingLabel, testFeaturesVectors,yTest)
    if processedSuccessfully == False:
        continue
    xTest = (xTest - mean) / std

    yTrain = np.array(yTrain)
    xTest = np.array(xTest)
    xTrain = np.array(xTrain)
    yTest = np.array(yTest)
    n_classes = len(np.unique(yTrain))
    uniqueClasses = np.unique(yTrain)


    GMMPredictions = []
    GMMFailed = False
    for i in range(len(uniqueClasses)):
        indices=np.where(yTrain==i)
        classifier = GaussianMixture(n_components=2, random_state=0,
                                 covariance_type='diag', tol=0.00001,
                                 reg_covar=1e-06,
                                 max_iter=10000)
        if xTrain[indices].shape[0] > 1:
            classifier.fit(xTrain[indices])


            mypredictions = []
            for iTest in range(len(uniqueClasses)):
                testIndices=np.where(yTest==iTest)
                if len(testIndices[0]) != 0:
                    mypredictions.append(classifier.score(xTest[testIndices]))
                else:
                    mypredictions.append(-1)
        else:
            GMMFailed = True
        GMMPredictions.append(mypredictions)

    if GMMFailed == True:
        continue
    print("##############Testing " + str(total+1) + " ###########")

    minDistPrediction = (MinimumDistanceClassifier(xTest, xTrain, len(uniqueClasses), yTrain) - 1)
    GMMPrediction = np.argmax(GMMPredictions ,axis=0)[yTest[0]]
    KNNPrediction = KNN(xTest, xTrain, 3, yTrain)

    if (KNNPrediction != GMMPrediction) and (GMMPrediction == minDistPrediction):
        if np.fabs(GMMPredictions[GMMPrediction][yTest[0]]-GMMPredictions[KNNPrediction][yTest[0]]) < 0.6:
            GMMPrediction = KNNPrediction

    uniquePredictions, uniquePredictionsCount = np.unique([KNNPrediction, minDistPrediction, GMMPrediction], return_counts=True)
    classification = uniquePredictions[np.argmax(uniquePredictionsCount)]

    accuracy += accuracy_score([yTest[0]], [classification])
    total = total + 1
    print("Current accuracy is ", accuracy / total)


print("Total accuracy is ", accuracy/total)