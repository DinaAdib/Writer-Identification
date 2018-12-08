import csv

from commonfunctions import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture

##########Global Variables############
formsFolderName = "forms"
windowWidth=14
featuresCount = 3
iterationsCount = 40
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
for t in range(iterationsCount):

    trainingFormIDs, testingFormID, trainingLabels, testingLabel = pickRandomForms(candidateWriters)
    # trainingFormIDs = ['e07-094','g06-011b', 'g06-026b', 'g07-065', 'g07-069a']
    # trainingFormIDs = ['e07-094', 'e07-098', 'g06-011b', 'g06-026b', 'g07-065', 'g07-069a']

    # trainingLabels = [0,1,1,2,2]
    # testingFormID =
    # testingFormID = ['e07-101','e07-105','g06-031b','g06-037b','g07-071a','g07-074a']
    # testingFormID = ['e07-101','g06-031b']
    # testingFormID = ['e07-101']
    # testingLabel = 1
    # testingLabel = [0]
    # formsFeaturesVectors = {writerID: np.empty([0, 14]) for writerID in writers.values()}
    # trainingFormIDs = ['f04-100']
    xTrain = np.empty([0, featuresCount])
    yTrain = []
    xTest = np.empty([0, featuresCount])
    yTest = []
    featuresVectors = []
    formsFeaturesVectors = np.empty([0, featuresCount])

    ##########Processing Training form#################
    for trainingIndex, formID in enumerate(trainingFormIDs):
        filename = formsFolderName + "/" + formID + ".png"##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
        formsFeaturesVectors, labels = getLabeledData(filename, windowWidth, trainingLabels[trainingIndex], formsFeaturesVectors, yTrain)
        xTrain = formsFeaturesVectors

    xTrain, mean, std = normalizeFeatures(xTrain)

    testFeaturesVectors = np.empty([0, featuresCount])
    ##########Processing testing form#################
    # for testingIndex, testFormID in enumerate(testingFormID):
        # filename = formsFolderName + "/" + testFormID + ".png"  ##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
        # testFeaturesVectors, yTest = getLabeledData(filename, windowWidth, testingLabel[testingIndex], testFeaturesVectors,yTest)
        # xTest = testFeaturesVectors

    filename = formsFolderName + "/" + testingFormID + ".png"  ##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
    testFeaturesVectors, yTest = getLabeledData(filename, windowWidth, testingLabel, testFeaturesVectors,
                                                yTest)
    xTest = testFeaturesVectors

    xTest = (xTest - mean) / std

    yTrain = np.array(yTrain)
    xTest = np.array(xTest)
    xTrain = np.array(xTrain)
    yTest = np.array(yTest)
    n_classes = len(np.unique(yTrain))
    uniqueClasses = np.unique(yTrain)


    predictions = []

    # Visualization of features
    # fig = plt.figure()
    # c = ['r','g','b']
    # ax = fig.add_subplot('111', projection='3d')
    # for i in range(len(uniqueClasses)):
    #     indices = np.where(yTrain == i)
    #     plot(np.mean(xTrain[indices, 0]), np.mean(xTrain[indices, 1]),i, title='Training Data',
    #          xlabel='Features', ylabel='Feature 2', zlabel='Class.', color_style_str=c[i],
    #          label_str="Rectangle",
    #          figure=fig, axis=ax)
    #
    # plot(np.mean(xTest[:,0]), np.mean(xTest[:, 1]), 1, title='Training Data',
    #      xlabel='Features', ylabel='Feature 2', zlabel='Class.', color_style_str='c',
    #      label_str="Rectangle",
    #      figure=fig, axis=ax)
    # plt.show()
    ########### Trial with BIC #################3
    for i in range(len(uniqueClasses)):
        indices=np.where(yTrain==i)
        # n_components = range(1,4)
        #
        # models = [GaussianMixture(n, covariance_type='diag', random_state=0,  init_params = 'kmeans').fit(xTrain[indices])
        #           for n in n_components]
        #
        # bic_y=[m.bic(xTrain[indices]) for m in models]
        # aic_y=[m.aic(xTrain[indices]) for m in models]
        #
        # bestN=min(np.argmin(aic_y),np.argmin(bic_y))
        # print("Best n is ", bestN)
        classifier = GaussianMixture(n_components=2,random_state=0,
    covariance_type='diag', tol=0.00001,
    reg_covar=1e-06,
    max_iter=10000)
        if xTrain[indices].shape[0] != 0:
            classifier.fit(xTrain[indices])

            mypredictions = []
            # for iTest in range(len(uniqueClasses)):
            #     testIndices=np.where(yTest==iTest)
            #     if len(testIndices[0]) != 0:
            #         for testVector in xTest[testIndices]:
            #             testVector = testVector.reshape(1,-1)
            #             mypredictions.append(classifier.predict_proba(testVector))
            #         else:
            #             mypredictions.append(-1)
            # for iTest in range(len(uniqueClasses)):
            #     testIndices=np.where(yTest==iTest)
            #     if len(testIndices[0]) != 0:
            #         mypredictions.append(classifier.predict(xTest[testIndices]))
            #     else:
            #         mypredictions.append(-1)

            mypredictions = []
            for iTest in range(len(uniqueClasses)):
                testIndices=np.where(yTest==iTest)
                if len(testIndices[0]) != 0:
                    # for testVector in xTest[testIndices]:
                    # testVector = testVector.reshape(1,-1)
                    mypredictions.append(classifier.score(xTest[testIndices]))
                else:
                    mypredictions.append(-1)




            predictions.append(mypredictions)
        else:
            print("xtrain's shape is ", xTrain[indices].shape)



    # predictions.append(MinimumDistanceClassifier(xTest, xTrain, len(uniqueClasses), yTrain))

    print("##############Testing###########")
    print(predictions)

    print(np.argmax(predictions ,axis=0))

    for iTest in range(len(uniqueClasses)):
        testIndices=np.where(yTest==iTest)
        if len(testIndices[0]) != 0:
            accuracy+= accuracy_score([iTest], [np.argmax(predictions,axis=0)[iTest]])
            print("Current accuracy is ", accuracy/(t+1))


print("Total accuracy is ", accuracy/iterationsCount)