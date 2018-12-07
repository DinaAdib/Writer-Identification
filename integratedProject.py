import csv

from commonfunctions import *
from sklearn.mixture import GaussianMixture

##########Global Variables############
formsFolderName = "forms"
windowWidth=14
featuresCount = 2
iterationsCount = 10
accuracy = 0

################Code##################
if glob.glob(formsFolderName + '/*.png') != True:
    print("Please make sure that this folder exists")

writers = labelData()
writerForms = labelForms()
candidateWriters = {k:v for k, v in writerForms.items() if len(v) > 2}


for t in range(iterationsCount):

    trainingFormIDs, testingFormID, trainingLabels, testingLabel = pickRandomForms(candidateWriters)
    # trainingFormIDs = ['e07-094', 'e07-098','g06-011b', 'g06-026b', 'g07-065', 'g07-069a']
    # trainingFormIDs = ['e07-094', 'e07-098', 'g06-011b', 'g06-026b', 'g07-065', 'g07-069a']

    # trainingLabels = [0,0,1,1,2,2]
    # testingFormID =
    # testingFormID = ['e07-101','e07-105','g06-031b','g06-037b','g07-071a','g07-074a']
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

    xTrain = normalizeFeatures(xTrain)



    ##########Processing testing form#################
    testFeaturesVectors = np.empty([0, featuresCount])
    filename = formsFolderName + "/" + testingFormID + ".png"  ##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
    testFeaturesVectors, yTest = getLabeledData(filename, windowWidth, testingLabel, testFeaturesVectors,yTest)
    xTest = testFeaturesVectors

    xTest = normalizeFeatures(xTest)

    yTrain = np.array(yTrain)
    xTrain = np.array(xTrain)
    yTest = np.array(yTest)
    n_classes = len(np.unique(yTrain))
    uniqueClasses = np.unique(yTrain)


    predictions = []

    n_components = np.arange(1, 21)



    ########### Trial with BIC #################3
    for i in range(len(uniqueClasses)):
        indices=np.where(yTrain==i)
        # n_components = range(1,np.array(indices).shape[1])
        #
        # models = [GaussianMixture(n, covariance_type='diag', random_state=0,  init_params = 'kmeans').fit(xTrain[indices])
        #           for n in n_components]
        #
        # bic_y=[m.bic(xTrain[indices]) for m in models]
        # aic_y=[m.aic(xTrain[indices]) for m in models]
        #
        # bestN=min(np.argmin(aic_y),np.argmin(bic_y))

        classifier = GaussianMixture(n_components=2,
    covariance_type='diag', tol=0.00001,
    reg_covar=1e-06,
    max_iter=10000)
        classifier.fit(xTrain[indices])

        mypredictions = []
        for iTest in range(len(uniqueClasses)):
            testIndices=np.where(yTest==iTest)
            if len(testIndices[0]) != 0:
                mypredictions.append(classifier.score(xTest[testIndices]))
            else:
                mypredictions.append(-1)

        predictions.append(mypredictions)

    print("##############Testing###########")
    print(predictions)

    print(np.argmax(predictions ,axis=0))
    for iTest in range(len(uniqueClasses)):
        testIndices=np.where(yTest==iTest)
        if len(testIndices[0]) != 0:
            accuracy+= accuracy_score([iTest], [np.argmax(predictions,axis=0)[iTest]])
            print("Accuracy is ", accuracy_score([iTest], [np.argmax(predictions,axis=0)[iTest]]))


print("Total accuracy is ", accuracy/iterationsCount)