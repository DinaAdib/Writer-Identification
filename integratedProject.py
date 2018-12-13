import time
from commonfunctions import *
from sklearn.mixture import GaussianMixture

##########Global Variables############
formsFolderName = "forms"
windowWidth=14
featuresCount = 14##change here
iterationsCount = 200
accuracy = 0

writers = labelData()
writerForms = labelForms()
candidateWriters = {k:v for k, v in writerForms.items() if len(v) > 2}

total = 0
# create output files
f = open("results.txt", "w+")
f.close()
f = open("time.txt", "w+")
f.close()

while total < iterationsCount: #H

# for iterationFolder in sorted(glob.glob("data/*")): #H
    trainingFormIDs, testingFormID, trainingLabels, testingLabel = pickRandomForms(candidateWriters) #H
    # trainingFormIDs, testingFormID, trainingLabels = readData(iterationFolder) #H
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

    t0 = time.time()
    for trainingIndex, formID in enumerate(trainingFormIDs):
        filename = formsFolderName + "/" + formID + ".png"##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1) #H
        # filename = formID #H
        formsFeaturesVectors, labels, processedSuccessfully = getLabeledData(filename, windowWidth, trainingLabels[trainingIndex], formsFeaturesVectors, yTrain)
        if processedSuccessfully == False:
            continue
    if processedSuccessfully == False:
        continue

    xTrain = formsFeaturesVectors
    xTrain, mean, std = normalizeFeatures(xTrain)


    ########Processing Testing form###############
    filename = formsFolderName + "/" + testingFormID + ".png"  ##re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1) #H
    # filename = testingFormID #H
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
            mypredictions.append(classifier.score(xTest))
        else:
            GMMFailed = True
        GMMPredictions.append(mypredictions)

    if GMMFailed == True:
        continue
    print("##############Testing " + str(total+1) + " ###########")

    # minDistPrediction = MinimumDistanceClassifier(xTest, xTrain, len(uniqueClasses), yTrain)
    GMMPrediction = np.argmax(GMMPredictions ,axis=0)[0]
    KNNPrediction = KNN(xTest, xTrain, 3, yTrain)
    SVMPrediction = SVM(xTrain, yTrain, xTest)
    # if (KNNPrediction != GMMPrediction) and (GMMPrediction == minDistPrediction):
    #     if np.fabs(GMMPredictions[GMMPrediction][0]-GMMPredictions[KNNPrediction][0]) < 0.6:
    #         GMMPrediction = KNNPrediction

    uniquePredictions, uniquePredictionsCount = np.unique([KNNPrediction, GMMPrediction, SVMPrediction], return_counts=True)
    classification = uniquePredictions[np.argmax(uniquePredictionsCount)]
    t1 = time.time()
    print(classification)

    f = open("results.txt", "a")
    f.write(str(classification+1)+"\n")
    f.close()

    f = open("time.txt", "a")
    f.write(str(round(t1-t0, 2)) + "\n")
    f.close()

    accuracy += accuracy_score([yTest[0]], [classification])
    total = total + 1
    print("Current accuracy is ", accuracy / total)


print("Total accuracy is ", accuracy/total)