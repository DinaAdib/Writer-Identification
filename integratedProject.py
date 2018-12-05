import csv

from commonfunctions import *
from sklearn.mixture import GaussianMixture

##########Global Variables############
formsFolderName = "Testing"
windowWidth=14
featuresVectors = []

################Code##################
if glob.glob(formsFolderName + '/*.png') != True:
    print("Please make sure that this folder exists")

writers = labelData()
# formsFeaturesVectors = {writerID: np.empty([0, 14]) for writerID in writers.values()}
formsFeaturesVectors = np.empty([0, 14])
yTrain = []

count = 0
# For each form, extract lines and their corresponding features
for filename in glob.glob(formsFolderName + '/*.png'):
    formID = re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)
    extractedLines = preprocessForm(filename, formsFolderName)
    if len(extractedLines) != 0:
        featuresVectors = np.array(getFeaturesVectors(extractedLines, windowWidth))
        if featuresVectors is not None:
            formsFeaturesVectors = np.concatenate((formsFeaturesVectors, featuresVectors))
            for i in range(len(extractedLines)):
                yTrain.append(writers[formID])
            # formsFeaturesVectors[writers[formID]] = np.array(formsFeaturesVectors[writers[formID]])
            with open("Features"+ writers[formID]+".csv", 'a') as filedata:
                writer = csv.writer(filedata, delimiter=',')
                for featuresVector in featuresVectors:
                    writer.writerow(featuresVector)
    else:
        print("Zero extracted lines")
        # np.savetxt("Features"+ writers[formID]+".csv", formsFeaturesVectors[writers[formID]], delimiter=",")

yTrain = np.array(yTrain)
X_train = formsFeaturesVectors
n_classes = len(np.unique(yTrain))
uniqueClasses = np.unique(yTrain)

#-0---------------------------------------------
# __init__(self, n_components=1, covariance_type='full', tol=1e-3,
#                  reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
#                  weights_init=None, means_init=None, precisions_init=None,
#                  random_state=None, warm_start=False,
#                  verbose=0, verbose_interval=10)


# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GaussianMixture(n_components=n_classes,
                    covariance_type=covar_type, max_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[yTrain == i].mean(axis=0)
                                  for i in uniqueClasses])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)

    # h = plt.subplot(2, n_classifiers / 2, index + 1)

    # for n, color in enumerate('rgb'):
    #     data = X_train[yTrain == n]
    #     plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
    #                 label=yTrain)
    # Plot the test data with crosses
    # for n, color in enumerate('rgb'):
    #     data = X_test[y_test == n]
    #     plt.plot(data[:, 0], data[:, 1], 'x', color=color)

    # y_train_pred = np.array(classifier.predict(X_train))
    y_train_pred = classifier.predict(X_train)
    for writerID in uniqueClasses:
        yTrain[yTrain == writerID] =  int(np.where(uniqueClasses == writerID)[0][0])
    # test=list(enumerate(yTrain))
    # ytest=
    trainAccuracy = 0
    y_train_pred=y_train_pred.astype("uint")
    yTrain=yTrain.astype("uint")
    for pred_index in range(len(y_train_pred)):
        if y_train_pred[pred_index] == yTrain[pred_index]:
            trainAccuracy += 1

    print("Accuracy: " , trainAccuracy*100/len(y_train_pred))
    # plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
    #          transform=h.transAxes)

    # y_test_pred = classifier.predict(X_test)
    # test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    # plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
    #          transform=h.transAxes)