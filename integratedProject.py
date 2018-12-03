from commonfunctions import *


##########Global Variables############
formsFolderName = "formsE-H"
windowWidth=14

################Code##################
if glob.glob(formsFolderName + '/*.png') != True:
    print("Please make sure that this folder exists")

count = 0
# For each form, extract lines and their corresponding features
for filename in glob.glob(formsFolderName + '/*.png'):
    writers = labelData()
    formsFeaturesVectors = {writerID : [] for writerID in writers.values()}
    formID = re.match(r"" + formsFolderName + "/(.*)\.png", filename).group(1)

    extractedLines = preprocessForm(filename, formsFolderName)
    featuresVectors = getFeaturesVectors(extractedLines, windowWidth)
    if featuresVectors is not None:
        formsFeaturesVectors[writers[formID]].append(featuresVectors)

