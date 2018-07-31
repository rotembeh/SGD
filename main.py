from preprocessing import preproc, loadTest
import numpy as np
from mnist import MNIST

TOTALDATASIZE = 60000 #60000 real value
batchSize = 50
numOfBatches = 30
numOfLabels = 10
pictureSize = 784 + 1 #1 for Bias
numOfIterations = 25
dataSize = 15000 #data block size
printModulo = 100

def main():
    testImgs, testLabels = loadTest()
    preImages, preLabels = preproc()
    images = preImages[:dataSize]
    labels = preLabels[:dataSize]
    labels = initLabels(labels)
    images = np.array(images)
    images = np.asmatrix(images)
    print("preproc done")
    W = np.random.rand(numOfLabels , pictureSize)
  #  W = np.zeros((numOfLabels,pictureSize))
    W = np.asmatrix(W)
    E = calcE(W, images, labels)
    print("RANDOM CLASIFYING RESAULTS:")
    classifyImages(W, testImgs, testLabels)
    print("----------------------------")
    print ("first E = ", E)

    for t in range(int(TOTALDATASIZE / dataSize)):
        print("learning block ",t)
        images = preImages[t*dataSize:(t+1)*dataSize]
        labels = preLabels[t*dataSize:(t+1)*dataSize]
        labels = initLabels(labels)
        images = np.array(images)
        images = np.asmatrix(images)
        for k in range (numOfIterations):
            s = np.random.randint(dataSize , size = (numOfBatches, batchSize))
            for j in range (numOfBatches):
                gradient = calcMBgradient(W, s[j], images, labels)
                alpha = 0.5 ##can be change to 1/(k+1)
                W = updateWeights(W, alpha, gradient)
            if k % printModulo == 0:
                E = calcE(W, images, labels)
                print ("k= ", k, "new E = ", E)

    E = calcE(W, images, labels)
    print ("Last E = ", E)
    print("Done")
    print("TRANING DATA TEST:")
    classifyImages(W, preImages[:dataSize], preLabels[:dataSize])
    print("TESTING DATA TEST:")
    classifyImages(W, testImgs, testLabels)



def initLabels(labels):
    labels = np.asarray(labels, dtype="int32")
    labelsPower = np.power(2, labels)
    labels = (((labelsPower[:, None] & (1 << np.arange(numOfLabels)))) > 0).astype(int)
    return labels

def calcE(W, images, labels):
    E = 0
    ETA = calcETA(images, W)
    for k in range (numOfLabels):
        Ck = labels[:,k]
        Ck = np.transpose(Ck)
        devisor = np.zeros((dataSize, 1))
        devisor = np.asmatrix(devisor)
        for j in range (numOfLabels):
            multiply = images * np.transpose(W[j])
            multiply -= ETA
            exponent = np.exp(multiply)
            devisor = np.add(devisor, exponent)
        devisor = np.divide(1, devisor)
        devisor = np.diagflat(devisor)
        devided = images * np.transpose(W[k])
        devided -= ETA
        devided = np.exp(devided)
        element2 = np.log(devisor * devided)
        E += Ck * element2
    return (-1)*E / dataSize

##batch is list in size batchSize
def calcMBgradient(W, batch, images, labels):
    batchImageList = []
    batchLabelList = []
    gradient = []
    for i in range (batchSize):
        batchImageList.append(images[batch[i]])
        batchLabelList.append(labels[batch[i]])
    X = np.stack(batchImageList, axis=0)
    labelsBatchMatrix = np.stack(batchLabelList, axis=0)
    ETA = calcETA(X, W)
    for p in range(numOfLabels):
        Cp = labelsBatchMatrix[:,p]
        Cp = np.asmatrix(Cp)
        devisor = np.zeros((batchSize, 1))
        devisor = np.asmatrix(devisor)
        for j in range (numOfLabels):
            multiply = X * np.transpose(W[j])
            multiply -= ETA
            exponent = np.exp(multiply)
            devisor = np.add(devisor, exponent)
        devisor = np.divide(1, devisor)
        devisor = np.diagflat(devisor)
        devided = (X * np.transpose(W[p]))
        devided -= ETA
        devided = np.exp(devided)
        element2 = devisor * devided
        element2 = np.subtract(element2, np.transpose(Cp))
        Wp = np.transpose(X) * element2
        Wp = np.asarray(Wp)
        gradient.append(Wp)
    gradient = np.stack(gradient, axis=0)
    gradient = np.asmatrix(gradient)
    return gradient/batchSize

def updateWeights(W, alpha, gradient):
    W = np.add(W, -alpha * gradient)
    return W

def calcETA(images, W):
    maxValues = images * np.transpose(W[0])
    for j in range(numOfLabels):
        multiplyWj = images * np.transpose(W[j])
        maxValues = np.maximum(maxValues, multiplyWj) ##wise element maximum
    return maxValues

##for debugging:
def classImg(img, W):
    maxValue = img * np.transpose(W[0])
    classify = -1
    for j in range(numOfLabels):
        Pr = img * np.transpose(W[j])
        if (Pr >= maxValue):
            maxValue = Pr
            classify = j
    return classify

def classifyImages(W, images, labels):
    totalImages = len(labels)
    print("Testing ", totalImages," images...")
    labelsMatrix = np.asmatrix(labels)
    predictionMatrix = images * np.transpose(W)
    compareVector = np.argmax(predictionMatrix, axis=1) - np.transpose(labelsMatrix)
    countMistakes = np.count_nonzero(compareVector)
    countCorrect = totalImages-countMistakes
    print("corret: ", countCorrect, ". mistakes: ", countMistakes)
    print("Accuracy: ", countCorrect/totalImages)
    mistakesCounters = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(compareVector)):
        if compareVector[i] != 0:
        #    print ("real: ", labels[i], ". our Guess: ", np.argmax(predictionMatrix, axis=1)[i])
            mistakesCounters[labels[i]] += 1
    #print (mistakesCounters)

if __name__ == '__main__':
	main()