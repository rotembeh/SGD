from preprocessing import preproc, loadTest
import numpy as np
from mnist import MNIST

batchSize = 1
numOfLabels = 10
pictureSize = 785
dataSize = 1 #data block size

def main():

    preImages, preLabels = preproc()
    images = preImages[:dataSize]
    labels = preLabels[:dataSize]
    labels = initLabels(labels)
    images = np.array(images)
    images = np.asmatrix(images)
    W = np.random.rand(numOfLabels , pictureSize)
    W = np.asmatrix(W)

    print("Gradient Test: ")
    GradientTest(images, labels, W) #must be called at the end


def calcE(W, images, labels):
    E = 0
##    print("call from calcE with images")
    ETA = calcETA(images, W)
    for k in range(numOfLabels):
        Ck = labels[:,k]
        Ck = np.transpose(Ck)
        devisor = np.zeros((dataSize, 1))
        devisor = np.asmatrix(devisor)
        for j in range (numOfLabels):
            multiply = images * np.transpose(W[j])
            multiply -= ETA
            exponent = np.exp(multiply)
            devisor = np.add(devisor, exponent)
     ##   print(len(devisor) - np.count_nonzero(devisor))
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
    for p in range(numOfLabels): #gradient just for one weight
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
    return gradient[0]
    return gradient/batchSize

def calcETA(images, W):
    return 0
    maxValues = images * np.transpose(W[0])
    for j in range(numOfLabels):
        multiplyWj = images * np.transpose(W[j])
        maxValues = np.maximum(maxValues, multiplyWj) ##wise element maximum
    return maxValues

def GradientTest(images, labels, W):
    img = images[:1]
    label = labels[:1]
    d = np.random.rand(pictureSize)
    e0 = np.random.rand()
    results1 = []
    results2 = []
    for i in range(10): #gradient by w[0]
        ei = e0 * np.power(0.5,i)
        W[0] = W[0] + d * ei
     #   Wtag = []
      #  Wtag.append(W0plusD)
        elem1 = calcE(W, img, label)
        W[0] = W[0] - d * ei
        elem2 = calcE(W, img, label)
        results1.append(np.asscalar(np.abs(elem1 - elem2)))
        multiply = ei * np.transpose(d)
        elem3 = (multiply) * np.transpose(calcMBgradient(W, [0], img, label))
        results2.append (np.asscalar(np.abs(elem1 - elem2 - elem3)))
    print("results1 : ", results1)
    print("results2 : ", results2)

def GradientTestX(images, labels, W):
    img = images[:1]
    label = labels[:1]
    d = np.random.rand(pictureSize)
    e0 = np.random.rand()
    results1 = []
    results2 = []
    for i in range(10):
        ei = e0 * np.power(0.5,i)
        imgPlusD = img[0] + d * ei
        imgTag = []
        imgTag.append(imgPlusD)
        imgTag = np.asmatrix(imgTag)
        elem1 = calcE(W, imgPlusD, label)
        elem2 = calcE(W, img, label)
        results1.append(np.abs(elem1 - elem2))
        elem3 = ei * np.transpose(d) * calcMBgradient(W, [0], img, label)
        results2.append(np.abs(elem1 - elem2 - elem3))
    print("results1 : ", results1)
    print("results2 : ", results2)

def initLabels(labels):
    labels = np.asarray(labels, dtype="int32")
    labelsPower = np.power(2, labels)
    labels = (((labelsPower[:, None] & (1 << np.arange(numOfLabels)))) > 0).astype(int)
    return labels


if __name__ == '__main__':
    main()