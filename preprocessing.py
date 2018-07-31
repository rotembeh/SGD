from mnist import MNIST
import numpy as np

def preproc():
    mndata = MNIST('mnistData')

    images, labels = mndata.load_training()
    images = np.divide(images, 255)
    images = np.insert(images, 0, 1, axis=1) #bias pixel
    s = np.arange(images.shape[0])
    np.random.shuffle(s)
    labels = np.asarray(labels)
    return images[s], labels[s];

def loadTest():
    mndata = MNIST('mnistData')
    images, labels = mndata.load_testing()
    images = np.divide(images, 255)
    images = np.insert(images, 0, 1, axis=1) #bias pixel
    return images, labels;

def printImages(images, labels):
     for i in range(len(images)):
             print(MNIST.display(images[i],threshold=0.5))
