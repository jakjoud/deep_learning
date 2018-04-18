import numpy as np

class IDPrep:
    def __init__(self):
        self.data = np.genfromtxt('datasets/eyed_train.csv',delimiter=',',skip_header=True)
        X = np.genfromtxt('datasets/eyed_test.csv', delimiter=',', skip_header=True)
        self.X_test = np.reshape(X, (28000,64,64,1))

    def generateTrainingData(self):
        np.random.shuffle(self.data)
        X = self.data[:, 1:]
        Y = self.data[:, 0]
        print(X.shape)
        X = np.reshape(X, (42000, 64, 64, 1))
        Y = np.reshape(Y, (42000, 1))

        self.X_train = X[:41500, :, :, :]
        self.X_dev = X[41500:, :, :, :]
        self.Y_train = Y[:41500, :]
        self.Y_train = self.Y_train.reshape((1, self.Y_train.shape[0]))
        self.Y_dev = Y[41500:, :]
        self.Y_dev = self.Y_dev.reshape((1, self.Y_dev.shape[0]))
        self.classes = [0, 1, 2, 3]