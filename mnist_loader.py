import numpy as np

class MNISTPrep:
    def __init__(self):
        self.data = np.genfromtxt('datasets/mnist_train.csv',delimiter=',',skip_header=True)
        X = np.genfromtxt('datasets/mnist_test.csv', delimiter=',', skip_header=True)
        self.X_test = np.reshape(X, (28000, 28, 28, 1))
        self.X_test_line = np.reshape(X, (28000, 784))

    def generateTrainingData(self):
        np.random.shuffle(self.data)
        X = self.data[:, 1:]
        Y = self.data[:, 0]
        print(X.shape)
        X = np.reshape(X, (42000, 28, 28, 1))
        X_line = np.reshape(X, (42000, 784))
        Y = np.reshape(Y, (42000, 1))

        self.X_train = X[:41500, :, :, :]
        self.X_train_line = X_line[:41500, :]
        self.X_dev = X[41500:, :, :, :]
        self.X_dev_line = X_line[41500:, :]
        self.Y_train = Y[:41500, :]
        self.Y_train = self.Y_train.reshape((1, self.Y_train.shape[0]))
        self.Y_dev = Y[41500:, :]
        self.Y_dev = self.Y_dev.reshape((1, self.Y_dev.shape[0]))
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

p = MNISTPrep()
p.generateTrainingData()

print(p.X_train_line.shape)