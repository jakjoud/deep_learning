import numpy as np
import keras
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from mnist_loader import MNISTPrep
import matplotlib.pyplot as plt
import keras.backend as K

K.set_image_data_format('channels_last')
prep = MNISTPrep()

#352497

def mymodel(input_shape):


    model = Sequential()
    model.add(Dense(800, input_dim=input_shape, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.225))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.225))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(10, activation='softmax'))

    return model

def compileRun(epochs=10, batch_size=128):
    prep.generateTrainingData()
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = prep.X_train_line, prep.Y_train, prep.X_dev_line, prep.Y_dev, prep.classes

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    model = mymodel(784)
    opt = Adam(lr=0.0005)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)
    history = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size)

    preds = model.evaluate(x=X_test, y=Y_test)

    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # serialize model to JSON
    model_json = model.to_json()
    with open("output/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("output/model.h5")
    print("Saved model to disk")

    meta = open("data/meta.txt", "w")
    meta.write("Loss = " + str(preds[0]) + "\n")
    meta.write("Accuracy = " + str(preds[1]) + "\n")
    meta.write("Epochs = " + str(epochs) + "\n")
    meta.write("Batch size = " + str(batch_size) + "\n")
    meta.close()

    res = model.predict(prep.X_test_line)
    toPrint = np.argmax(res, axis=1)

    toPrint = np.asanyarray(toPrint)
    np.savetxt("data/result_cnn.csv", toPrint, delimiter=",")

bpower = 64
batch_size = bpower
epochs = 15
compileRun(epochs, batch_size)