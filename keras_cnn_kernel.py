import numpy as np
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from mnist_loader import MNISTPrep
import matplotlib.pyplot as plt
import keras.backend as K

K.set_image_data_format('channels_last')
prep = MNISTPrep()

def mymodel(input_shape, option1 = True, option2 = True):
    X_input = Input(input_shape)


    # Optical perception
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0')(X_input)
    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv2')(X)
    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    # Cognitive treatment
    X = Dropout(0.25)(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu', name='fc1')(X)
    X = Dense(1024, activation='relu', name='fc2')(X)
    X = Dropout(0.20)(X)
    X = Dense(512, activation='relu', name='fc3')(X)
    X = Dropout(0.20)(X)
    X = Dense(1024, activation='relu', name='fc4')(X)
    X = Dropout(0.15)(X)
    X = Dense(512, activation='relu', name='fc5')(X)
    X = Dropout(0.10)(X)
    X = Dense(10, activation='softmax', name='final')(X)

    model = Model(inputs=X_input, outputs=X, name='MNISTDigits')

    return model

def compileAndRun(index, option1=True, option2=True, epochs=12, batch_size=128):
    prep.generateTrainingData()
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = prep.X_train, prep.Y_train, prep.X_dev, prep.Y_dev, prep.classes

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

    model = mymodel((28,28,1))
    opt = Adam(lr=0.0001)
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)
    history = model.fit(x = X_train, y = Y_train, epochs = epochs, batch_size=batch_size)

    preds = model.evaluate(x = X_test, y = Y_test)

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
    with open("output/model_cnn.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("output/model_cnn.h5")
    print("Saved model to disk")

    meta = open("data/meta.txt", "w")
    meta.write("Loss = " + str(preds[0]) + "\n")
    meta.write("Accuracy = " + str(preds[1]) + "\n")
    meta.write("Epochs = " + str(epochs) + "\n")
    meta.write("Batch size = " + str(batch_size) + "\n")
    meta.close()

    res = model.predict(prep.X_test)
    toPrint = np.argmax(res,axis=1)

    toPrint = np.asanyarray(toPrint)
    np.savetxt("data/result_cnn.csv", toPrint, delimiter=",")

bpower = 32
batch_size = bpower
epochs = 30
compileAndRun(1, True, True, epochs, batch_size)