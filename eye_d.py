import numpy as np
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from eye_d_prep import IDPrep
import keras.backend as K

K.set_image_data_format('channels_last')
prep = IDPrep()

def mymodel(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv0')(X_input)
    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv2')(X)
    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv4')(X)
    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv5')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)


    X = Dropout(0.25)(X)
    X = Flatten()(X)
    X = Dense(4, activation='softmax', name='fc1')(X)

    model = Model(inputs=X_input, outputs=X, name='EyeD')

    return model

def compileAndRun(index, epochs=12, batch_size=128):
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

    model = mymodel((64,64,1))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    Y_train = keras.utils.to_categorical(Y_train, 4)
    Y_test = keras.utils.to_categorical(Y_test, 4)
    model.fit(x = X_train, y = Y_train, epochs = epochs, batch_size=batch_size)

    preds = model.evaluate(x = X_test, y = Y_test)

    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    meta = open("data/meta"+str(index), "w")
    meta.write("Loss = " + str(preds[0]) + "\n")
    meta.write("Accuracy = " + str(preds[1]) + "\n")
    meta.write("Epochs = " + str(epochs) + "\n")
    meta.write("Batch size = " + str(batch_size) + "\n")
    meta.close()

    res = model.predict(prep.X_test)
    toPrint = np.argmax(res,axis=1)

    toPrint = np.asanyarray(toPrint)
    np.savetxt("data/mnist_result"+str(index)+".csv", toPrint, delimiter=",")