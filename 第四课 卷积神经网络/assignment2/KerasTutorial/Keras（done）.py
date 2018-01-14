import numpy as np
import keras
#import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils

from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

### keras visualization
from keras.utils import plot_model


################# import data set #################
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))



################# building a model in keras #################
def HappyModel(input_shape):
    """
        Implementation of the HappyModel.

        Arguments:
        input_shape -- shape of the images of the dataset

        Returns:
        model -- a Model() instance in Keras
    """

    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(2,2))(X_input)
    X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)

    X = ZeroPadding2D(padding=(1,1))(X)
    X = Conv2D(16, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)

    X = ZeroPadding2D(padding=(1,1))(X)
    X = Conv2D(32, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)

    # FC
    X = Flatten()(X)
    Y = Dense(1, activation='sigmoid')(X)

    model = Model(inputs = X_input, outputs = Y, name = 'HappyModel')
    return model


################# train a model by keras #################
# 1.create the model
happyModel = HappyModel((64,64,3))
# 2.compile the model to configure the learning process
happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'] )
# 3.train the model
    # Note that if you run fit() again, the model will continue to train with the parameters it has already learnt instead of reinitializing them.
happyModel.fit(x = X_train, y=Y_train, batch_size=32, epochs=10)
# 4.test/evaluate the model
preds = happyModel.evaluate(x=X_test, y=Y_test)
print("\n Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))


################# test your own image #################
img_path = 'images/gone.jpg'
img = image.load_img(img_path, target_size=(64,64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))

plot_model(happyModel, to_file='happymodel.png', show_shapes = True)

print("\n ***************** done here!!")