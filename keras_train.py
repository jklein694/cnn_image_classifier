from keras.layers import *
from sklearn.model_selection import train_test_split

import image_augmentation_channels as ia
import keras_cnn

model = keras_cnn.Xception()

x, y = ia.load_data(299, 15000)

X_train, y_train, X_test, y_test = train_test_split(x, y)

# Add a forth dimension since Keras expects a list of images
X_train = np.expand_dims(X_train, axis=0)

# Scale the input image to the range used in the trained network
X_train = keras_cnn.preprocess_input(X_train)
# Load the separate test data set

model.fit(X_train, y_train,
          epochs=5,
          shuffle=True,
          verbose=2,
          batch_size=64)

test_error_rate = model.evaluate(X_test, y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
