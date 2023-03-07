import numpy as np
from PIL import Image
import os
from tensorflow import keras


#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

## modify the training data
#x_train = x_train.reshape((-1, 28, 28, 1))
#x_train = x_train / 255.0

## build the neural network model
#model = keras.Sequential([
#    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#    keras.layers.MaxPooling2D((2, 2)),
#    keras.layers.Flatten(),
#    keras.layers.Dense(10, activation='softmax')
#])

## compile the model
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## train the model
#model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

## save the model to a file
#model.save('model.h5')


# load the model from a file
model = keras.models.load_model('model.h5')

for file_name in os.listdir('img'):
    if file_name.endswith(".png"):
        # load and convert the image to a numpy array
        img = Image.open(f'img/{file_name}').convert('L')
        img = img.resize((28, 28))
        img = np.array(img)

        # normalize the image data
        img = img / 255.0

        # reshape the image to a shape accepted by the model
        img = img.reshape((-1, 28, 28, 1))

        # make a prediction on the image
        pred = model.predict(img)
        digit = np.argmax(pred)

        print(f"Recognized digit for file {file_name} is: {digit}")
input("Press Enter to end...")
