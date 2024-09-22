import numpy as np
from PIL import Image
import os
from tensorflow import keras

# Load the model from a file
try:
    model = keras.models.load_model('model.h5')
except FileNotFoundError:
    print("Model file not found!")
    exit(1)

# Define a function to load and preprocess an image
def load_image(file_name: str) -> np.ndarray:
    img = Image.open(f'img/{file_name}').convert('L')
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape((-1, 28, 28, 1))
    return img

# Iterate over the images in the directory
for file_name in os.listdir('img'):
    if file_name.endswith(".png"):
        try:
            img = load_image(file_name)
            pred = model.predict(img)
            digit = np.argmax(pred)
            print(f"Recognized digit for file {file_name} is: {digit}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

input("Press Enter to end...")
