import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def prep_model_input(img) -> np.array:
    """
    Converts image into format for the tf CNN model.
    """
    img = 255-img
    img = img / 255.0
    return img


class SNET_Model():

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "weights/SNET.keras")
        self.model = load_model(model_path, compile=False)

    def predict_digits(self, images) -> list:
        """
        Takes a list of binary images of shape 32x32 to predict a digit present in each image.
        """
        images = np.array([prep_model_input(img) for img in images])
        if images.ndim < 3:  # add batch dimension for single image case
            images = np.expand_dims(images, axis=0)
        predictions = self.model.predict(images, verbose=0)
        predicted_digits = np.argmax(predictions, axis=1)
        return predicted_digits.tolist()


if __name__ == "__main__":
    pass
