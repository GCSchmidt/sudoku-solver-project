import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def prep_model_input(img):
    """
    Converts image into format for the tf CNN model.
    """
    img = 255-img
    img = img / 255.0
    img_array = tf.expand_dims(img, axis=0)
    return img_array


class SNET_Model():

    def __init__(self):
        self.model = load_model("./weights/SNET.keras", compile=False)
 
    def predict_digit(self, img) -> int:
        """
        Takes a binary image of shape 32x32 to predict a digit present in the image.
        """
        prediction = self.model.predict(prep_model_input(img), verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        return predicted_digit


if __name__ == "__main__":
    pass
