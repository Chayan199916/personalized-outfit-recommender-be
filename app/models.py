import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50


def create_model():
    base_model = ResNet50(weights='imagenet',
                          include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    model.trainable = False
    return model
