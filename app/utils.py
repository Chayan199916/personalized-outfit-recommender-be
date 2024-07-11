import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf

# Load model here so it is initialized only once
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalMaxPooling2D()
])
model.trainable = False


def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result


def custom_image_generator(directory, batch_size=32, target_size=(224, 224)):
    import os
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    files = [os.path.join(directory, f) for f in os.listdir(
        directory) if os.path.isfile(os.path.join(directory, f))]
    num_files = len(files)

    while True:
        for offset in range(0, num_files, batch_size):
            batch_files = files[offset:offset + batch_size]
            batch_images = []

            for file in batch_files:
                img = load_img(file, target_size=target_size)
                img_array = img_to_array(img)
                batch_images.append(img_array)

            batch_images = np.array(batch_images)
            batch_images = preprocess_input(batch_images)
            yield batch_images, batch_files
