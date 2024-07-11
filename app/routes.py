from flask import Blueprint, request, jsonify
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import base64
from app.utils import extract_features

main = Blueprint('main', __name__)

with open('embeddings.pkl', 'rb') as f:
    feature_list = pickle.load(f)

with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

feature_list = np.array(feature_list)
neighbors = NearestNeighbors(
    n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)


@main.route('/', methods=['GET'])
def home():
    return 'success', 200


@main.route('/find_similar', methods=['POST'])
def find_similar():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        feature = extract_features(filepath)
        distances, indices = neighbors.kneighbors([feature])

        similar_files = [filenames[i] for i in indices[0][1:6]]
        # Convert images to base64
        similar_images_base64 = []
        for filename in similar_files:
            with open(filename, "rb") as image_file:
                encoded_string = base64.b64encode(
                    image_file.read()).decode('utf-8')
                similar_images_base64.append(encoded_string)

        return jsonify(similar_images_base64)
