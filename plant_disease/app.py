from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import json
import joblib

app = Flask(__name__)

# Paths to model and JSON directories
MODEL_DIR = "Models"
JSON_DIR = "json_files"

# Mapping for model file names and label encoders
MODEL_MAPPING = {
    "Tomato": {
        "ResNet": ("ResNet_model.h5", "tomato_disease_info.json", (224, 224)),
        "DenseNet": ("DenseNet_model.h5", "tomato_disease_info.json", (224, 224)),
        "Inception": ("Inception_model.h5", "tomato_disease_info.json", (299, 299)),
        "InceptionResNet": ("InceptionResNetV2_model.h5", "tomato_disease_info.json", (299, 299)),
        "VGG": ("VGG_model.h5", "tomato_disease_info.json", (224, 224))
    },
    "Potato": {
        "MobileNet": ("potato_mobilenetv2.h5", "potato_disease_info.json", (224, 224)),
        "ResNet": ("potato_resnet50.h5", "potato_disease_info.json", (224, 224)),
        "SVM": ("potato_svm_model.pkl", "potato_disease_info.json", "potato_label_encoder.pkl", (224, 224)),
        "Random Forest": ("potato_random_forest_model.pkl", "potato_disease_info.json", "potato_label_encoder.pkl", (224, 224))
    },
    "Sugarcane": {
        "MobileNet": ("sugarcane_mobilenet_model.h5", "sugarcane_disease_info.json", (224, 224)),
        "ResNet": ("sugarcane_resnet_model.h5", "sugarcane_disease_info.json", (224, 224)),
        "SVM": ("sugarcane_svm_model.pkl", "sugarcane_disease_info.json", "sugarcane_label_encoder.pkl", (100, 100)),
        "Random Forest": ("sugarcane_rf_model.pkl", "sugarcane_disease_info.json", "sugarcane_label_encoder.pkl", (100, 100))
    }
}

def preprocess_image(image_file, target_size):
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/team")
def team():
    return render_template("teams.html")


@app.route("/predict", methods=["POST"])
def predict():
    plant = request.form["dataset"]
    model_name = request.form["model"]
    image_file = request.files["image"]

    model_info = MODEL_MAPPING[plant][model_name]
    model_path = os.path.join(MODEL_DIR, model_info[0])
    json_path = os.path.join(JSON_DIR, model_info[1])

    if model_path.endswith(".h5"):
        target_size = model_info[2]
        model = load_model(model_path)
        image = preprocess_image(image_file, target_size)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        with open(json_path) as f:
            disease_info = json.load(f)
        label = list(disease_info.keys())[predicted_class]
    else:
        target_size = model_info[3]
        model = joblib.load(model_path)
        image = preprocess_image(image_file, target_size)
        image = image.reshape(-1, target_size[0] * target_size[1] * 3)
        label_encoder_path = os.path.join(MODEL_DIR, model_info[2])
        label_encoder = joblib.load(label_encoder_path)
        prediction = model.predict(image)
        label = label_encoder.inverse_transform(prediction)[0]
        with open(json_path) as f:
            disease_info = json.load(f)

    result = disease_info[label]
    return render_template("result.html", label=label, result=result, disease_info=result)

if __name__ == "__main__":
    app.run(debug=True)
