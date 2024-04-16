from flask import Flask, request, jsonify, render_template
from Cracked_Detection.pipeline.training_pipeline import TrainPipeline
from Cracked_Detection.utils.main_utils import decodeImage, encodeImageIntoBase64
from ultralytics import YOLO  
import base64
import os,glob,time,re
import shutil
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

@app.route("/")
def home():
    return render_template("index.html")


# For training pipeline

@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image found in request'}), 400

    image = request.json['image']
    filename = "data/inputImage.jpg"
    decodeImage(image, filename)

    # Run YOLO prediction command
    model = YOLO('runs/segment/train2/weights/best.pt')
    # Predict using YOLO
    results = model.predict(source=filename,save=True)

    parent_directory = "runs/segment/"
    directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    print(directories)
    pattern = "predict"
    filtered_directories = [d for d in directories if d.startswith(pattern) and d[len(pattern):].isdigit()]
    latest_directory = max(filtered_directories, key=lambda x: int(re.search(r'\d+$', x).group()))
    print(latest_directory)
    image_path = os.path.join(parent_directory, latest_directory, "inputImage.jpg").replace('\\', '/')

    # Encode the resulting image into base64
    opencodedbase64 = encodeImageIntoBase64(image_path)

    result = {"image": opencodedbase64.decode('utf-8')}
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
