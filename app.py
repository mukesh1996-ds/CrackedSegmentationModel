from flask import Flask, request, jsonify, render_template
from Cracked_Detection.pipeline.training_pipeline import TrainPipeline
from Cracked_Detection.utils.main_utils import decodeImage, encodeImageIntoBase64
from ultralytics import YOLO  
import base64
import os,glob,time,re
import shutil
from flask_cors import CORS, cross_origin

def calculate_damage_percentage(results, image_width, image_height):
    total_area = image_width * image_height
    damaged_area = 0

    for result in results:
        if len(result) >= 4:  # Ensure the result contains at least 4 elements (x, y, width, height)
            box_width = result[2]
            box_height = result[3]
            damaged_area += box_width * box_height

    percentage_damage = (damaged_area / total_area) * 100
    return percentage_damage


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
    results = model.predict(source=filename, save=True)
    print(results)
    # Check if any cracks were detected
    if results:
        crack_detected = True
    else:
        crack_detected = False

    # Calculate the percentage of images with cracks
    total_images = 1  # Assuming only one image is processed per request
    images_with_cracks = 1 if crack_detected else 0
    percentage_cracked = (images_with_cracks / total_images) * 100
    print(percentage_cracked)
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

    result = {"image": opencodedbase64.decode('utf-8'), "percentage_cracked": percentage_cracked}
    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)
