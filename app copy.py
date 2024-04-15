from flask import Flask, request, jsonify, render_template
from Cracked_Detection.pipeline.training_pipeline import TrainPipeline
from ultralytics import YOLO  
import base64
import os
import shutil

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def decodeImage(encoded_image, filename):
    with open(filename, "wb") as fh:
        fh.write(base64.b64decode(encoded_image))

def get_latest_predict_folder():
    predict_folders = [folder for folder in os.listdir('runs/segment') if folder.startswith('predict')]
    if not predict_folders:
        return None
    
    predict_numbers = [int(folder.replace('predict', '')) for folder in predict_folders]
    latest_number = max(predict_numbers)
    latest_folder = f"predict{latest_number}"
    return os.path.join('runs/segment', latest_folder)



def move_and_encode_latest_image_into_base64():
    latest_predict_folder = get_latest_predict_folder()
    
    if latest_predict_folder:
        input_image_path = os.path.join(latest_predict_folder, 'inputImage.jpg')
        with open(input_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        os.remove(input_image_path)
        # Delete only the predict folder inside runs/segment
        shutil.rmtree(latest_predict_folder)
        return encoded_string
    else:
        return None



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

    model = YOLO('runs/segment/train2/weights/best.pt')
    results = model.predict(source=filename, save=True)

    opencodedbase64 = move_and_encode_latest_image_into_base64()

    result = {"image": opencodedbase64.decode('utf-8')}

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
