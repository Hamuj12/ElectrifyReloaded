from flask import Flask, request, jsonify
from PIL import Image
import io
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.getcwd(), 'best.pt')
model = YOLO(model_path)

def prepare_image(image):
    return image.resize((640, 640))

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        print("POST request received")
        print(request.files)
        if request.files.get("image"):
            image = request.files["image"]
            filename = secure_filename(image.filename)  # Use the secure_filename function to ensure a safe filename
            image.save(filename)  # Save the image to the server's file system
            print("image received: " + filename)
            image = Image.open(filename)
            image = prepare_image(image)

            # Predict
            results = model.predict(image)

            # Get bounding boxes and confidences
            bboxes = results[0].boxes.xyxy.tolist()
            print("bboxes: " + str(bboxes))
            confidences = results[0].boxes.conf.tolist()
            print(str(confidences))
            classes = results[0].boxes.cls.tolist()
            print(str(classes))

            # Prepare the response
            data = {"predictions": []}
            for (box, confidence, cls) in zip(bboxes, confidences, classes):
                r = {"bbox": box, "confidence": confidence, "class": cls}
                data["predictions"].append(r)

            os.remove(filename)  # Delete the image from the server's file system

            # Return the data dictionary as a JSON response
            return jsonify(data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
