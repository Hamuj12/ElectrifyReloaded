from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import io
import os
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.getcwd(), 'best.pt')
model = YOLO(model_path)

def prepare_image(image):
    return image.resize((640, 640))

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the base64 image
        base64_image = request.json.get("image")
        image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
        image = prepare_image(image)

        # Predict
        results = model.predict(image)

        # Get bounding boxes and confidences
        bboxes = results[0].boxes.xyxy.tolist()
        confidences = results[0].boxes.conf.tolist()
        classes = results[0].boxes.cls.tolist()

        # Draw the bounding boxes on the image
        draw = ImageDraw.Draw(image)
        for bbox, conf in zip(bboxes, confidences):
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0], bbox[1] - 10), f"{conf:.2f}", fill="red")

        # Convert the image back to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Return the base64 image
        return jsonify({"image": base64_image})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
