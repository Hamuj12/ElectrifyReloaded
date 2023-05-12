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

        cropped_images = []
        for bbox in bboxes:
            # Crop the image with preserved aspect ratio
            xmin, ymin, xmax, ymax = bbox
            width, height = xmax - xmin, ymax - ymin
            aspect_ratio = width / height
            
            cropped = image.crop(bbox)
            cropped.thumbnail((640, 640/aspect_ratio))  # Adjust size as needed, preserving aspect ratio

            # Rotate the image if needed
            if cropped.height > cropped.width:
                cropped = cropped.rotate(90, expand=True)

            buffered = io.BytesIO()
            cropped.save(buffered, format="JPEG")
            cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            cropped_images.append(cropped_image_base64)

        # Draw the bounding boxes and confidence scores on the image
        draw = ImageDraw.Draw(image)
        for bbox, conf in zip(bboxes, confidences):
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0], bbox[1] - 10), f"{conf:.2f}", fill="red")

        # Convert the final image to base64
        final_image_buffered = io.BytesIO()
        image.save(final_image_buffered, format="JPEG")
        final_image_base64 = base64.b64encode(final_image_buffered.getvalue()).decode('utf-8')

        # Return the base64 images
        return jsonify({"image": final_image_base64, "cropped_images": cropped_images})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
