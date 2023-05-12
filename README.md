# Electrify Reloaded

Electrify Reloaded is an advanced resistor recognition system powered by computer vision and deep learning. It allows you to quickly and accurately identify and analyze resistors in images, providing valuable insights for various applications in electronics and electrical engineering.

## Key Features

- **Resistor Detection**: The system employs state-of-the-art deep learning models to detect and locate resistors in images. It accurately identifies the position and size of resistors, even in complex scenes.

- **Bounding Box Visualization**: Electrify Reloaded provides visualizations of bounding boxes around detected resistors, making it easy to understand the location and extent of each resistor in an image.

- **Crop Resistor Images**: The system automatically crops and extracts individual resistor images from the original scene. This functionality enables further analysis and processing of individual resistors, such as applying specialized algorithms or feeding them into subsequent models.

- **Confidence Scores**: Electrify Reloaded assigns confidence scores to each detected resistor, indicating the model's confidence level in its predictions. This allows you to assess the reliability of the results and make informed decisions based on the level of confidence.

- **React Native App**: The system includes a user-friendly mobile application developed using React Native. It provides an intuitive interface for users to select images, run the inference process, visualize the results, and explore the cropped resistor images.

## Technologies Used

- **Ultralytics YOLO**: Electrify Reloaded leverages the powerful YOLO (You Only Look Once) object detection framework provided by Ultralytics. It offers fast and accurate object detection capabilities, making it ideal for real-time applications.

- **Flask**: The server-side of the application is built using Flask, a lightweight and flexible web framework for Python. I handle image processing requests, run the inference, and return the results to the client.

- **PIL (Python Imaging Library)**: The PIL library is used for image manipulation tasks, such as drawing bounding boxes, cropping images, and saving the processed images.

- **Expo Image Picker**: The Expo Image Picker library is utilized in the React Native app to provide an easy-to-use interface for users to select images from their device's gallery.

## Getting Started

To get started with Electrify Reloaded, follow these steps:

1. Clone the repository to your local machine.
2. Set up the Python environment with the required dependencies specified in the `requirements.txt` file.
3. Install the necessary npm packages by running `npm install` in the `app` directory of the project.
4. Start the Flask server by running `python app.py` in the root directory.
5. Launch the React Native app on your device using Expo by running `npm start` in the `app` directory.
6. Use the app to select an image from the gallery and initiate the inference process.
7. Explore the results, including the original image with bounding boxes and confidence scores, as well as the cropped resistor images.

## Contributing

I welcome contributions to Electrify Reloaded! If you have any ideas, improvements, or bug fixes, feel free to submit a pull request. Please make sure to follow the established coding style and guidelines.

## License

Electrify Reloaded is licensed under the MIT License. Feel free to use, modify, and distribute the code for both commercial and non-commercial purposes.

## Acknowledgements

I would like to acknowledge the contributions of the open-source community and the authors of the libraries and frameworks used in this project. Their efforts have made it possible to create a robust and efficient resistor recognition system.

