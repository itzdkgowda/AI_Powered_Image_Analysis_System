# AI-powered Image Analysis System

This project is a web-based application that performs real-time object detection on images and video streams using YOLOv8. Built with Flask, OpenCV, and Ultralytics' YOLOv8, it supports both image uploads and live webcam detection through a simple UI.

## 🚀 Features

- Upload an image and receive detection results with bounding boxes
- Run real-time object detection using your webcam
- Detect custom classes trained with YOLOv8 (e.g., phone, mouse, steel bottle, plastic bottle)
- Simple web interface using Flask and HTML/CSS
- Fully modular and extensible for future upgrades like segmentation, analytics, or deployment

## 🖼️ Interface

- 📁 **Upload Image**: Send an image to the server and receive annotated results
- 🎥 **Live Detection**: Activate your webcam and see objects detected in real-time


🖥️ 1. Set Up Your Environment
git clone 
crete virtual environmen
Make sure you're inside your virtual environment (e.g., tfod-env) and your project has this structure:

Install the dependencies:
pip install -r requirements.txt

🚀 2. Run the Web App (Detection via Upload & Live Cam)

python app.py

Then open http://localhost:5000 in your browser.

Click 📁 Upload Image to analyze any photo

Click 🎥 Live Detection to activate your webcam

> This runs using the trained YOLOv8 weights and gives real-time results.



