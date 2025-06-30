import os
import cv2
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from glob import glob

# Initialize Flask app
app = Flask(__name__)

# Set folders for uploads and predictions
UPLOAD_FOLDER = 'static/uploads'
PRED_FOLDER = os.path.join(UPLOAD_FOLDER, 'predicted')
os.makedirs(PRED_FOLDER, exist_ok=True)  # Create prediction folder if it doesn't exist

# Load the YOLO model with pre-trained weights
model = YOLO('weights/best.pt')

# Home route - renders the upload page
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']  # Get the uploaded image file
    if file and file.filename != '':
        # Save the uploaded file securely
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # Run YOLO model prediction on the uploaded image
        model.predict(
            source=path,
            conf=0.25,
            save=True,
            project=UPLOAD_FOLDER,
            name='predicted',
            exist_ok=True
        )

        # Find the latest predicted image from the prediction folder
        preds = sorted(glob(os.path.join(PRED_FOLDER, '*.jpg')), key=os.path.getmtime)
        if preds:
            # Prepare image path to be rendered in HTML
            pred_url = '/' + preds[-1].replace('\\', '/')
            return render_template('result.html', image_path=pred_url)

    return "Upload failed", 400  # Return error if upload fails

# Generator function to stream frames from webcam with YOLO predictions
def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run real-time detection on each frame
        results = model.predict(source=frame, stream=True, conf=0.4)
        for r in results:
            frame = r.plot()  # Draw bounding boxes on frame

        # Encode frame and yield to the client browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# Route for streaming webcam feed with detection
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for live webcam page
@app.route('/live')
def live():
    return render_template('live.html')

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
