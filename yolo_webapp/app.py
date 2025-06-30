import os
import cv2
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from glob import glob

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PRED_FOLDER = os.path.join(UPLOAD_FOLDER, 'predicted')
os.makedirs(PRED_FOLDER, exist_ok=True)

model = YOLO('weights/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        model.predict(source=path, conf=0.25, save=True, project=UPLOAD_FOLDER, name='predicted', exist_ok=True)

        preds = sorted(glob(os.path.join(PRED_FOLDER, '*.jpg')), key=os.path.getmtime)
        if preds:
            pred_url = '/' + preds[-1].replace('\\', '/')
            return render_template('result.html', image_path=pred_url)

    return "Upload failed", 400

def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, stream=True, conf=0.4)
        for r in results:
            frame = r.plot()
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('live.html')

if __name__ == '__main__':
    app.run(debug=True)
