# app.py
from flask import Flask, render_template, Response, jsonify
from utils import load_model, generate_frames, get_latest_stats

app = Flask(__name__)

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(
            model,
            #source="http://192.168.1.4:8080/video" # Change to your IP camera URL or 0 for webcam   
            source=0 # Use 0 for webcam
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stats')
def stats():
    return jsonify(get_latest_stats())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
