# Real-Time Object Detection

A lightweight real-time object detection system using **YOLOv8**, **Flask**, and a mobile camera as the video source. The app displays a live video stream in the browser, detects allowed objects, draws bounding boxes, and shows stats like FPS and object counts. It also provides browser-based speech alerts using the **Web Speech API**.

---

## ✅ Features
- Real-time object detection with YOLOv8  
- Works with Android IP Webcam app  
- Web-based live video stream  
- Bounding boxes and confidence scores  
- Object count and FPS statistics  
- Browser speech alerts for detected objects  
- Easy to modify detection source and allowed classes

---

## 🔧 Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:
```
ultralytics==8.3.195
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
opencv-python==4.11.0.86
numpy==1.26.4
Flask==2.3.0
flask-cors==3.0.10
Pillow==11.3.0
```

*(pyttsx3 and gtts were removed because speech is handled in the browser)*

---

## 📱 Setup: Android Camera (IP Webcam App)

1. Install **IP Webcam** app from the Play Store.  
2. Open the app → click the **three dots (menu)**.  
3. Tap **Start server** at the bottom.  
4. A URL will appear (e.g. `http://192.168.1.4:8080/video`).  
5. Copy this URL.  
6. Replace the `source` in `app.py`:

```python
source="http://YOUR_LOCAL_IP:8080/video"
```

7. From the app, go to **Actions → Run in Background**  
   to keep the camera active after exiting.

---

## ⚙️ Optional Camera Tweaks
You can configure quality, FPS, resolution, and orientation:
- Open the video URL in your browser.
- Modify settings from the app interface.
- If the stream lags, reduce FPS or resolution from the app settings.

---

## ▶️ How to Run the App

```bash
python app.py
```

Then open the browser at:

```
http://localhost:5000
```

The page will show:
- Live video feed
- Bounding boxes
- Detected object counts
- Automatic voice alerts

---

## 🧠 Project Idea

This system demonstrates:
- Real-time AI inference on live camera input
- Lightweight browser-based visualization
- Speech interaction without extra Python libraries

You can use it for:
- Security monitoring  
- Counting objects or people  
- Voice alerts in restricted zones  
- Smart IoT / surveillance projects

---

## 🚀 Future Improvements
Here are some possible extensions:

1. **Depth Estimation Integration**  
   Use a model like MiDaS to estimate distance.  
   Example: increase alert volume when the object gets closer.

2. **Tracking Objects Over Time**  
   Add object tracking (e.g., ByteTrack, DeepSORT).

3. **Save Frames or Alerts**  
   Store violations or detections with timestamps.

4. **Multi-Camera Support**  
   Add multiple video feeds and switch between them.

5. **Custom Models**  
   Train YOLOv8 on custom classes and plug in easily.

---

## 📁 File Structure (Example)
```
project/
│
├── app.py
├── utils.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

---

## ✅ Ready to Use
Just update the IP camera URL in `app.py`, run the Flask server, and start detecting in real time.

Feel free to add images or GIFs later for better documentation.
