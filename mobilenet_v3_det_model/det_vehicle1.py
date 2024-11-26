import os
import cv2
from flask import Flask, Response, render_template

app = Flask(__name__)

# Threshold to detect objects, which also represent the probability
thres = 0.6

# Load class names from file
classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Paths to the model files
configPath = os.path.abspath("ssd_mobilenet_v3_large_coco_2020_01_14 .pbtxt")
weightsPath = os.path.abspath("frozen_inference_graph.pb")

# Check if files exist
if not os.path.isfile(configPath):
    print(f"Error: Config file {configPath} does not exist.")
    exit()

if not os.path.isfile(weightsPath):
    print(f"Error: Weights file {weightsPath} does not exist.")
    exit()

# Load the DNN model
try:
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(640, 480)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
except cv2.error as e:
    print(f"Error loading model: {e}")
    exit()

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    
    # Debug prints to check return values
    print("classIds:", classIds)
    print("confs:", confs)
    print("bbox:", bbox)
    
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId > 0 and classId <= len(classNames):
                className = classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box, className])
                    if draw:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                print(f"Warning: classId {classId} is out of range")
    return img, objectInfo

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    frame_skip = 0  # Number of frames to skip
    frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Process only every (frame_skip+1)th frame
        if frame_count % (frame_skip + 1) == 0:
            result, objectInfo = getObjects(img, thres, 0.5, objects=["car", "person", "bus", "motorcycle"])

            # Draw detected objects
            for (box, className) in objectInfo:
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
                cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5008)
