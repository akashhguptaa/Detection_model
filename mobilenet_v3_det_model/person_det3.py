import cv2
from flask import Flask, Response, render_template
import psutil  # Import psutil to get system information

app = Flask(__name__)

# Threshold to detect object
thres = 0.55

# Load class names from file
classNames = []
classFile = "ssd_mobilenetv3_model_label.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Paths to the model files
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "mobilenetv3_frozen_inference.pb"

# Load the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    person_count = 0  # Initialize person count
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                if className == "person":
                    person_count += 1  # Increment person count
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo, person_count

def generate_frames():
    # Replace the URL below with your live stream URL
    stream_url = "http://labcam5.local:9081"  # Example: "http://192.168.1.100:8080/video"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        exit()

    frame_skip = 0  # Number of frames to skip
    frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Process only every (frame_skip+1)th frame
        if frame_count % (frame_skip + 1) == 0:
            result, objectInfo, person_count = getObjects(img, thres, 0.4, objects=['person'])

            # Add the person count to the frame
            cv2.putText(img, f'Persons: {person_count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

            # Get CPU and RAM usage
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent

            # Add CPU and RAM usage to the frame
            cv2.putText(img, f'CPU: {cpu_usage}%', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f'RAM: {ram_usage}%', (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

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
    app.run(host='0.0.0.0', port=5003)