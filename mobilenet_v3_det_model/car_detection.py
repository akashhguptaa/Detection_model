import cv2

# Load the pre-trained Haar Cascade classifier for car detection
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Initialize video capture from the default webcam (0 is typically the default)
video = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    # If the frame is not captured correctly, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert the frame to grayscale (Haar cascades require grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars in the grayscale frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the resulting frame with detections
    cv2.imshow('Car Detection from Webcam', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
