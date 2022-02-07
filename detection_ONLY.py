import cv2
from object_detection import ObjectDetection


# capture frames from a video
cap = cv2.VideoCapture(r'Low_light.mp4')



val = int(input("Enter value 1 for Haar Cascade or 2 for DNN YOLO model: "))

if val==1:
    # Trained XML classifiers describes some features of some object we want to detect
    car_cascade = cv2.CascadeClassifier(r'haarcascade_car.xml')
else:
    # Initialize Object Detection
    od = ObjectDetection()


#loop runs if capturing has been initialized.
while True:
    ret, frames = cap.read()

    if val == 1:
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        boxes_HAAR = car_cascade.detectMultiScale(gray, 1.2, 3)   #scale factor=1.2 , minNeigbour=3
        boxes = boxes_HAAR
    else:
        (class_ids, scores, boxes_DNN) = od.detect(frames)
        boxes = boxes_DNN


    for (x, y, w, h) in boxes:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Result', frames )

    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
