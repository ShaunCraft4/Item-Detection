'''
Use the below code for training the model with the "data.yaml" file, after which you will get the .pt file mentioned in the code:

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=20, imgsz=64)

'''

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read() 
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Digit Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

