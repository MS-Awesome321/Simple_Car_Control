from ultralytics import YOLO

model = YOLO("/Users/mayanksengupta/Desktop/CV_Training/yolo11n-seg.pt")  
model.train(data="/Users/mayanksengupta/Desktop/CV_Training/CV Training.v1i.yolov11/data.yaml")