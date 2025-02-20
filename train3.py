from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-CA.yaml')
    model.train(data='dataset2/2.yaml',
                cache=False,
                resume=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=10,
                workers=4,
                amp = False, 
                device='0,1',
                optimizer='SGD', # using SGD
                name = 'YOLOv8n_CA_voc',
                )