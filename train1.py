########Train##########################################

# from ultralytics import YOLO
# 即 ultralytics文件夹 所在绝对路径
# if __name__ == '__main__':
# # Load a model
# # model = YOLO('yolov8n.yaml')  # build a new model from YAML
# # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#     model = YOLO('yolov8t.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

#     # Train the model
#     model.train(data='1.yaml', epochs=100, imgsz=640)

# ########Val##########################################

if __name__ == '__main__':
# Load a model
    from ultralytics import YOLO
    model = YOLO('ultralytics/cfg/models/v8/yolov8s-cls.yaml').load("yolov8s-cls.pt")  # load an official model
    model.train(data='dataset1', 
                epochs=200, 
                imgsz=224, 
                batch=32, 
                device='1', 
                workers=4, 
                optimizer='SGD', 
                project='runs/train2',
                )