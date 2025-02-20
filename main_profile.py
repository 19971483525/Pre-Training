import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # choose your yaml file
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml')
    model.info(detailed=False)
    model.profile(imgsz=[640, 640])
    model.fuse()