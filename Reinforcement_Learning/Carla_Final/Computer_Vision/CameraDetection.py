"""
Class CameraDetection uses a Pytorch YOLOv8 model to detect objects in a camera feed.
The model is trained on our custom dataset

__author__ = "Bavo Lesy"
source: https://learnopencv.com/ultralytics-yolov8/
"""

from Carla_Final.CONFIG import CONFIG
from Carla_Final.Computer_Vision.utils.yolov8 import YOLOv8
from Computer_Vision.utils.yolo_visualization import np


class CameraDetection():
    def __init__(self):
        self.device = 'cuda:0'
        self.iou = 0.5
        self.conf = 0.5
        self.model = YOLOv8(CONFIG['detection_model_folder'], conf_thres=self.conf, iou_thres=self.iou)


    def detect(self, image):
        """
        Run object detection
        :param image: The image to detect objects on
        :return: The image with detections, whether a red light is detected, the speed limit that is detected
        """
        image = np.asarray(image)
        boxes, scores, class_ids = self.model(image)
        combined_img = self.model.draw_detections(image)
        red_light_conf = 0.0
        green_light_conf = 0.0
        speed_limit_conf_30 = 0.0
        speed_limit_conf_60 = 0.0
        speed_limit_conf_90 = 0.0
        speed_limit = -1

        for i in range(len(boxes)):
            if class_ids[i] == 6:  # red light
                if red_light_conf < scores[i]:
                    red_light_conf = scores[i]
            elif class_ids[i] == 8:  # green light
                if green_light_conf < scores[i]:
                    green_light_conf = scores[i]
            elif class_ids[i] == 9:  # speed limit 30
                if speed_limit_conf_30 < scores[i]:
                    speed_limit_conf_30 = scores[i]
            elif class_ids[i] == 10:  # speed limit 60
                if speed_limit_conf_60 < scores[i]:
                    speed_limit_conf_60 = scores[i]
            elif class_ids[i] == 11:  # speed limit 90
                if speed_limit_conf_90 < scores[i]:
                    speed_limit_conf_90 = scores[i]

        if speed_limit_conf_30 > speed_limit_conf_90 and speed_limit_conf_30 > speed_limit_conf_60:
            speed_limit = 30
        elif speed_limit_conf_60 > speed_limit_conf_90 and speed_limit_conf_60 > speed_limit_conf_30:
            speed_limit = 60
        elif speed_limit_conf_90 > speed_limit_conf_60 and speed_limit_conf_90 > speed_limit_conf_30:
            speed_limit = 90

        return combined_img, red_light_conf > green_light_conf, speed_limit
