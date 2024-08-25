"""
Script to test the CameraDetection model on a folder of images

__author__ = "Bavo Lesy"
"""

import Computer_Vision.CameraDetection as CameraDetection
import numpy as np
import cv2
import os
from PIL import Image
import time

def main():

    #init camera detection model
        camera_detection = CameraDetection.CameraDetection()
        #detect on all images in folder
        for image in os.listdir('/Computer_Vision/images'):
            # image as PIL image
            image = Image.open('/Computer_Vision/images/' + image)
            # remove alpha channel
            image = image.convert('RGB')

            t0 = time.perf_counter()
            image = camera_detection.detect(image)
            t1 = time.perf_counter()
            print(f"Time to run inference: {t1 - t0:0.4f} seconds")
            #show image
            #convert image to CV 16 F
            image = np.array(image, dtype=np.uint8)
            cv2.imshow('image', image)
            #wait for keypress
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
