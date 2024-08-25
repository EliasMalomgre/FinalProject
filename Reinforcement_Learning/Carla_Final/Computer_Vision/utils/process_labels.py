"""
Process Labels from https://github.com/martisaju/CARLA-Speed-Traffic-Sign-Detection-Using-Yolo into Pascal VOC format


__author__ = "Bavo Lesy"



split by ' '
Original Labels:
    speed = {0,1,2} with 0 = 30km/h, 1 = 60km/h, 2 = 90km/h
    x_coord between 0 and 1 (*1600)
    y_coord between 0 and 1 (*900)
    width between 0 and 1 (*1600)
    height between 0 and 1 (*900)
Pascal VOC Labels:
    x_min = x_coord - width/2
    x_max = x_coord + width/2
    y_min = y_coord - height/2
    y_max = y_coord + height/2
    classification = {speed_30, speed_60, speed_90}
"""
import shutil

from pascal_voc_writer import Writer
import os

def process(input_path,  output_path):
    """
    Process the labels from the CARLA dataset into Pascal VOC format
    :param input_path: folder containing the original dataset and their labels
    :param output_path: folder to store the Pascal VOC formatted labels
    """
    writer = Writer(input_path + '.jpg', 1600, 1200)
    # open label file from label path
    # the file is in the format of: speed x_coord y_coord width height
    with open(input_path + '.txt', 'r') as file:
        for line in file:
            line = line.split(' ')
            speed = line[0]
            x_coord = float(float(line[1])*1600)
            y_coord = float(float(line[2])*1200)
            width = float(float(line[3])*1600)
            height = float(float(line[4])*1200)
            x_min = x_coord - width/2
            x_max = x_coord + width/2
            y_min = y_coord - height/2
            y_max = y_coord + height/2
            if speed == '0':
                classification = 'speed_30'
            elif speed == '1':
                classification = 'speed_60'
            elif speed == '2':
                classification = 'speed_90'
            writer.addObject(classification, x_min, y_min, x_max, y_max)
    writer.save(output_path + '.xml')
    #also copy the image to the output path
    shutil.copy(input_path + '.jpg', output_path + '.jpg')


if __name__ == '__main__':
    # process all files in a specified folder
    input_path = "C:/Users/Bavo Lesy/Downloads/dataset_CARLA/dataset_CARLA"
    output_path = "C:/Users/Bavo Lesy/Downloads/dataset_CARLA/output"
    for file in os.listdir(input_path):
        # make sure labels are not empty
        if file.endswith('.txt') and os.stat(input_path + '/' + file).st_size != 0:
            process(os.path.join(input_path, file[:-4]), os.path.join(output_path, file[:-4]))
