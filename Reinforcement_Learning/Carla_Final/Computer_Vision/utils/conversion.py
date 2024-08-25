"""
Various function used to process the dataset

__author__ = "Bavo Lesy"
"""

import os

import numpy as np
import plyfile
import os
import shutil

def convert_2_bin(directory, directory2, k):
    #Convert every file in this folder
    for filename in os.listdir(directory):
        lidar_path = os.path.join(directory, filename)
        plydata = plyfile.PlyData.read(lidar_path)
        lidar = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'], plydata['vertex']['I']]).transpose()
        # flip y axis
        lidar[:, 1] *= -1
        # save to bin file with intensity as color
        # start numbering at 000000.bin and increment
        # Make directory if it doesn't exist
        if not os.path.exists('output/lidar_output/bin'):
            os.makedirs('output/lidar_output/bin')
        lidar.tofile('output/lidar_output/bin/' + str(k).zfill(6) + '.bin')

        # # Rename label filename to match bin files
        # # take out /ply and replace it with /bin
        # label_path = os.path.join(directory2, filename[:-3])+ 'txt'
        # label = open(label_path, 'r')
        # # save file under new name
        # if not os.path.exists('output/lidar_output/labels'):
        #     os.makedirs('output/lidar_output/labels')
        # new_label = open('output/lidar_output/labels/' + str(k).zfill(6) + '.txt', 'w')
        # new_label.write(label.read())
        k += 1
    return k



def parse_over_files(directory, i, j):
    """
    Rename all files in a directory to a new name better suited for training our model
    :param directory: directory to parse
    :param i: number to start at for xml files
    :param j: number to start at for png files
    :return: numbers to start at for next directory
    """
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):

            newname = str(i).zfill(6) + ".xml"
            # if not exist
            i += 1
            if not os.path.exists('output/camera_output/annotations'):
                os.makedirs('output/camera_output/annotations')
            target = r'output/camera_output/annotations/' + newname
            shutil.copyfile(os.path.join(directory, filename), target)

        if filename.endswith(".jpg"):

            newname = str(j).zfill(6) + ".jpg"
            # if not exist
            if not os.path.exists('output/camera_output/images'):
                os.makedirs('output/camera_output/images')
            target = r'output/camera_output/images/' + newname
            shutil.copyfile(os.path.join(directory, filename), target)
            j += 1

    return i, j

def parse_lidar(directory, k):
    """
    Rename all files in a directory to a new name better suited for training our model
    :param directory: directory to parse
    :param k: number to start at for bin files
    :return: numbers to start at for next directory
    """
    for filename in os.listdir(directory):
        # rename all bin files
        newname = str(k).zfill(6) + ".bin"
        # if not exist
        if not os.path.exists('output/lidar_output/bin'):
            os.makedirs('output/lidar_output/bin')
        target = r'output/lidar_output/bin/' + newname
        shutil.copyfile(os.path.join(directory, filename), target)
        k += 1
    return k


def parse_labels(directory, l):
    """
    Rename all lidar label files in a directory to a new name better suited for training our model
    :param directory: directory to parse
    :param l: number to start at for label files
    :return: numbers to start at for next directory
    """
    for filename in os.listdir(directory):
        # make sure it doenst end with _2.txt
        if filename.endswith("_2.txt"):
            # rename all txt files
            newname = str(l).zfill(6) + ".txt"
            # if not exist
            if not os.path.exists('output/lidar_output/labels2'):
                os.makedirs('output/lidar_output/labels2')
            target = r'output/lidar_output/labels2/' + newname
            shutil.copyfile(os.path.join(directory, filename), target)
            l += 1
    return l


def make_file(filename, begin,end):
    """
    Make a file with all the names of the lidar measurements, used for training the SFA3D
    :param filename: name of the file to make
    :param begin: number to start at
    :param end: number to end at
    :return: None
    """
    # make a .txt file with all the names of the files in the directory
    # this should be 00001, 00002, 00003, etc until specified z
    # make a .txt file with all the names of the files in the directory
    # this should be 00001, 00002, 00003, etc until specified z
    if not os.path.exists('output/lidar_output/'):
        os.makedirs('output/lidar_output/')
    # create a text file named train.txt
    train = open('output/lidar_output/' + filename, 'w')
    for i in range(begin, end):
        train.write(str(i).zfill(6) + '\n')
    train.close()

def make_calib_file(n):

    # Copy all of the calib files and rename them
    for filename in os.listdir('SFA3D/dataset/kitti/training/calib'):
        newname = str(n).zfill(6) + ".txt"
        target = r'SFA3D/dataset/kitti/training/calib/' + newname
        shutil.copyfile(os.path.join('SFA3D/dataset/kitti/training/calib/', filename), target)
        n += 1
    return n

def flip_y(directory, l):
    # flip the Y coordinate of all the label files
    for filename in os.listdir(directory):
        # make sure it doenst end with _2.txt
        if filename.endswith(".txt"):
            # rename all txt files
            newname = str(l).zfill(6) + ".txt"
            # if not exist
            if not os.path.exists('output/lidar_output/labels2'):
                os.makedirs('output/lidar_output/labels2')
            target = r'output/lidar_output/labels2/' + newname
            # open the file
            label = open(os.path.join(directory, filename), 'r')
            # check Y coordinate on every line
            new_label = open(target, 'w')
            for line in label:
                line = line.split()
                # Change first line to "Car"
                line[0] = "Car"
                # flip Y coordinate
                line[12] = str(-1 * float(line[12]))
                # flip yaw
                line[14] = str(-1 * float(line[14]))
                # write to new file
                new_label.write(' '.join(line) + '\n')

            l += 1



if __name__ == '__main__':
    i = 8000
    j = 8000
    """
    i, j = parse_over_files('C:/Users/Bavo Lesy/Downloads/pedestrian01-02-04-10/camera_output/Town01', i, j)
    i, j = parse_over_files('C:/Users/Bavo Lesy/Downloads/pedestrian01-02-04-10/camera_output/Town02', i, j)
    i, j = parse_over_files('C:/Users/Bavo Lesy/Downloads/pedestrian01-02-04-10/camera_output/Town04', i, j)
    i, j = parse_over_files('C:/Users/Bavo Lesy/Downloads/pedestrian01-02-04-10/camera_output/Town10HD', i, j)
    i, j = parse_over_files('C:/Users/Bavo Lesy/Downloads/ped03-05-v2/Town03', i, j)
    i, j = parse_over_files('C:/Users/Bavo Lesy/Downloads/ped03-05-v2/Town05', i, j)
    """
    i, j = parse_over_files('C:/Users/Bavo Lesy/OneDrive/Documenten/7e Semester/DAI/Project/dataset/camera/final_traffic_lights/train', i, j)
    print(i, j)
    """
    k = 0
    l = 0
    k = parse_lidar('output/lidar_output/Town01/bin', k)
    l = parse_labels('output/lidar_output/Town01/labels', l)
    k = parse_lidar('output/lidar_output/Town03/bin', k)
    l = parse_labels('output/lidar_output/Town03/labels', l)
    k = parse_lidar('output/lidar_output/Town04/bin', k)
    l = parse_labels('output/lidar_output/Town04/labels', l)
    k = parse_lidar('output/lidar_output/Town10HD/bin', k)
    l = parse_labels('output/lidar_output/Town10HD/labels', l)
    """

    #make_file("train.txt", 0, 3800)
    #make_file("val.txt", 3800, 3982)
    #make_file("test.txt", 0, 3982)
    #make_file("trainval.txt", 0, 3982)
    #make_calib_file(k)
    """
    # n = 1000
    # n = make_calib_file(n)
    # n = make_calib_file(n)
    """




