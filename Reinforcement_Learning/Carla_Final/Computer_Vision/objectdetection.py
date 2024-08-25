"""
Real-Time object detection in CARLA

__author__ = "Bavo Lesy"
"""

import logging
import os
import queue
import random
import time
import cv2
import numpy as np
import carla

import LidarDetection

from Computer_Vision.utils.utils_datageneration import generate_traffic


def main(model_path, town, num_vehicles, num_frames, ):
    # Setup world and spawn ego
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    #world = client.get_world()
    world = client.load_world(town)
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    ego = world.spawn_actor(bp, random.choice(spawn_points))
    ego.set_autopilot(True)

    # Sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05  # (20fps)
    world.apply_settings(settings)

    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # Spawn lidar
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '80')
    lidar_bp.set_attribute('rotation_frequency', '20')
    # fov
    lidar_bp.set_attribute('points_per_second', str(64 / 0.00004608))
    lidar_bp.set_attribute('upper_fov', str(2))
    lidar_bp.set_attribute('lower_fov', str(-24.8))
    # lidar_bp.set_attribute('horizontal_fov', str(360))
    # lidar_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    lidar = world.spawn_actor(lidar_bp,
                              carla.Transform(carla.Location(x=0, y=0, z=1.8), carla.Rotation(pitch=0, yaw=0, roll=0)),
                              attach_to=ego)

    # Create a queue to store and retrieve the sensor data
    lidar_queue = queue.Queue()
    lidar.listen(lidar_queue.put)


    world.tick()
    image = image_queue.get()
    lidar_data = lidar_queue.get()

    # Generate traffic
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    vehicles_list = generate_traffic(traffic_manager, client, blueprint_library, spawn_points, num_vehicles)

    # Load lidar model
    lidar_model = LidarDetection.Lidar()
    visualize = True # used to visualize the lidar detection

    # Main Game Loop, run until we reach the desired number of frames
    while image.frame < num_frames:
        world.tick()
        # Get image

        image = image_queue.get()
        # Get image as numpy array with shape (600, 800, 3)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        reshaped_array = np.reshape(array, (image.height, image.width, 4))
        array = reshaped_array[:, :, :3]
        array = array[:, :, ::-1]

        lidar_data = lidar_queue.get()
        if lidar_data.frame %  30 == 0:

            # save lidar data in .ply
            #if not os.path.exists('output/lidar_output/'+ town + '/ply/'):
            #    os.makedirs('output/lidar_output/'+ town + '/ply/')
            #lidar_path = 'output/lidar_output/' + town + '/ply/' + '%06d' % lidar_data.frame + '.ply'
            #lidar_data.save_to_disk(lidar_path)



            # Perform detection
            t0 = time.perf_counter()
            distance, img = lidar_model.get_distance(lidar_data, visualize)
            t1 = time.perf_counter()
            print(f"Time to run inference: {t1 - t0:0.4f} seconds")
            print(distance)

        # Display image
        # convert cv2 image to rbg image
        #cv2_im = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        #cv2.imshow('RaceAI RT-Classification',cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    print("Starting")

    model_path = 'models/saved_model_v1/'
    num_vehicles = 75
    num_frames = 10000

    main(model_path, "Town10HD", num_vehicles, num_frames)
