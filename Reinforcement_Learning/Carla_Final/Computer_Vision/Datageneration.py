"""
Generate Carla Dataset (images with pascal VOC labels and lidar with SFA3D labels)
for a given town for a given number of frames

__author__ = "Bavo Lesy"

used code from CARLA tutorial for bounding boxes: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
"""

import os
import queue
import cv2
from pascal_voc_writer import Writer

from Computer_Vision.utils.utils_datageneration import *


def main(town, num_of_vehicles, num_of_walkers, num_of_frames):
    # Simulator
    global semantic_list
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.load_world(town)
    # world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    blueprint = blueprint_library.filter('model3')

    ego = world.spawn_actor(blueprint[0], random.choice(spawn_points))

    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego)
    ego.set_autopilot(True)

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05  # (20fps)
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # Spawn liDar
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
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, y=0, z=1.8), carla.Rotation(pitch=0, yaw=0, roll=0)), attach_to=ego)
    # Create a queue to store and retrieve the sensor data
    lidar_queue = queue.Queue()
    lidar.listen(lidar_queue.put)
    #Semantic lidar
    sem_lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    sem_lidar_bp.set_attribute('channels', '32')
    sem_lidar_bp.set_attribute('range', '80')
    sem_lidar_bp.set_attribute('rotation_frequency', '20')
    # fov
    sem_lidar_bp.set_attribute('points_per_second', str(64 / 0.00004608 * 2))
    sem_lidar_bp.set_attribute('upper_fov', str(2))
    sem_lidar_bp.set_attribute('lower_fov', str(-24.8))

    sem_lidar = world.spawn_actor(sem_lidar_bp, carla.Transform(carla.Location(x=0, y=0, z=1.8), carla.Rotation(pitch=0, yaw=0, roll=0)), attach_to=ego)
    sem_lidar_queue = queue.Queue()
    sem_lidar.listen(sem_lidar_queue.put)
    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)

    # Get the bounding boxes from traffic lights used later for red light detection
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    vehicles_list = generate_traffic(traffic_manager, client, blueprint_library, spawn_points, num_of_vehicles)
    # Spawn pedestrians and also detect the bounding boxes

    # Detect traffic Lights bounding boxes

    world.tick()
    image = image_queue.get()
    pointcloud = lidar_queue.get()

    # Reshape the raw data into an RGB array
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # Reshape pointcloud data
    #lidar_data = np.frombuffer(pointcloud.raw_data, dtype=np.dtype('f4'))

    # Display the image in an OpenCV display window
    cv2.namedWindow('CARLA RaceAI', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('CARLA RaceAI', img)
    cv2.waitKey(1)
    i = 50
    boxes = []
    try:
        ### Game loop ###
        while image.frame < num_of_frames:
            # Retrieve and reshape the image
            world.tick()
            image = image_queue.get()
            pointcloud = lidar_queue.get()
            sem_pointcloud = sem_lidar_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # Get the camera matrix
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            # only take measurements every 50 frames
            if image.frame % 30 == 0:
                i = 0
                # Save the image -- for export
                # Initialize the exporter
                boxes = []
                sem_lidar_data = np.frombuffer(sem_pointcloud.raw_data, dtype=np.dtype(
                    [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('cos', 'f4'), ('index', 'u4'), ('semantic', 'u4')]))
                points = np.array([sem_lidar_data[:]['index'], sem_lidar_data[:]['semantic']])
                mask = np.array([sem_lidar_data[:]['index'], sem_lidar_data[:]['semantic']])[:][1] == 10
                semantic_list = np.unique(points[0][mask])
                for npc in world.get_actors():
                    # Filter out the ego vehicle
                    if npc.id != ego.id and npc.id in vehicles_list and npc.id in semantic_list:
                        bb = npc.bounding_box
                        dist = npc.get_transform().location.distance(ego.get_transform().location)

                        # Filter for the vehicles within 50m
                        if 0.5 < dist < 60:
                            forward_vec = ego.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - ego.get_transform().location

                            if forward_vec.dot(ray) > 1:
                                p1 = get_image_point(bb.location, K, world_2_camera)
                                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                x_max = -10000
                                x_min = 10000
                                y_max = -10000
                                y_min = 10000

                                for vert in verts:
                                    p = get_image_point(vert, K, world_2_camera)
                                    # Find the rightmost vertex
                                    if p[0] > x_max:
                                        x_max = p[0]
                                    # Find the leftmost vertex
                                    if p[0] < x_min:
                                        x_min = p[0]
                                    # Find the highest vertex
                                    if p[1] > y_max:
                                        y_max = p[1]
                                    # Find the lowest vertex
                                    if p[1] < y_min:
                                        y_min = p[1]
                                name = npc.type_id.split('.')[2]
                                classification = 'car'
                                if name == 'ambulance' or name == 'firetruck' or name == 'charger_police' or name == 'charger_police_2020':
                                    classification = 'emergency'
                                elif name == 'crossbike' or name == 'low_rider' or name == 'ninja' or name == 'zx125' or name == 'yzf':
                                    classification = 'motorcycle'
                                elif name == 'omafiets':
                                    classification = 'bicycle'
                                elif name == 'sprinter' or name == 'carlacola':
                                    classification = 'van'
                                    # Add the object to the frame (ensure it is inside the image)
                                x_min = np.clip(x_min, 0, image_w)
                                x_max = np.clip(x_max, 0, image_w)
                                y_min = np.clip(y_min, 0, image_h)
                                y_max = np.clip(y_max, 0, image_h)
                                if x_min != x_max and y_min != y_max:
                                    boxes.append([x_min, y_min, x_max, y_max, classification])
            i += 1
            if i == 3:
                # Compare the bounding boxes to every other bounding box
                # Filter out bad boxes
                for box in boxes:
                    for other_box in boxes:
                        # If the boxes are the same, skip
                        if box != other_box:
                            # Check if box is fully contained in other_box
                            if other_box[0] <= box[0] and other_box[1] <= box[1] and other_box[2] >= box[2] and \
                                    other_box[3] >= box[3]:
                                # If the box is fully contained, remove it
                                boxes.remove(box)
                                break
                image_path = 'output/camera_output/' + town + '/' + '%06d' % image.frame
                image.save_to_disk(image_path + '.png')
                writer = Writer(image_path + '.png', image_w, image_h)
                for box in boxes:
                    cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255, 255), 1)

                    writer.addObject(box[4], box[0], box[1], box[2], box[3])

                    # Save the bounding boxes in the scene
                writer.save(image_path + '.xml')
                cv2.imshow('CARLA RaceAI', img)
                # save the image
                if not os.path.exists('output/camera_output/' + town + '/bbox'):
                    os.makedirs('output/camera_output/' + town + '/bbox/')
                cv2.imwrite('output/camera_output/' + town + '/bbox/' + str(image.frame) + '.png', img)

            # Save liDAR data and create 3D bounding boxes
            if (pointcloud.frame % 30) - 2 == 0:
                # get location from the lidar sensor
                lidar_location = lidar.get_transform().location
                lidar_transform = lidar.get_transform()
                labels = []
                inv_transform = lidar_transform.get_inverse_matrix()
                for npc in world.get_actors():
                    # Filter out the ego vehicle
                    if npc.id != ego.id and npc.id in vehicles_list and npc.id in semantic_list:
                        transform = npc.get_transform()
                        if lidar_location.distance(transform.location) < 50:
                            # Get the bounding box of the vehicle
                            bounding_box = npc.bounding_box
                            # Get the corners of the bounding box
                            corners = bounding_box.extent
                            # Get the rotation of the vehicle
                            rotation = transform.rotation
                            # get the location of the vehicle
                            location = bounding_box.location
                            location = applyTransform(inv_transform, location)
                            # get type of the vehicle
                            name = npc.type_id.split('.')[2]
                            classification = 'car'
                            if name == 'ambulance' or name == 'firetruck' or name == 'charger_police' or name == 'charger_police_2020':
                                classification = 'emergency'
                            elif name == 'crossbike' or name == 'low_rider' or name == 'ninja' or name == 'zx125' or name == 'yzf':
                                classification = 'motorcycle'
                            elif name == 'omafiets':
                                classification = 'bicycle'
                            elif name == 'sprinter' or name == 'carlacola':
                                classification = 'van'
                            labels.append({
                                'type': "Car",
                                'truncated': 0,
                                'occluded': 0,
                                'alpha': 0,
                                'xmin': 0,
                                'ymin': 0,
                                'xmax': 0,
                                'ymax': 0,
                                'height': round(corners.z * 2, 2),
                                'width': round(corners.y * 2, 2),
                                'length': round(corners.x * 2, 2),
                                'x': round(location.x, 2),
                                'y': -round(location.y, 2),
                                'z': round(location.z, 2),
                                'yaw': -np.radians(rotation.yaw - lidar_transform.rotation.yaw),

                            })
                #make directory for the pointclouds
                if not os.path.exists('output/lidar_output/' + town + '/labels'):
                    os.makedirs('output/lidar_output/' + town + '/labels')
                if not os.path.exists('output/lidar_output/' + town + '/data'):
                    os.makedirs('output/lidar_output/' + town + '/data')

                with open('output/lidar_output/' + town + '/labels/' + '%06d' % pointcloud.frame + '.txt', 'w') as f:
                    for label in labels:
                        f.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % (
                            label['type'], label['truncated'], label['occluded'], label['alpha'], label['xmin'],
                            label['ymin'], label['xmax'], label['ymax'], label['height'], label['width'], label['length'],
                            label['x'], label['y'], label['z'], label['yaw']))

                lidar = np.frombuffer(pointcloud.raw_data, dtype=np.float32)
                lidar = lidar.reshape(-1, 4).astype(dtype=np.float32)
                # flip y
                lidar[:, 1] *= -1
                # save lidar data
                lidar_path = 'output/lidar_output/' + town + '/data/' + '%06d' % pointcloud.frame + '.bin'
                lidar.tofile(lidar_path)
                if cv2.waitKey(1) == ord('q'):
                    break
    finally:
        # Destroy the actors
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()
        for actor in world.get_actors().filter('sensor.*'):
            actor.destroy()
        print('All actors destroyed.')

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Measurement every 50 frames, we want 400 measurement per town, so 30 * 400 = 16000+ 4000 for bad measurements
    # TO DO: change weather dynamically for each town

    frames = 20000
    num_vehicle = 75
    num_pedestrian = 30
    #main('Town04', num_vehicle, num_pedestrian, frames)
    main('Town10HD', num_vehicle, num_pedestrian, frames)
    main('Town01', num_vehicle, num_pedestrian, frames)
    main('Town02', num_vehicle, num_pedestrian, frames)
    main('Town03', num_vehicle, num_pedestrian, frames)
    main('Town05', num_vehicle, num_pedestrian, frames)



