"""
Util functions for generating data in CARLA

__author__ = "Bavo Lesy"

source: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
"""

import logging

import numpy as np
import carla
import random
import math



def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


# Calculate 2D projection of 3D coordinate
def get_image_point(loc, K, w2c):
    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]


def generate_traffic(traffic_manager, client, blueprint_library, spawn_points, num_vehicles):
    """Generate traffic"""
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_random_device_seed(0)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.global_percentage_speed_difference(30.0)
    vehicle_bp = blueprint_library.filter('vehicle.*')
    # only four wheels
    vehicle_bp = [x for x in vehicle_bp if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = spawn_points
    number_of_spawn_points = len(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch = []
    vehicles_list = []

    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles:
            break
        blueprint = random.choice(vehicle_bp)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        # spawn
        # print("spawned")

        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, False):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    return vehicles_list


def generate_walkers(client, world, blueprint_library, num_walkers):
    """Generate walkers"""
    percentagePedestriansRunning = 0.0  # how many pedestrians will run
    percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
    walker_speed = []
    batch = []
    pedestrians_list = []
    SpawnActor = carla.command.SpawnActor
    controllers_list = []
    spawn_points = []
    for n in range(num_walkers):
        if n >= num_walkers:
            break
        spawn_point = carla.Transform()
        location = world.get_random_location_from_navigation()
        if location is not None:
            spawn_point.location = location
            spawn_points.append(spawn_point)
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    walker_speed2 = []
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            logging.error(response.error)
        else:
            pedestrians_list.append(response.actor_id)
            walker_speed2.append(walker_speed)
    
    #walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(pedestrians_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), pedestrians_list[i]))
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            logging.error(response.error)
        else:
            controllers_list.append(response.actor_id)
    # 4 Initialize controllers
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(len(controllers_list)):
        # start walker
        world.get_actor(controllers_list[i]).start()
        # set walk to random point
        world.get_actor(controllers_list[i]).go_to_location(world.get_random_location_from_navigation())
        # max speed
        world.get_actor(controllers_list[i]).set_max_speed(float(walker_speed[i]))
    return pedestrians_list





def get_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def applyTransform(transform, location):
    transform = np.array(transform)
    location = np.array([location.x, location.y, location.z ,1])
    location = location.dot(transform.T)
    return carla.Location(location[0],location[1], location[2])

### Get numpy 2D array of vehicles' location and rotation from world reference, also locations from sensor reference
def get_list_transform(vehicles_list, sensor):
    t_list = []
    for vehicle in vehicles_list:
        v = vehicle.get_transform()
        transform = [v.location.x, v.location.y, v.location.z, v.rotation.roll, v.rotation.pitch, v.rotation.yaw]
        t_list.append(transform)
    t_list = np.array(t_list).reshape((len(t_list), 6))

    transform_h = np.concatenate((t_list[:, :3], np.ones((len(t_list), 1))), axis=1)
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    transform_s = np.dot(world_sensor_matrix, transform_h.T).T

    return t_list, transform_s


def degrees_to_radians(degrees):
    return degrees * math.pi / 180