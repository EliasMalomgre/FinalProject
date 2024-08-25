"""
Utility functions for interaction with the Carla Server


"""
import math
import random

import carla

from Carla_Final.CONFIG import CONFIG

def spawn_vehicles(vehicle_spawn_points, world, tm_port, number_of_vehicles, number_of_wheels=4):
    """
    Spawns vehicles in a Carla World
    :param vehicle_spawn_points: List of possible spawn points
    :param number_of_vehicles: Number of vehicles that need to be spawned
    :param env: The environment
    :param number_of_wheels: Number of wheels that the vehicles must have
    :return: /
    """
    random.shuffle(vehicle_spawn_points)  # Shuffle spawnpoints for randomness
    count = number_of_vehicles  # Number of vehicles you want to spawn
    vehicle_list = []
    tempcount = -1
    try_counter = 0
    if count > 0:
        for spawn_point in vehicle_spawn_points:
            res, vehicle = try_spawn_random_vehicle_at(spawn_point, world, tm_port, 'vehicle.*', number_of_wheels=[number_of_wheels])
            if res:
                vehicle_list.append(vehicle)
                count -= 1
            if count <= 0:
                break
    while count > 0:
        if tempcount == count:
            try_counter += 1
        if try_counter == 50:
            break  # Break so we don't get stuck when there aren't enough spawnpoints left
        tempcount = count
        res, vehicle = try_spawn_random_vehicle_at(random.choice(vehicle_spawn_points), env, 'vehicle.*', number_of_wheels=[number_of_wheels])
        if res:
            vehicle_list.append(vehicle)
            count -= 1
    return vehicle_list


def generate_walkers(client, world, blueprint_library, num_walkers):
    """
    Code to spawn pedestrians in a Carla world
    :param client: The client connected to the Carla server
    :param world: The world in which to generate pedestrians
    :param blueprint_library: the blueprint library from which to extract pedestrian blueprints
    :param num_walkers: The number of pedestrians to spawn
    :return:
    """
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
            pass
        else:
            pedestrians_list.append(response.actor_id)
            walker_speed2.append(walker_speed)

    # walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(pedestrians_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), pedestrians_list[i]))
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            pass
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


def set_server_view(vehicle, world):
    """
    Sets the view of the spectator on the Carla server to the location of the vehicle passed as a birdeye following mode
    :param vehicle: The vehicle to follow
    :param world: The world in which the vehicle is driving
    :return: /
    """
    transforms = vehicle.get_transform()
    server_view_x = vehicle.get_location().x - 5 * transforms.get_forward_vector().x
    server_view_y = vehicle.get_location().y - 5 * transforms.get_forward_vector().y
    server_view_z = vehicle.get_location().z + 3
    server_view_pitch = 0
    server_view_yaw = transforms.rotation.yaw
    server_view_roll = transforms.rotation.roll
    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
            carla.Rotation(pitch=server_view_pitch, yaw=server_view_yaw, roll=server_view_roll),
        )
    )


def connect_Client():
    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client(CONFIG['host'], CONFIG['port'])
    client.set_timeout(CONFIG['timeout'])
    world = client.load_world(CONFIG['server_map'], carla.MapLayer.NONE)
    print('Carla server connected!')
    tm = client.get_trafficmanager(CONFIG['tmport'])
    tm_port = tm.get_port()
    print("Traffic manager connected")
    return client, world, tm, tm_port


def try_spawn_random_vehicle_at(transform, world, tm_port, blueprint, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random blueprint.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint: carla.ActorBlueprint = create_vehicle_blueprint(world, blueprint,
                                                               number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
        vehicle.set_autopilot(True, tm_port)
        return True, vehicle
    return False, None


def create_vehicle_blueprint(world, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.
    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
        blueprint_library = blueprint_library + [x for x in blueprints if
                                                 int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
        if not color:
            color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
    return bp


def try_spawn_ego_vehicle_at(transform, world, ego_bp):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = world.try_spawn_actor(ego_bp, transform)
    if vehicle is not None:
        return True, vehicle
    return False, None


def get_speed(vehicle):
    """
  Compute speed of a vehicle in Kmh
  :param vehicle: the vehicle for which speed is calculated
  :return: speed as a float in Kmh
  """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def show_spawnpoints():
    """Tool for drawing every possible in a world with its number so you know where to spawn a car if you want a
    certain location
    It is all drawn in the server view of carla"""
    print('connecting to Carla server...')
    client = carla.Client(CONFIG['host'], CONFIG['port'])
    client.set_timeout(CONFIG['timeout'])
    world = client.load_world(CONFIG['server_map'], carla.MapLayer.NONE)
    print('Carla server connected!')
    settings = world.get_settings()
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print("Traffic manager connected")
    waypoints = list(world.get_map().get_spawn_points())
    for i in range(1, len(waypoints)):
       begin = waypoints[i].location
       angle = math.radians(waypoints[i].rotation.yaw)
       end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
       world.debug.draw_arrow(begin, end, arrow_size=0.5, life_time=10.0)
       world.debug.draw_string(begin,
                         str(i),life_time=1000)