import carla


def get_lidar_sensor(world):
    """
    Returns a lidar sensor blueprint and its transformation in a certain world
    :param world: World to spawn it in
    :return: lidar_bp: blueprint of lidar
    lidar_trans: transformation of lidar
    """
    lidar_height = 1.8
    lidar_trans = carla.Transform(carla.Location(x=0.0, z=lidar_height))
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '60')
    lidar_bp.set_attribute('rotation_frequency', '20')

    lidar_bp.set_attribute('points_per_second', str(256 / 0.00004608))
    lidar_bp.set_attribute('upper_fov', str(3))
    lidar_bp.set_attribute('lower_fov', str(-22))
    return lidar_bp, lidar_trans
