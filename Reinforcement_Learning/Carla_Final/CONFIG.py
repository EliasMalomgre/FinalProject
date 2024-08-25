import carla
"""Config file containing all adjustable parameters"""

CONFIG = {
    'training_mode': False,
    # False: Turns on pygame window with HUD, Lidar, Object detection. True: Carla Server view only
    'evaluation': True,  # True uses the loaded policy to evaluate your results, False: Training
    'draw_spawnpoints': False, # Helper tool to draw all spanwpoints in a world in server view (does not launch
    # training or evaluation when active!)

    # Connecting to Carla
    "host": "localhost",
    "RETRIES_ON_ERROR": 30,
    "timeout": 100.0,
    "server_map": "Town04",
    'port': 2000,  # connection port
    'tmport': 8000,  # Traffic Manager port

    # Pygame display
    "width": 1280,
    "height": 600,
    "hudwidth": 800, #Width of the camera view of the car
    "hudheight": 600, # height of the camera view
    "lidarwidth": 480, # Width of the lidar view
    "lidarheight": 480, #  Height of the lidar view

    # Environment setup
    "Weather": carla.WeatherParameters.ClearNoon,
    'number_of_vehicles': 300,
    'number_of_walkers': 100,
    'dt': 0.1,  # time interval between two frames
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle

    'max_ego_spawn_times': 200,  # maximum times to try to spawn ego vehicle

    "max_dist": 60,  # Maximum distance to consider to cars in front

    # Folders for lidar and object detection models
    'lidar_model_folder': 'C:\\Users\\amich\Documents\GitHub\ProjectDAI\Computer_Vision\models\lidar-cars-ped-200.pth',
    'detection_model_folder': 'C:\\Users\\amich\Documents\GitHub\ProjectDAI\Computer_Vision\models\\120-car-ped.onnx',

    # Folder for the logo to display on the pygame window
    'logo_dir': 'C:\\Users\\amich\\Documents\GitHub\ProjectDAI\Reinforcement_Learning\Carla_Final\logo-color.png'

}
