from __future__ import division

import copy
import math
import random
import time
import weakref
from collections import deque

import carla
import gym
import numpy as np
import pygame
from gym import spaces
from gym.utils import seeding

from CONFIG import CONFIG
from Carla_Final.Computer_Vision.CameraDetection import CameraDetection
from Carla_Final.Computer_Vision.LidarDetection import Lidar
from Carla_Final.Utils.CarlaUtils import try_spawn_ego_vehicle_at, try_spawn_random_vehicle_at, set_server_view, \
    spawn_vehicles, generate_walkers, connect_Client, create_vehicle_blueprint, get_speed
from Carla_Final.Utils.KeyBoardControl import KeyboardControl
from Carla_Final.Utils.utils import draw_boundingbox, draw_waypoints
from Carla_Final.agents.navigation.CustomAgent import CustomAgent
from Carla_Final.manager.LidarManager import get_lidar_sensor
from HUD.HUD import HUD
from manager.CameraManager import CameraManager


class CarWorld(gym.Env):
    """The environment, central processing unit of everything
    -Connects to the Carla Server
    -Sets up the carla world
    -Contains step, reset and reward function for RL training"""

    def __init__(self):

        # Traffic manager
        self.tm_port = None
        self.tm = None

        self.time_step = None
        self.route_polygon = None
        self.previous_speed = 0
        self.vehicle_list = []

        self.collision_sensor = None

        # Lidar
        self.lidar_sensor = None
        self.lidar_bp = None
        self.lidar_trans = None
        self.lidar_height = None
        self.lidar_data = None
        self.lidar_fct = Lidar()
        self.detection_fct = CameraDetection()
        self.walker_spawn_points = None
        self.world: carla.World = None
        self.client: carla.Client = None
        self.vehicle_spawn_points = None
        self.boxes = None

        # Parameters
        self.previous_action = None
        self.display_width = CONFIG['width']  # rendering screen size
        self.display_height = CONFIG['height']
        self.number_of_vehicles = CONFIG['number_of_vehicles']
        self.number_of_walkers = CONFIG['number_of_walkers']
        self.dt = CONFIG['dt']
        self.max_ego_spawn_times = CONFIG['max_ego_spawn_times']
        self.map = CONFIG['server_map']
        self.debug = False  # For drawing waypoints and bounding boxes of lidar

        self.lead_car_speed = 0

        self.goal_speed = 0

        self.agent: CustomAgent = None

        self.server_clock = pygame.time.Clock()
        self.ego = None
        self.camera_manager: CameraManager = None
        self.keyboard_control = KeyboardControl()

        self.training = CONFIG['training_mode']

        """
        Action and observation space definitions
        """
        self.action_space = spaces.Box(np.array([-1.0]), np.array([1.0]),
                                       dtype=np.float64)
        # Speed
        min_speed = -np.inf
        max_speed = np.inf

        # Distance
        self.lidar_distance = 0
        self.true_distance = 0
        min_distance = -np.inf
        max_distance = np.inf
        self.desired_distance_low = 4
        self.desired_distance_high = 8

        # Desired speeds
        self.desired_speeds = [60, 90]  # Possible desired speeds
        self.desired_speed = random.choice(self.desired_speeds) / 3.6  # Pick randomly

        # Overtaking
        self.overtaking = 0
        self.overtaking_counter = 0
        self.train_overtaking = 0
        self.previous_des_speed = None  # Place holder for previous desired speed when overtaking

        # Traffic light detected
        self.tl = 0

        # Observation space
        low = np.tile(np.array(
            [
                min_speed,
                min_distance,
                min_speed,
                -1.0,
                min_speed,
                0,
                0
            ],
            dtype=np.float32
        ), 3).flatten()

        high = np.tile(np.array(
            [
                max_speed,
                max_distance,
                max_speed,
                1.0,
                max_speed,
                1,
                1
            ],
            dtype=np.float32
        ), 3).flatten()

        # Observation space: Own speed, distance, desired_speed, previous action, speed_limit, overtaking, traffic_light_detected
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.frame_stack = deque(maxlen=3)  # Frame stacked observation space

        self.episode_reward = 0

        self.manual_reset = False

        # Connect Client to Carla server
        self.client, self.world, self.tm, self.tm_port = connect_Client()

        # Set weather
        self.world.set_weather(CONFIG['Weather'])

        # Get spawn points
        self.get_spawn_points()

        if self.world == None:
            print("None World")

        # Create the ego vehicle blueprint
        self.ego_bp = create_vehicle_blueprint(self.world, CONFIG['ego_vehicle_filter'], color='49,8,8')
        self.ego_bp.set_attribute('role_name', 'hero')  # For traffic manager

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # We don't spawn the lidar when training
        if not self.training:
            self.lidar_bp, self.lidar_trans = get_lidar_sensor(self.world)

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        self.settings.no_rendering_mode = True if not self.training else False
        self.settings.max_substep_delta_time = 0.01
        self.settings.max_substeps = 10
        self.world.apply_settings(self.settings)

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Initialize the renderer
        if not self.training:
            self._init_renderer()

        # Tick
        weak_self = weakref.ref(self)
        self.world.on_tick(lambda timestamp: CarWorld.on_world_tick(weak_self, timestamp))
        self.server_fps = 0.0
        self.simulation_time = 0

        # Reset
        self.reset()

    def reset(self):
        """Reset function for the RL agent
        Resets the state of the Carla World and all parameters of the Environment"""
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
        time.sleep(0.1)

        self.manual_reset = False
        self.agent = None
        self.train_overtaking = 0

        self.episode_reward = 0

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*',
                                'controller.ai.walker', 'walker.*', 'vehicle', 'sensor'])

        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None

        time.sleep(0.5)  # Sleep so Carla has time to catch up
        self.vehicle_list = []  # Empty vehicle list

        # Disable sync mode while resetting for stability
        self._set_synchronous_mode(False)
        self.tm.set_synchronous_mode(False)
        time.sleep(0.5)

        """Uncomment lines bellow if you want to load a random world every new episode
        Has varying results and crashes occasionally and without reason. Enough RAM and video RAM is necessary """
        # if self.reset_step != 0:
        #    self.client.load_world(random.choice(self.client.get_available_maps()), reset_settings=False)
        #    time.sleep(5.5) # Sleep so Carla can load everything

        # Setup traffic manager again
        self.tm = self.client.get_trafficmanager(CONFIG['tmport'])
        self.tm_port = self.tm.get_port()
        time.sleep(1)
        self.tm.set_synchronous_mode(False)

        # if self.reset_step != 0:
        #    self.setup_world()
        #    time.sleep(2)

        # Pick a random desired speed
        self.desired_speed = random.choice(self.desired_speeds) / 3.6
        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()
            transform = random.choice(self.vehicle_spawn_points)
            res, vehicle = try_spawn_ego_vehicle_at(transform, self.world, self.ego_bp)  # Spawn the go vehicle
            if res:
                self.ego = vehicle
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)
        time.sleep(1.5)

        if not self.training:
            self.camera_manager = CameraManager(self.ego)
            self.camera_manager.set_sensor()
        self.agent = CustomAgent(self.ego)

        # Add lidar sensor
        if not self.training:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

            def get_lidar_data(data):
                self.lidar_data = data

        if self.training:
            set_server_view(self.ego, self.world)

        # Add the collision sensor
        self.add_collision_sensor()

        # Empty collision history
        self.collision_hist = []

        # Let the cars slow down randomly for training following
        if random.random() < 0.25:
            self.tm.global_percentage_speed_difference(25)
        # Enable sync mode
        self.settings.synchronous_mode = True
        self.tm.set_synchronous_mode(True)

        self.world.apply_settings(self.settings)

        self.previous_action = 0

        # Spawn vehicles
        self.vehicle_list = spawn_vehicles(self.vehicle_spawn_points, self.world, self.tm_port,
                                           self.number_of_vehicles)
        generate_walkers(self.client, self.world, self.world.get_blueprint_library(), self.number_of_walkers)

        self.overtaking = 0
        self.overtaking_counter = 0

        new_speed, new_distance, speed_limit, _, traffic_light_detected = self._get_obs()
        # Observation space: Own speed, distance, desired_speed, previous action, speed_limit, overtaking,
        # traffic_light_detected
        state = (new_speed, new_distance, self.desired_speed, self.previous_action, speed_limit, self.overtaking,
                 traffic_light_detected)
        for i in range(3):
            self.frame_stack.append(state)
        self.previous_speed = new_speed

        # Update timestamps
        self.time_step = 0
        self.reset_step += 1
        return np.array(list(self.frame_stack)).flatten()

    def step(self, action):
        """Step function:
        Agent takes a step here"""
        action = action[0] if isinstance(action, np.ndarray) else action

        total_reward = 0
        # Apply action to ego vehicle
        self.ego.apply_control(self.agent.run_step(action))
        # Tick the world (automatically executes 'on world tick' as well)
        self.world.tick()
        # Update timestamps
        self.time_step += 1
        self.total_step += 1
        new_speed, new_distance, speed_limit, lead_car_speed, traffic_light_detected = self._get_obs()
        new_distance = new_distance[0] if isinstance(new_distance, np.ndarray) else new_distance
        new_speed = new_speed[0] if isinstance(new_speed, np.ndarray) else new_speed
        speed_limit = speed_limit[0] if isinstance(speed_limit, np.ndarray) else speed_limit
        lead_car_speed = lead_car_speed[0] if isinstance(lead_car_speed, np.ndarray) else lead_car_speed
        total_reward += self._get_reward(new_speed, new_distance, speed_limit, lead_car_speed,
                                         action, traffic_light_detected)

        # If collision occurs
        if len(self.collision_hist) >= 1:
            self.collision_sensor.stop()
            total_reward = -100

        # Activate overtaking randomly for training purposes
        if self.train_overtaking == 1 and random.random() < 0.1 and self.overtaking == 0 and self.training:
            self.overtaking = 1

        if self.overtaking == 1:
            temp_des = self.desired_speed
            self.desired_speed = speed_limit

            if self.overtaking_counter < 25:
                if self.overtaking_counter == 0:
                    self.previous_des_speed = temp_des
                self.overtaking_counter += 1
            else:
                self.overtaking_counter = 0
                self.overtaking = 0
                self.desired_speed = self.previous_des_speed

        # Observation space: Own speed, distance, desired_speed, previous action, speed_limit, overtaking,
        # traffic_light_detected
        self.frame_stack.append(np.array(
            [new_speed, new_distance, self.desired_speed, action, speed_limit, self.overtaking,
             traffic_light_detected]).flatten())
        state = np.array(list(self.frame_stack)).flatten()

        self.previous_action = action
        self.previous_speed = new_speed
        self.episode_reward += total_reward
        info = {}
        return state, total_reward, self._terminal(), copy.deepcopy(info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _init_renderer(self):
        """Initialize the pygame view renderer.
        """
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        logo = pygame.image.load(CONFIG["logo_dir"])
        logo = pygame.transform.scale(logo, (480, 120))
        self.display.blit(logo, (800, 480))

        self.hud = HUD(CONFIG["hudwidth"], CONFIG["hudheight"], CONFIG["server_map"])

    def _set_synchronous_mode(self, synchronous=True):
        """Set synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _get_obs(self):
        """Get an observation from our Carla world"""
        if not self.training:  # If pygame is active
            # Process, keyboard events
            result = self.keyboard_control.parse_events(self.world, self.agent.get_local_planner().get_waypoints(),
                                                        self.client)
            if result is not None:
                self.process_keyboard(result)

        vehicle, self.true_distance, x_of_front = self.get_true_distance()
        traffic_light_detected, _, _ = self.agent.detect_traffic_light(max_distance=5.5)
        self.tl = 1 if traffic_light_detected else 0

        self.red_light = None
        self.speed_limit_cam = None

        if not self.training:  # If we use lidar, object detection and pygame window
            pygame.event.get()
            tic = time.perf_counter()
            new_distance, img, self.boxes = self.lidar_fct.get_distance(self.lidar_data, False, self.ego,
                                                                        self.route_polygon)
            toc = time.perf_counter()
            print(f"Lidar took: {toc - tic:0.4f} seconds")
            surface = pygame.image.frombuffer(img.tobytes(), img.size, 'RGB')
            self.lidar_distance = new_distance - x_of_front - 1.6  # Correction factor
            self.display.blit(surface, (800, 0))
            tic = time.perf_counter()
            img = self.camera_manager.get_image()
            if img is not None:
                image, self.red_light, self.speed_limit_cam = self.detection_fct.detect(img)
                self.camera_manager.set_image(image)
            toc = time.perf_counter()
            print(f"Object detection: {toc - tic:0.4f} seconds")
            self.camera_manager.render(self.display)
            self.hud.render(self.display)

            # Display on pygame
            pygame.display.flip()

        new_speed = get_speed(self.ego) / 3.6  # Convert to m/s
        if self.training:  # Not using Lidar
            if self.true_distance < 0:
                new_distance = CONFIG['max_dist']  # Output max distance
            else:
                new_distance = self.true_distance
        else:  # using Lidar
            if self.lidar_distance < 0:
                new_distance = CONFIG['max_dist']  # Output max distance
            else:
                new_distance = self.lidar_distance
        if self.speed_limit_cam is not None and self.ego.get_speed_limit() == self.speed_limit_cam:
            desired_speed = self.speed_limit_cam / 3.6
        else:
            desired_speed = self.ego.get_speed_limit() / 3.6

        if vehicle is None:
            lead_car_speed = new_speed
        else:
            lead_car_speed = get_speed(vehicle) / 3.6
        self.lead_car_speed = lead_car_speed * 3.6

        return new_speed, new_distance, desired_speed, lead_car_speed, traffic_light_detected

    def _get_reward(self, new_speed, new_distance, speed_limit, lead_car_speed, action, tl_detected):
        """Calculate the step reward."""

        # https://www.way.com/blog/how-to-save-gas-tips-for-fuel-efficient-driving/
        # https://www.nrcan.gc.ca/energy-efficiency/transportation-alternative-fuels/personal-vehicles/fuel-efficient-driving-techniques/21038
        # https://kitchingroup.cheme.cmu.edu/blog/2013/01/31/Smooth-transitions-between-discontinuous-functions/
        # https://kitchingroup.cheme.cmu.edu/blog/2013/02/27/Smooth-transitions-between-two-constants/
        # https://math.stackexchange.com/questions/3877887/smooth-transition-function-with-fixed-start-end-points

        def exp(var):
            """Exponent function for reward calculations. Protected against overflow"""
            if var > 500:
                return math.exp(500)
            else:
                return math.exp(var)

        # Speed Rewards
        # Transition band between goal speeds using smooth step function
        # Smoothing this out will also help decrease fuel consumption
        speed_var = self.desired_speed if lead_car_speed > self.desired_speed else lead_car_speed
        dist_var = (new_distance + 3 - 4 * new_speed / 10) / (60.001 + 3 - 4 * new_speed / 10)
        transition = exp((-0.7) / dist_var) / (exp((-0.7) / dist_var) + (exp((-0.7) / (1 - dist_var))))
        goal_speed = speed_var + (25 - speed_var) * transition

        # If the goal speed is faster than we want, pick the desired speed
        if goal_speed > self.desired_speed:
            goal_speed = self.desired_speed

        # If the goal speed is higher than the speed limit, pick the speed limit
        if goal_speed > speed_limit:
            goal_speed = speed_limit

        # If we are getting too close to the car in front, pick the speed of the car in front
        if new_distance < (self.desired_distance_high + self.desired_distance_low) / 2 + 3 / 10 * new_speed:
            goal_speed = speed_var

        # If overtaking, disregard all other information and drive up to the speed limit
        if self.overtaking == 1:
            goal_speed = speed_limit

        # If a red traffic light is detected, just stop
        if tl_detected == 1:
            goal_speed = 0

        self.goal_speed = goal_speed

        # Difference between current and goal speed
        verr = abs(new_speed - goal_speed)

        # If the car in front is stopped, and we are getting too close, also stop
        if speed_var < 0.01 and new_distance < (self.desired_distance_low + self.desired_distance_high) / 2:
            if new_speed < 0.01:
                speed_reward = 5
            else:
                speed_reward = -10
        else:
            # Reward for being close to the goal speed
            speed_reward = 15 * math.exp(-1 * math.pow(verr, 2)) - 10

        # Speed Bonus
        speed_bonus = 5 if verr < 0.3 else 0

        # Acceleration/deceleration bonus
        if (self.previous_speed < goal_speed and action > 0) or (self.previous_speed > goal_speed and action < 0) or (
                self.previous_speed == goal_speed and action == 0):
            acc_bonus = 4
        else:
            acc_bonus = -4

        # Bonus/Penalty for getting closer to the goal speed
        if abs(self.previous_speed - goal_speed) > abs(self.previous_speed - goal_speed):
            acc_bonus += 3
        else:
            acc_bonus -= 3

        # Distance rewards
        if (new_distance < self.desired_distance_low) and (action >= -0.9):
            distance_reward = -15  # If we are too close
        else:
            distance_reward = 0

        # Reward penalty so action is not too different from previous
        speed_change_reward = -1 * abs(action - self.previous_action) + 0.5

        # Penalty for high acceleration
        hard_acceleration_penalty = -2 * math.pow(abs(action), 2) if action >= 0 else 0

        reward = speed_reward + distance_reward + speed_bonus + speed_change_reward + acc_bonus + hard_acceleration_penalty

        # In case it is put in a np array
        if isinstance(reward, np.ndarray):
            reward = reward[0]
        else:
            reward = reward

        return reward

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        return len(
            self.collision_hist) > 0 or self.time_step > 3500 or self.episode_reward > 28000 or self.manual_reset or (
                       self.train_overtaking == 1 and self.time_step > 2000)

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for vehicle in self.vehicle_list:
            if vehicle.is_alive:
                vehicle.destroy()
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()

    @staticmethod
    def on_world_tick(weak_self, timestamp):
        """
        Executes on every world tick
        :param weak_self: Weak reference to world to avoid recursive referencing
        :param timestamp: Current timestamp
        :return:
        """
        self = weak_self()
        if not self:
            return
        if not self.training:
            if self.debug:
                if self.agent is not None:
                   draw_waypoints(self.world, self.agent.get_local_planner().get_waypoints())
                if self.boxes is not None:
                    draw_boundingbox(self.world, self.boxes, self.ego)
        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.simulation_time = timestamp.elapsed_seconds
        overtaking = self.overtaking
        if self.ego is None:
            speed_lim = 0
        else:
            speed_lim = self.ego.get_speed_limit()

        if not self.training:
            self.hud.tick(self.world, self.server_clock, self.server_fps, self.simulation_time, self.ego,
                          self.lidar_distance, self.true_distance, speed_lim, self.episode_reward, self.lead_car_speed,
                          self.desired_speed * 3.6, overtaking, self.tl, self.goal_speed * 3.6)

        if self.training and self.ego is not None:
            self.set_server_view()

    def get_true_distance(self):
        """Return the true distance to the next car in front"""
        x_of_front = 0
        if self.agent is None:
            return 0, -1, x_of_front
        else:
            intersection, car, distance, x_of_front, self.route_polygon = self.agent.distance_tool(None,
                                                                                                   max_distance=CONFIG[
                                                                                                       'max_dist'])
            if intersection:
                return car, distance, x_of_front

        return None, -1, x_of_front

    def get_spawn_points(self):
        """Fetch the spawnpoints of the world we are currently in"""
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

    def process_keyboard(self, result):
        """Process the keyboard inputs"""
        if result == 0:
            self.debug = not self.debug
        if 1 <= result <= 7:
            self._clear_all_actors()
            self.client.load_world('Town0' + str(result), map_layers=carla.MapLayer.NONE)
        if result == 10:
            self.manual_reset = True
        if result == 11:
            self.desired_speed = 90 / 3.6 if self.desired_speed == 60 / 3.6 else 60 / 3.6
        if result == 12:
            self.overtaking = 0 if self.overtaking == 1 else 1
        if result == 13:  # Let the cars drive fast
            self.tm.global_percentage_speed_difference(-100)
        if result == 14:
            self.tm.global_percentage_speed_difference(30)

    def setup_world(self):
        """
        Setup a newly loaded world
        :return:
        """
        self.get_spawn_points()
        self.world.set_weather(CONFIG['Weather'])

    def add_collision_sensor(self):
        """
        Add the collision sensor the ego vehicle
        :return:
        """
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            actor_we_collide_against = event.other_actor.type_id
            if not self.training:
                self.hud.notification("Collided with: " + str(actor_we_collide_against), 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
