import math
import random
from collections import deque

import matplotlib.pyplot as plt

import gym
import ray
import torch.cuda
from gym import spaces
import numpy as np
from ray.rllib.algorithms.sac import SACConfig, SAC
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQNConfig, ApexDQN

DISCRETE_ACTIONS = True


class Car:
    def __init__(self, min_speed, max_speed, current_speed, max_acc, max_brake, position) -> None:
        super().__init__()
        self.speed = current_speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.max_brake = max_brake
        self.position = position
        self.acceleration = 0

    def step(self, action, steps, dt):
        return self.speed

    def get_speed(self):
        return self.speed

    def set_speed(self, speed):
        self.speed = speed

    def get_max_acc(self):
        return self.max_acc

    def get_max_brake(self):
        return self.max_brake

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = pos

    def get_acc(self):
        return self.acceleration

    # Returns the current state of the car
    def get_status(self):
        return self.speed, self.acceleration, self.position


class EgoCar(Car):
    def __init__(self, min_speed, max_speed, current_speed, max_acc, max_brake, desired_speed, position) -> None:
        super().__init__(min_speed, max_speed, current_speed, max_acc, max_brake, position)
        self.desired_speed = desired_speed

    # https: // www.scribd.com / doc / 253507303 / Simple - Car - Dynamics
    def step(self, action, steps, dt):
        # Determine acceleration
        if action >= 0:
            self.acceleration = self.max_acc * action
        else:
            self.acceleration = self.max_brake * action

        # Update speed
        self.speed += self.acceleration * dt

        # Update Position
        self.position += self.speed * dt
        return self.speed

    def get_desired_speed(self):
        return self.desired_speed

    def set_desired_speed(self, speed):
        self.desired_speed = speed


class LeadCar(Car):

    def __init__(self, min_speed, max_speed, current_speed, max_acc, max_brake, position, beh_type, baseline_speed,
                 period, deviation) -> None:
        super().__init__(min_speed, max_speed, current_speed, max_acc, max_brake, position)
        self.leaving_road = False
        self.beh_type = beh_type
        self.baseline_speed = baseline_speed
        self.period = period
        self.deviation = deviation

    def step(self, action, steps, dt):
        # Different behaviours
        if self.beh_type == 0:  # Sinusoid
            self.speed = math.sin(steps / self.period) * self.deviation + self.baseline_speed
        elif self.beh_type == 1:  # Triangle Wave
            self.speed = 4 / (self.period * 10 * 6) * self.deviation * abs(
                ((
                         steps - 6 * self.period * 10 / 4) % self.period * 10) - 6 * self.period * 10 / 2) - self.deviation + self.baseline_speed
        elif self.beh_type == 2:  # Maintain set speed
            None
        elif self.beh_type == 3:  # Random acceleration and deceleration
            if self.speed < 0:
                self.speed += self.max_acc * (random.random()) * dt
            else:
                self.speed += self.max_acc * (random.uniform(-1., 1.)) * dt
        elif self.beh_type == 4:  # Backwards
            self.speed = -1 * abs(self.speed)

        if random.random() < 0.0001:  # Simulate leaving the road
            self.position += 50

        self.position += self.speed * dt
        return self.speed

    def is_leaving_road(self):
        return self.leaving_road

    def set_leaving_road(self, val: bool):
        self.leaving_road = val

    def set_bl_speed(self, val):
        self.baseline_speed = val

    def set_beh(self, beh):
        self.beh_type = beh

    def get_beh(self):
        return self.beh_type

    def set_period(self, period):
        self.period = period

    def set_deviation(self, dev):
        self.deviation = dev


class CarWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def render(self, mode="human"):
        identifier = str(random.randint(0, 100000))
        plt.figure(1)  # Speed
        plt.title("Speed Evolution " + identifier)
        plt.plot(self.ego_car_history["ego_speed"], color="m")
        plt.plot(self.lead_car_history["lead_speed"], color="k")
        plt.plot(self.ego_car_history["desired_speed"], color="g")
        plt.show(block=False)

        plt.figure(2)  # Distance
        plt.title("Distance Evolution " + identifier)
        plt.plot(self.distance_history["distance"], color="b")
        plt.plot(self.distance_history["distance_min"], color="r")
        plt.plot(self.distance_history["distance_max"], color="g")
        plt.show(block=False)

        plt.figure(3)  # Reward
        plt.title("Reward Evolution " + identifier)
        plt.plot(self.reward_history)
        plt.show(block=False)

        plt.figure(3)  # Position
        plt.title("Position Evolution" + identifier)
        plt.plot(self.ego_car_history["ego_position"], color="m")
        plt.plot(self.lead_car_history["lead_position"], color="k")
        plt.show(block=False)

    def __init__(self, env_config=None):

        if DISCRETE_ACTIONS:
            self.number_of_actions = 101
            self.action_space = spaces.Discrete(self.number_of_actions)
        else:
            self.action_space = spaces.Box(np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32),
                                           dtype=np.float32)

        self.env_config = env_config

        # Time Step
        self.dt = 0.05

        # Ego Car
        min_speed = -np.inf
        max_speed = np.inf
        max_speed_deviation = 3
        desired_speed = random.uniform(9, 30)
        initial_pos = 0
        current_speed = 0
        self.ego_car = EgoCar(min_speed=min_speed, max_speed=max_speed, current_speed=current_speed,
                              max_acc=max_speed_deviation, max_brake=max_speed_deviation, desired_speed=desired_speed,
                              position=initial_pos)

        # Lead Car
        lead_car_speed = 0
        lead_car_min_speed = -np.inf
        lead_car_max_speed = np.inf
        beh_type = random.randint(0, 3)
        initial_pos_lead = random.randint(50, 200)
        deviation = random.uniform(2, 5)
        period = random.uniform(150, 250)
        baseline = random.uniform(4, 7)
        self.lead_car = LeadCar(min_speed=lead_car_min_speed, max_speed=lead_car_max_speed,
                                current_speed=lead_car_speed, max_acc=max_speed_deviation,
                                max_brake=max_speed_deviation, beh_type=beh_type, position=initial_pos_lead,
                                deviation=deviation, baseline_speed=baseline, period=period)

        # Distance
        min_distance = -np.inf
        max_distance = np.inf
        self.distance = self.lead_car.get_position() - self.ego_car.get_position()
        self.desired_distance_low = 4
        self.desired_distance_high = 8

        # History
        self.ego_car_history = {"ego_position": [], "ego_speed": [], "desired_speed": []}  # Past positions and speed
        self.lead_car_history = {"lead_position": [], "lead_speed": []}  # Past positions and speed
        self.distance_history = {"distance": [], "distance_min": [], "distance_max": []}  # Past distances
        self.reward_history = list()  # Past rewards

        self.previous_action = 0
        self.previous_speed = 0

        low = np.tile(np.array(
            [
                min_speed,
                min_distance,
                min_speed,
                -1
            ],
            dtype=np.float32
        ), 3).flatten()

        high = np.tile(np.array(
            [
                max_speed,
                max_distance,
                max_speed,
                1
            ],
            dtype=np.float32
        ), 3).flatten()

        # Observation space: Own speed, distance, desired_speed, previous action
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state = None

        self.frame_stack = deque(maxlen=3)

        # Time Management
        self.count = 0
        self.steps = 0

        # Reset
        self.reset()

    def reset(self, seed=None,):
        if seed is not None:
            random.seed(seed)

        # Ego Car
        own_speed = random.randint(0,1)
        self.ego_car.set_speed(own_speed)
        self.ego_car.set_position(0)
        self.ego_car.set_desired_speed(random.uniform(9, 30))
        lead_car_speed = random.uniform(0., 20.)

        # Lead Car
        self.lead_car.set_deviation(random.uniform(2, 5))
        self.lead_car.set_period(random.uniform(150, 250))
        self.lead_car.set_speed(lead_car_speed)
        self.lead_car.set_bl_speed(random.uniform(4, 7))
        self.lead_car.set_position(random.randint(50, 200))
        self.lead_car.set_beh(random.randint(0, 4))

        # Distance
        distance = self.lead_car.get_position() - self.ego_car.get_position()
        desired_speed = self.ego_car.get_desired_speed()

        # Reset Counter
        self.count = 0

        # History tracking reset
        self.ego_car_history = {"ego_position": [], "ego_speed": [], "desired_speed": []}  # Past positions and speed
        self.lead_car_history = {"lead_position": [], "lead_speed": []}  # Past positions and speed
        self.distance_history = {"distance": [], "distance_min": [], "distance_max": []}  # Past distances
        self.reward_history = list()  # Past rewards

        # Conversions if np array
        distance = distance[0] if isinstance(distance, np.ndarray) else distance
        desired_speed = desired_speed[0] if isinstance(desired_speed, np.ndarray) else desired_speed
        own_speed = own_speed[0] if isinstance(own_speed, np.ndarray) else own_speed

        # Initialize the rest
        self.distance = distance
        self.previous_action = 0

        state = (own_speed, distance, desired_speed, self.previous_action)

        for i in range(3):
            self.frame_stack.append(state)

        self.previous_speed = 0



        return np.array(list(self.frame_stack)).flatten()

    def step(self, action):

        if DISCRETE_ACTIONS:
            action = (action - (self.number_of_actions - 1) / 2) / ((self.number_of_actions - 1) / 2)

        # Update Ego Car
        new_speed = self.ego_car.step(action, self.count, self.dt)
        desired_speed = self.ego_car.get_desired_speed()

        # Update Lead Car
        new_lead_car_speed = self.lead_car.step(action, self.count, self.dt)

        # Calculate Distance
        new_distance = self.lead_car.get_position() - self.ego_car.get_position()

        new_distance = new_distance[0] if isinstance(new_distance, np.ndarray) else new_distance
        new_speed = new_speed[0] if isinstance(new_speed, np.ndarray) else new_speed

        # See whether it is done
        done = self.is_done(new_distance)

        if new_distance <= 0:
            reward = -20
        else:
            reward = self.calculate_reward(new_distance, new_speed, action, desired_speed)
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            else:
                reward = reward

        info = {}

        self.steps += 1
        self.count += 1

        self.previous_action = action
        self.previous_speed = new_speed

        desired_speed = desired_speed[0] if isinstance(desired_speed, np.ndarray) else desired_speed
        self.distance = new_distance
        self.update_history(reward)

        #action = action[0] if isinstance(action, np.ndarray) else action

        self.frame_stack.append(np.array([new_speed, new_distance, desired_speed, action]).flatten())
        state = np.array(list(self.frame_stack)).flatten()

        return state, reward, done, info

    def is_done(self, distance):
        return distance <= 0 or sum(self.reward_history) > 25000 or self.count > 5000

    def calculate_reward(self, new_distance, new_speed, action, desired_speed):
        # https://www.way.com/blog/how-to-save-gas-tips-for-fuel-efficient-driving/
        # https://www.nrcan.gc.ca/energy-efficiency/transportation-alternative-fuels/personal-vehicles/fuel-efficient-driving-techniques/21038
        # https://kitchingroup.cheme.cmu.edu/blog/2013/01/31/Smooth-transitions-between-discontinuous-functions/
        # https://kitchingroup.cheme.cmu.edu/blog/2013/02/27/Smooth-transitions-between-two-constants/

        # Speed Rewards
        # Transition band between goal speeds
        # Smoothing this out will also help decrease fuel consumption
        dist_var = self.desired_distance_high * 10 if new_distance > self.desired_distance_high * 10 else new_distance
        transition = 1 / (1 + math.exp(-(1 - (dist_var - 4 * self.desired_distance_high)) / 6))
        speed_var = desired_speed if self.lead_car.get_speed() > desired_speed else self.lead_car.get_speed()
        goal_speed = speed_var + (desired_speed - speed_var) * (1 - transition)

        verr = abs(new_speed - goal_speed)

        speed_reward = 15 * math.exp(-1 / 40 * math.pow(verr, 2)) - 10  # Try /20 /50 /60

        # Speed Bonus
        speed_bonus = 1 if verr < 0.5 else 0

        # Distance rewards
        if new_distance < self.desired_distance_low:
            distance_reward = -7  # Compensate for good speed reward (= max 5)
        else:
            distance_reward = 0

        # Speed Change Reward
        if DISCRETE_ACTIONS:
            action = (action - (self.number_of_actions - 1) / 2) / ((self.number_of_actions - 1) / 2)

        speed_change_reward = -0.5 * abs(action - self.previous_action)+1

        reward = speed_reward + distance_reward + speed_bonus + speed_change_reward

        return reward

    def update_history(self, reward):
        self.ego_car_history["ego_position"].append(self.ego_car.get_position())
        self.ego_car_history["ego_speed"].append(self.ego_car.get_speed())
        self.ego_car_history["desired_speed"].append(self.ego_car.get_desired_speed())
        self.lead_car_history["lead_position"].append(self.lead_car.get_position())
        self.lead_car_history["lead_speed"].append(self.lead_car.get_speed())
        self.distance_history["distance"].append(self.distance)
        self.distance_history["distance_min"].append(self.desired_distance_low)
        self.distance_history["distance_max"].append(self.desired_distance_high)
        self.reward_history.append(reward)

print(torch.cuda.is_available())
if __name__ == "__main__":
    ray.init(num_gpus=1)
    """ApexDQN"""
    config = ApexDQNConfig().environment(env=CarWorld)
    config.framework_str = "torch"
    config.rollouts(num_rollout_workers=2, rollout_fragment_length=12)
    config.resources(num_gpus=0.6, num_gpus_per_worker=0.1)
    config.training(gamma=0.99, num_steps_sampled_before_learning_starts=20000,
                    optimizer={"num_replay_buffer_shards": 1},
                    replay_buffer_config={
                        "no_local_replay_buffer": True,
                        "type": "MultiAgentPrioritizedReplayBuffer",
                        "capacity": 1000000,
                        "prioritized_replay_alpha": 0.6,
                        # Beta parameter for sampling from prioritized replay buffer.
                        "prioritized_replay_beta": 0.4,
                        # Epsilon to add to the TD errors when updating priorities.
                        "prioritized_replay_eps": 1e-6,
                    },
                    train_batch_size=512)
    config.exploration_config = {
        'epsilon_timesteps': 200000,
        'final_epsilon': 0.02,
    }
    # config.evaluation_config["explore"] = False
    # config.evaluation_interval = 1
    # config.evaluation_duration = 1
    # config.evaluation_duration_unit = "episodes"

    # tune.run(
    #    ApexDQN,
    #    # restore="C:/Users/amich/ray_results/NewRewardFct/SAC_CarWorld_192fc_00000_0_2022-12-13_20-41-12/checkpoint_000400",
    #    # resume=True,
    #    name="ApexDQN",
    #    checkpoint_freq=200,
    #    # callbacks=([WandbLoggerCallback(project="test-project", api_key="65b5edd9170d5604128b0423c18c388666b0672c")]),
    #    config=config.to_dict(),
    #
    # )

    algo = ApexDQN(env=CarWorld, config=config.to_dict())

    ### Uncomment if you want to load a checkpoint
    # algo.load_checkpoint(
    #    "C:\\Users\\amich\Documents\GitHub\ProjectDAI\Reinforcement_Learning\BasicSimulator\checkpoints")

    env = CarWorld(config.to_dict())
    for i in range(1000):
        results = algo.train()
        print(pretty_print(results))

        # Checkpointing
        if i % 10 == 0:
            algo.save_checkpoint("./checkpoints")

        # Evaluation
        done = False
        obs = env.reset()
        while not done:
            action = algo.compute_single_action(observation=obs, explore=False)
            obs, reward, done, info = env.step(action)
        env.render()

    # Test in 10 random cases after training
    for i in range(10):
        config = {"Testnr: ", i}
        env = CarWorld(config)

        done = False
        obs = env.reset()
        while not done:
            action = algo.compute_single_action(observation=obs, explore=False)
            obs, reward, done, info = env.step(action)
        env.render()
