import ray
from ray import tune
from ray.rllib import Policy
from ray.rllib.algorithms.sac import SACConfig, SAC

from Carla_Final.CONFIG import CONFIG
from Carla_Final.Utils.CarlaUtils import show_spawnpoints
from Environment import CarWorld


def main():
    if CONFIG['draw_spawnpoints']:
        show_spawnpoints()
    else:
        if CONFIG["evaluation"]:
            env = CarWorld()
            restored_policy = Policy.from_checkpoint(
                "C:\\Users\\amich\\ray_results\SAC_2023-01-22_12-05-55\SAC_CarWorld_bdda2_00000_0_2023-01-22_12-05-55\checkpoint_003000\policies\default_policy")

            for i in range(20):

                done = False
                obs = env.reset()
                tot_rew = 0
                while not done:
                    action = restored_policy.compute_single_action(obs=obs)[0][0]
                    print(action)
                    action = float(action)
                    obs, reward, done, info = env.step(action)
                    tot_rew += reward
                print("episode reward: ", tot_rew)

        else:  # Training
            try:
                ray.init(num_gpus=1)
                config = SACConfig().environment(env=CarWorld)
                config.framework_str = "torch"
                config.rollouts(num_rollout_workers=1)
                config.resources(num_gpus=0.6, num_gpus_per_worker=0.1)
                config.exploration_config = {  # Improve exploration with random steps
                    "random_timesteps": 5000
                }

                tune.run(SAC,
                         checkpoint_freq=20,
                         checkpoint_at_end=True,
                         keep_checkpoints_num=5,
                         restore="C:\\Users\\amich\\ray_results\SAC_2023-01-22_22-39-29\SAC_CarWorld_403c7_00000_0_2023-01-22_22-39-29\checkpoint_004360",
                         config=config.to_dict())

            finally:
                ray.shutdown()


if __name__ == '__main__':
    main()
