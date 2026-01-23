import os
import cv2
import gymnasium as gym
import numpy as np
import torch
import ale_py
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

PATH = "assault"

"""
https://docs.pytorch.org/docs/stable/user_guide/index.html
https://docs.opencv.org/4.12.0/
https://stable-baselines3.readthedocs.io/en/master/
https://ale.farama.org/gymnasium-interface/
Filip Patuła s28615, Michał Bedra s28854
"""

SEED = 48

TOTAL_TIMESTEPS = 20000
BUFFER_SIZE = 300000
STACK_SIZE = 4
OBS_IMG_SHAPE = (105, 80)
STACK_IMG_SHAPE = (STACK_SIZE, 105, 80)
INITIAL_EPS = 1.0
FINAL_EPSILON = 0.1
EXPLORATION_FRACTION = 0.2
FEATURES_SIZE = 512
INITIAL_LEARNING_RATE = 0.00005
LEARNING_STARTS = 1000

class CNNQNetwork(BaseFeaturesExtractor):
    """
    Class with Convolutional Neural Network used for DQN
    :param observation_space: (gym.Space) Observation space from environment
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = FEATURES_SIZE):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float()))


def observations_to_grayscale(observations: np.ndarray):
    """
    Returns grayscale images from observations
    :param observations: (np.ndarray) Observations stack from environment
    """
    grayscaled_imgs = np.empty(shape=STACK_IMG_SHAPE)
    for i in range(STACK_SIZE):
        gray_img = cv2.cvtColor(observations[i], cv2.COLOR_RGB2GRAY)
        gray_img = cv2.resize(
            gray_img, OBS_IMG_SHAPE, interpolation=cv2.INTER_AREA
        )
        gray_img = gray_img.reshape(OBS_IMG_SHAPE)
        grayscaled_imgs[i] = gray_img
    return grayscaled_imgs


def run_game_model():
    """
    Trains DQN model with defined hyperparams or loads it from path, then runs environment with dqn agent
    """
    gym.register_envs(ale_py)

    env = None
    model = None

    if os.path.isfile("assault.zip"):
        model = DQN.load(path=PATH)
    else:
        env = gym.make("ALE/Assault-v5")

        env = gym.wrappers.GrayscaleObservation(env=env)
        env = gym.wrappers.ResizeObservation(env=env, shape=OBS_IMG_SHAPE)
        env = gym.wrappers.FrameStackObservation(env, STACK_SIZE)

        policy_kwargs = dict(
            features_extractor_class=CNNQNetwork,
            features_extractor_kwargs=dict(features_dim=FEATURES_SIZE),
        )

        model = DQN(policy="CnnPolicy",
                    env=env,
                    verbose=1,
                    exploration_initial_eps=INITIAL_EPS,
                    exploration_final_eps=FINAL_EPSILON,
                    exploration_fraction=EXPLORATION_FRACTION,
                    seed=SEED,
                    buffer_size=BUFFER_SIZE,
                    learning_rate=INITIAL_LEARNING_RATE,
                    learning_starts=LEARNING_STARTS,
                    policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=8)
        model.save(PATH)

    env = gym.make("ALE/Assault-v5", render_mode="human")
    env = gym.wrappers.FrameStackObservation(env, STACK_SIZE)

    obs, info = env.reset()

    while True:
        gray_imgs = observations_to_grayscale(obs)
        action, states = model.predict(gray_imgs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    run_game_model()