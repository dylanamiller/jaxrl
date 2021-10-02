import datetime
import importlib
import os
import time
import multiprocessing as mp

import jax
import jax.numpy as jnp
import haiku as hk
import rlax

import numpy as np
import pandas as pd
import scipy

import msgpack
import zmq

import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import gym

from models import MuZeroModel
from replay_buffer import ReplayBuffer, Reanalyse
from shared_storage import SharedStorage
from self_play import SelfPlay
from trainer import Trainer



class MuZero:
    def __init__(self, config):
        env_name = config['env']
        self.config = config
        try:
            self.env = gym.make(env_name)
        except ModuleNotFoundError as err:
            print(
                f'{env_name} is not a valid environment'
            )
            raise err

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        if not os.path.isfile(config['log_dir']):
            os.mkdir(config['log_dir'])

        # create unique id for logging
        date = datetime.date.today()
        uid = np.random.randint(10e9)
        exp_id = date + '_' + str(uid)
        exp_path = os.path.join(config['log_dir'], exp_id)

        self.writer = SummaryWriter(log_dir=exp_path)

        self.model = MuZeroModel(config['model_config'])
        self.checkpoint['weights'] = self.model.get_parameters()

    def train(self):
        self.trainer = Trainer(self.checkpoint, self.config)
        self.storage = SharedStorage(self.checkpoint, self.config)
        self.replay_buffer = ReplayBuffer(self.checkpoint, self.replay_buffer, self.config)

        # need to generate different seeds for each instance of self_play
        # can test now with single instance
        self.self_play = SelfPlay(self.checkpoint, self.env, self.config, self.config['seed'])

        # need too create workers for above defined objects
        # need to kick off workers for self_play and trainer that each take the other two
        # storage and replay buffer are accessible by both trainer and self_play

