import datetime
import importlib
import os
import sys
import time
from multiprocessing import shared_memory

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
    def __init__(self, game_name, config):
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name'
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

        # set parameters and optimizer state (if possible) to shared memory
        # to avoid all the deepcopies (once behavior is better understood)
        self.checkpoint['weights'] = self.model.get_parameters()

    def train(self):
        # initialize shared memory block and add SharedStorage object
        sm_shared_storage = shared_memory.SharedMemory(
            name='shared_storage', create=True, size=sys.getsizeof(SharedStorage)
        )
        shared_storage = sm_shared_storage.buf
        shared_storage = SharedStorage(self.checkpoint, self.config,)

        # initialize shared memory block and add ReplayBuffer object
        sm_replay_buffer = shared_memory.SharedMemory(
            name='replay_buffer', create=True, size=sys.getsizeof(ReplayBuffer)
        )
        replay_buffer = sm_replay_buffer.buf
        replay_buffer = ReplayBuffer(self.checkpoint, self.config,)

        # initialize trainer worker
        self.trainer_worker = mp.Process(
            target=Trainer(self.checkpoint,
                           self.config,
                           ).continuous_update_weights,
            args=(
                shared_storage, 
                replay_buffer,
            ),
            daemon=True,
        )

        # initialize self play workers
        self.self_play_workers = list()
        for batch_ix in range(self.config['num_games']):
            worker = mp.Process(
                target=SelfPlay(self.checkpoint,
                                self.Game,
                                self.config,
                                self.config['seed'] + batch_ix
                                ).continuous_self_play,
                args=(
                    shared_storage, 
                    replay_buffer,
                ),
                daemon=True,
            )
            self.self_play_workers.append(worker)

        # start workers
        [worker.start() for worker in self.self_play_workers]
        self.trainer_worker.start()
