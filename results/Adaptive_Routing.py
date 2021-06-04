import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from agents.DQN_agents.DQN_multi_agent import DQN
from environments.Adaptive_Routing_Environment import Adaptive_Routing_Environment
from agents.Trainer_multi_agent import Trainer
from utilities.data_structures.Config import Config

import time
import gym
import math
import random
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from pytorchGAT.models.definitions.GAT import GAT


# import environments.Adaptive_Routing_Environment_D
# from environments.Utils import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device is cpu
# device="cpu"
config = Config()
config.seed = 1

# breakpoint()
# num_possible_states =config.environment.utils.get_state_diminsion()
# embedding_dimensions = [[num_possible_states, 20]]
# print("Num possible states ", num_possible_states)

config.num_episodes_to_run = 100
config.file_to_save_data_results = "Data_and_Graphs/Adaptive_Routing.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/Adaptive_Routing.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = False

config.hyperparameters = {
    "GAT":{
    'num_of_epochs': 10000, 
    'patience_period': 1000,
    'lr': 0.005, 
    'weight_decay': 0.0005, 
    'should_test': False, 
    'dataset_name': 'CORA', 
    'should_visualize': False, 
    'enable_tensorboard': False, 
    'console_log_freq': 100, 
    'checkpoint_freq': 1000, 
    'num_of_layers': 1, 
    'num_heads_per_layer': [2], 
    'num_features_per_layer': [1, 1], 
    'add_skip_connection': False, 
    'bias': True, 
    'dropout': 0.6,
    },

    "DQN_Agents": {
        "linear_hidden_units": [30, 10],
        "learning_rate": 0.01,
        "buffer_size": 10000,
        "batch_size": 64,
        "final_layer_activation": "None",
        # "columns_of_data_to_be_embedded": [0],
        # "embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 10,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.01,
        "exploration_cycle_episodes_length": None,
        "learning_iterations": 1,
        "clip_rewards": False
    },

    "SNN_HRL": {
        "SKILL_AGENT": {
            "num_skills": 20,
            "regularisation_weight": 1.5,
            "visitations_decay": 0.9999,
            "episodes_for_pretraining": 300,
            "batch_size": 256,
            "learning_rate": 0.001,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            # "embedding_dimensions": [embedding_dimensions[0],
            #                          [20, 6]],
            "batch_norm": False,
            "gradient_clipping_norm": 2,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False
        },

        "MANAGER": {
            "timesteps_before_changing_skill": 6,
            "linear_hidden_units": [10, 5],
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "batch_size": 256,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            # "embedding_dimensions": embedding_dimensions,
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 50,
            "discount_rate": 0.99,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False

        }

    },

    "Actor_Critic_Agents": {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],

        "columns_of_data_to_be_embedded": [0],
        # "embedding_dimensions": embedding_dimensions,
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 50.0,
        "normalise_rewards": True,
        "clip_rewards": False

    },


    "DIAYN": {

        "num_skills": 5,
        "DISCRIMINATOR": {
            "learning_rate": 0.01,
            "linear_hidden_units": [20, 10],
            "columns_of_data_to_be_embedded": [0],
            # "embedding_dimensions": embedding_dimensions,
        },

        "AGENT": {
            "learning_rate": 0.01,
            "linear_hidden_units": [20, 10],
        }
    },


    "HRL": {
        "linear_hidden_units": [10, 5],
        "learning_rate": 0.01,
        "buffer_size": 40000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        # "embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 400,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.01

    }
}


gat = GAT(
        num_of_layers=config.hyperparameters["GAT"]['num_of_layers'],
        num_heads_per_layer=config.hyperparameters["GAT"]['num_heads_per_layer'],
        num_features_per_layer=config.hyperparameters["GAT"]['num_features_per_layer'],
        add_skip_connection=config.hyperparameters["GAT"]['add_skip_connection'],
        bias=config.hyperparameters["GAT"]['bias'],
        dropout=config.hyperparameters["GAT"]['dropout'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

gat.train()

config.environment = Adaptive_Routing_Environment(GAT=gat,dim=3,encode="one_hot", embed_press=True ,Num_Flows=5,skip_routing=[],random_trips=True,Max_Sim_Time=600,device=device,Log=False,rolling_window=10)


if __name__== '__main__':
    AGENTS = [DQN] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()