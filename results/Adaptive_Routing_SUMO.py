import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from agents.DQN_agents.DQN_multi_agent import DQN
from environments.sumo.AR_SUMO_Environment import AR_SUMO_Environment as envm
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
import xml.etree.ElementTree as ET

treeTrips=ET.parse('./environments/sumo/toronto_trips.xml')
rootTrips = treeTrips.getroot()

device = torch.device("cpu" if torch.cuda.is_available() else "cpu") #device is cpu
config = Config()
config.seed = 1

config.use_GPU = False
config.exp_name="toronto"
config.should_load_model=False
config.should_save_model=False
routing_modes=["Q_routing","TTSPWRR","TTSP"]
config.routing_mode=routing_modes[1]
# -------------------------------------------------
config.num_episodes_to_run = 10
config.Max_number_vc=200
config.uniform_demand_period=5
config.biased_demand_period=50
config.traffic_period=500
config.episode_period=5000

config.demand_scale=1
config.congestion_epsilon=0.25
config.congestion_speed_factor=0.1

config.biased_demand=[['195203644','-23973402#5']] #list of the biased O-D demands 
config.uniform_demands=[
        [trip_xml.attrib["origin"],trip_xml.attrib["destination"]] 
            for trip_xml in rootTrips.findall("trip")
            ]
config.next_uniform_demand_index=0
# -------------------------------------------------
config.file_to_save_data_results = "Data_and_Graphs/Adaptive_Routing.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/Adaptive_Routing.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = True
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
        "epsilon_decay_rate_denominator": config.num_episodes_to_run/100,
        "stop_exploration_episode":config.num_episodes_to_run-10,
        "random_episodes_to_run":0,
        "linear_hidden_units": [3,3],
        "learning_rate": 0.01,
        "buffer_size": 10000,
        "batch_size": 64,
        "final_layer_activation": None,
        # "columns_of_data_to_be_embedded": [0],
        # "embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "tau": 0.01,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "exploration_cycle_episodes_length": None,
        "learning_iterations": 1,
        "clip_rewards": False
    },
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
config.GAT_parameters=gat.parameters()
config.GAT=gat

config.environment = envm(config,device=device)

if __name__== '__main__':
    AGENTS = [DQN] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()