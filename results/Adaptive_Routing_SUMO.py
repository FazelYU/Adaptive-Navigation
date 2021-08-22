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

config = Config()
config.use_GPU = True
config.training_mode=False

routing_modes=["Q_routing_2_hop","Q_routing_1_hop","Q_routing_0_hop","Q_routing","TTSPWRR","TTSP"]
config.routing_mode=routing_modes[5]
# if config.routing_mode in ["TTSPWRR","TTSP"]:
#     config.training_mode=False
config.does_need_network_state=config.routing_mode in ["Q_routing_2_hop","Q_routing_1_hop","Q_routing_0_hop"]
config.does_need_network_state_embeding=config.routing_mode in ["Q_routing_2_hop","Q_routing_1_hop"]
config.retain_graph=config.does_need_network_state_embeding

config.network_name="toronto"
config.exp_name=config.routing_mode

assert(torch.cuda.is_available())
config.device = torch.device(0)
config.seed = 1
config.envm_seed=100
config.should_load_model= False if  config.routing_mode== "TTSPWRR" or \
                                    config.routing_mode=="TTSP" else \
                                    not config.training_mode
config.should_save_model=False if  config.routing_mode== "TTSPWRR" or \
                                    config.routing_mode=="TTSP" else \
                                    config.training_mode

# -------------------------------------------------
config.num_episodes_to_run = 800 if config.training_mode else 5

config.Max_number_vc=200
config.uniform_demand_period=5
config.biased_demand_2_uniform_demand_ratio=0.1

config.traffic_period=500
config.max_num_sim_time_step_per_episode=5000

config.demand_scale=1
config.congestion_epsilon=0.25
config.congestion_speed_factor=0.1

config.biased_demand=[['23973402#0','435629850']] #list of the biased O-D demands 
# config.biased_demand=[['-gneE19','-gneE25']]

config.uniform_demands=[
        [trip_xml.attrib["origin"],trip_xml.attrib["destination"]] 
            for trip_xml in rootTrips.findall("trip")
            ]

config.next_uniform_demand_index=0
config.num_biased_vc_dispatched=0
config.num_uniform_vc_dispatched=0
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
config.model_version="V2"

num_GAT_layers=1 if config.routing_mode=="Q_routing_1_hop" else 2
num_GAT_heads_per_layer=[3]*num_GAT_layers
num_GAT_features_per_layer=[4]*(num_GAT_layers+1)

config.hyperparameters = {
    "GAT":{
    'lr': 0.01, 
    # 'weight_decay': 0.0005, 
    'num_of_layers': num_GAT_layers, 
    'num_heads_per_layer': num_GAT_heads_per_layer, 
    'num_features_per_layer': num_GAT_features_per_layer, 
    'add_skip_connection': False, 
    'bias': True, 
    'dropout': 0.6,
    },

    "DQN_Agents": {
        "epsilon_decay_rate_denominator": config.num_episodes_to_run/100,
        "stop_exploration_episode":config.num_episodes_to_run-10,
        "random_episodes_to_run":0,
        "linear_hidden_units": [8,6],
        "learning_rate": 0.01,
        "buffer_size": 10000,
        "batch_size": 64,
        "final_layer_activation": None,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "num-new-exp-to-learn":1,
        "tau": 0.01,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "exploration_cycle_episodes_length": None,
        "learning_iterations": 1,
        "clip_rewards": False
    },

}


gat = GAT(
        config=config,
        num_of_layers=config.hyperparameters["GAT"]['num_of_layers'],
        num_heads_per_layer=config.hyperparameters["GAT"]['num_heads_per_layer'],
        num_features_per_layer=config.hyperparameters["GAT"]['num_features_per_layer'],
        add_skip_connection=config.hyperparameters["GAT"]['add_skip_connection'],
        bias=config.hyperparameters["GAT"]['bias'],
        dropout=config.hyperparameters["GAT"]['dropout'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(config.device)

gat.train()
config.GAT=gat
config.GAT_parameters=gat.parameters()
config.GAT_optim=optim.Adam(config.GAT_parameters,
                            lr=config.hyperparameters["GAT"]["lr"],
                            eps=1e-4)
                            # weight_decay=config.hyperparameters["GAT"]['weight_decay'])
# Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
config.network_state=[]
config.edge_index=[]

config.environment = envm(config)

if __name__== '__main__':
    AGENTS = [DQN] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()