import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import time
import gym
import math
import random
import numpy as np


from utilities.data_structures.Config import Config

import torch
import torch.optim as optim
from pytorchGAT.models.definitions.GAT import GAT

# import the simulator, the parser for the map data, network model, environment, and utils of the environment
import traci
import xml.etree.ElementTree as ET
from environments.sumo.model.network import RoadNetworkModel
from environments.sumo.AR_SUMO_Environment import AR_SUMO_Environment
from environments.sumo.Utils import Utils

# import the agents and the trainer
from agents.DQN_agents.DQN_multi_agent import DQN
from agents.Trainer_multi_agent import Trainer

routing_modes=["Q_routing_2_hop","Q_routing_1_hop","Q_routing_0_hop","Q_routing","TTSPWRR","TTSP"]
network_names=["5x6","UES_Manhatan","toronto"]

gpu_num=int(sys.argv[1])
algorithm_num=int(sys.argv[2])
network_num=int(sys.argv[3])

config = Config()

config.use_GPU = True
assert(torch.cuda.is_available())
config.device = torch.device(gpu_num)


config.routing_mode=routing_modes[algorithm_num]
network_name=network_names[network_num]

config.training_mode=True

config.does_need_network_state=config.routing_mode in ["Q_routing_2_hop","Q_routing_1_hop","Q_routing_0_hop"]
config.does_need_network_state_embeding=config.routing_mode in ["Q_routing_2_hop","Q_routing_1_hop"]
config.retain_graph=config.does_need_network_state_embeding
# config.exp_name=config.routing_mode
config.should_load_model= False if  config.routing_mode== "TTSPWRR" or \
                                    config.routing_mode=="TTSP" else \
                                    not config.training_mode
config.should_save_model=False if  config.routing_mode== "TTSPWRR" or \
                                    config.routing_mode=="TTSP" else \
                                    config.training_mode



config.Constants = {
    "NETWORK":network_name,
    "SUMO_PATH" : "/usr/share/sumo", #path to sumo in your system
    "SUMO_GUI_PATH" : "/usr/share/sumo/bin/sumo-gui", #path to sumo-gui bin in your system
    "SUMO_SHELL_PATH":"/usr/share/sumo/bin/sumo",
    "SUMO_CONFIG" : "./environments/sumo/networks/{}/network.sumocfg".format(network_name), #path to your sumo config file
    "ROOT" : "./",
    "Network_XML" : "./environments/sumo/networks/{}/{}.net.xml".format(network_name,network_name),
    'Additional_XML':"./environments/sumo/networks/{}/{}_additional.add.xml".format(network_name,network_name),
    'Analysis_Mode': True,
    'LOG' : False,
    'WARNINGS': False,
    'WHERE':False,
    'Vis_GAT':False,
    'Simulation_Delay' : '0',

    'il_lane_ID_subscribtion_code': 0x51,
    'il_last_step_vc_IDs_subscribtion_code': 0x12,
    'il_last_step_vc_count_subscribtion_code': 0x10,

    'vc_road_ID_subscribtion_code': 0x50,
    'vc_lane_ID_subscribtion_code':0x51,
    }



config.num_episodes_to_run = 600 if config.training_mode else 5

config.Max_number_vc=200
config.uniform_demand_period=5
config.biased_demand_2_uniform_demand_ratio=0.1

config.traffic_period=500
config.max_num_sim_time_step_per_episode=5000

config.demand_scale=1
config.congestion_epsilon=0.25
config.congestion_speed_factor=0.1

if network_name=="5x6":
    config.biased_demand=[['-gneE19','-gneE25']]
elif network_name=="UES_Manhatan" :
    config.biased_demand=[['AA7AB7','BM0BN0']]
else:
    config.biased_demand=[['23973402#0','435629850']] #list of the biased O-D demands 

config.uniform_demands=[
        [trip_xml.attrib["origin"],trip_xml.attrib["destination"]] 
            for trip_xml in ET.parse('./environments/sumo/{}_trips.xml'.format(network_name)).getroot().findall("trip")
            ]

config.next_uniform_demand_index=0
config.num_biased_vc_dispatched=0
config.num_uniform_vc_dispatched=0
# -------------------------------------------------
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = True
config.model_version="Feb22"


if config.routing_mode=="Q_routing":
    num_GAT_layers=1 #Dummy initialization
    config.network_embed_size=0
    num_GAT_features_per_layer=[2,5] #Dummy initialization
    DQN_linear_hidden_units=[5,4]

if config.routing_mode=="Q_routing_0_hop":
    num_GAT_layers=1  #Dummy initialization
    config.network_embed_size=5
    num_GAT_features_per_layer=[2,5] #Dummy initialization
    DQN_linear_hidden_units=[8,6]


if config.routing_mode=="Q_routing_1_hop":
    num_GAT_layers=1
    config.network_embed_size=15
    num_GAT_features_per_layer=[5,10]
    DQN_linear_hidden_units=[15,7]

if config.routing_mode=="Q_routing_2_hop":
    num_GAT_layers=2
    config.network_embed_size=35
    num_GAT_features_per_layer=[5,15,15]
    DQN_linear_hidden_units=[12,9,6]

num_GAT_heads_per_layer=[1]*num_GAT_layers

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
        "linear_hidden_units": DQN_linear_hidden_units,
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

config.network_state=[]
config.edge_index=[]
config.seed = 1
config.envm_seed=100


def init_traci():
    """ initializes the simulator"""
    sys.path.append(os.path.join(config.Constants['SUMO_PATH'], os.sep, 'tools'))
    sumoBinary = config.Constants["SUMO_GUI_PATH"]
    sumoCmd = [sumoBinary, '-S', '-d', config.Constants['Simulation_Delay'], "-c", config.Constants["SUMO_CONFIG"],"--no-warnings","true"]
    traci.start(sumoCmd)

init_traci()       
config.network = RoadNetworkModel(config.Constants["ROOT"], config.Constants["Network_XML"])
config.utils=Utils(config)

config.environment = AR_SUMO_Environment(config)
config.network_state_size=config.environment.utils.get_network_state_size()

if config.routing_mode in ["Q_routing_1_hop","Q_routing_2_hop"]:
    assert(config.network_state_size==num_GAT_features_per_layer[0])

gat = GAT(
        config=config,
        num_of_layers=config.hyperparameters["GAT"]['num_of_layers'],
        num_heads_per_layer=config.hyperparameters["GAT"]['num_heads_per_layer'],
        num_features_per_layer=config.hyperparameters["GAT"]['num_features_per_layer'],
        add_skip_connection=config.hyperparameters["GAT"]['add_skip_connection'],
        bias=config.hyperparameters["GAT"]['bias'],
        dropout=config.hyperparameters["GAT"]['dropout'],
        log_attention_weights=not config.training_mode
    ).to(config.device)

gat.train()
config.GAT=gat
config.GAT_parameters=gat.parameters()
config.GAT_optim=optim.Adam(config.GAT_parameters,
                            lr=config.hyperparameters["GAT"]["lr"],
                            eps=1e-4)
                            # weight_decay=config.hyperparameters["GAT"]['weight_decay'])
# Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


if __name__== '__main__':
    AGENTS = [DQN] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()



