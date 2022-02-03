network_name="UES_Manhatan" 
cons = {
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