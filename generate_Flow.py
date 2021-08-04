import xml.etree.ElementTree as ET
import pdb
from decimal import Decimal
from environments.sumo.Utils import Constants
from environments.sumo.Utils import Utils
from environments.sumo.model.network import RoadNetworkModel
import os, sys
import traci
import random

def create_agent_dic():
        return {\
                node: [len(networkModel.get_in_edges(node)),len(networkModel.get_out_edges(node))] \
                for node in networkModel.graph.nodes() if \
                does_need_agent(node)
        }
def does_need_agent(node):
    if node==None: 
        return False
    
    if len(networkModel.get_out_edges(node))<2:
        return False
    
    for edge in networkModel.get_in_edges(node):
        if len(networkModel.get_edge_connections(edge))>1:
            return True

    return False

def init_traci():
    sys.path.append(os.path.join(Constants['SUMO_PATH'], os.sep, 'tools'))
    sumoBinary = Constants["SUMO_GUI_PATH"]
    sumoCmd = [sumoBinary, '-S', '-d', Constants['Simulation_Delay'], "-c", Constants["SUMO_CONFIG"]]
    traci.start(sumoCmd)

init_traci()

networkModel = RoadNetworkModel(Constants["ROOT"], Constants["Network_XML"])
agent_list=list(create_agent_dic())


rootTrips = ET.Element("trips")

for index in range(0,2000):
    source_node=random.choice(agent_list)
    sink_node=random.choice(agent_list)
    while (sink_node==source_node):
        sink_node=random.choice(agent_list)

    source_edge=random.choice(networkModel.get_out_edges(source_node))
    sink_edge=random.choice(networkModel.get_in_edges(sink_node))
    trip=ET.Element("trip")
    trip.attrib["origin"]=source_edge
    trip.attrib["destination"]=sink_edge
    rootTrips.append(trip)

treeTrips = ET.ElementTree(rootTrips)
breakpoint()
with open('./environments/sumo/toronto_trips.xml', 'w') as f:
    treeTrips.write(f, encoding='unicode')