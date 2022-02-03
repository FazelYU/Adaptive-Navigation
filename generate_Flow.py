import xml.etree.ElementTree as ET
import pdb
from decimal import Decimal
import Constants
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
    sys.path.append(os.path.join(Constants.cons['SUMO_PATH'], os.sep, 'tools'))
    sumoBinary = Constants.cons["SUMO_GUI_PATH"]
    sumoCmd = [sumoBinary, '-S', '-d', Constants.cons['Simulation_Delay'], "-c", Constants.cons["SUMO_CONFIG"]]
    traci.start(sumoCmd)

init_traci()

networkModel = RoadNetworkModel(Constants.cons["ROOT"], Constants.cons["Network_XML"])
agent_list=list(create_agent_dic())


rootTrips = ET.Element("trips")

if Constants.cons["NETWORK"]=="5x6":
    biased_demand=['-gneE19','-gneE25']
else:
    biased_demand=['23973402#0','435629850']

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
    if index%10==0:
        source_edge=biased_demand[0]
        sink_edge=biased_demand[1]
        trip=ET.Element("trip")
        trip.attrib["origin"]=source_edge
        trip.attrib["destination"]=sink_edge
        rootTrips.append(trip)

treeTrips = ET.ElementTree(rootTrips)
breakpoint()
with open('./environments/sumo/{}_trips.xml'.format(Constants.cons["NETWORK"]), 'w') as f:
    treeTrips.write(f, encoding='unicode')