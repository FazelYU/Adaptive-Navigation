import xml.etree.ElementTree as ET
import pdb
from decimal import Decimal
from environments.sumo.Utils import Constants
from environments.sumo.Utils import Utils
from environments.sumo.model.network import RoadNetworkModel
import os, sys
import traci

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
agent_dic=create_agent_dic()
breakpoint()
treeNET=ET.parse('./environments/sumo/networks/toronto/toronto.net.xml')
rootNET = treeNET.getroot()
lane_dic={}

for node_xml in rootNET.findall('junction'):
    # create edge and edge system objects store, them keyed by id
    if node_xml.attrib["type"]=="traffic_light" and node_xml.attrib["id"] not in agent_dic:
        node_xml.attrib["type"]="priority"

        for tl_xml in rootNET.findall('tlLogic'):
            if node_xml.attrib["id"] in tl_xml.attrib["id"]:
                rootNET.remove(tl_xml)

        for connection_xml in rootNET.findall('connection'):
            if "tl" in connection_xml.attrib:
                if node_xml.attrib["id"] in connection_xml.attrib["tl"]:
                    connection_xml.attrib.pop('tl')

                



with open('./environments/sumo/networks/toronto/toronto.net.xml', 'w') as f:
    treeNET.write(f, encoding='unicode')


