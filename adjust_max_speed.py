import xml.etree.ElementTree as ET
from decimal import Decimal
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
treeNET=ET.parse('./environments/sumo/networks/toronto/toronto.net.xml')
rootNET = treeNET.getroot()
lane_dic={}

for edge_xml in rootNET.findall('edge'):
    edgeID=edge_xml.attrib['id']
    if edgeID not in networkModel.edge_ID_dic:
        continue

    for lane_xml in edge_xml.findall('lane'):
        if Decimal(lane_xml.attrib["speed"])>25:
            lane_xml.attrib["speed"]="11.11"
           
breakpoint()

with open('./environments/sumo/networks/toronto/toronto.net.xml', 'w') as f:
    treeNET.write(f, encoding='unicode')


