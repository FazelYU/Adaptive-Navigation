import xml.etree.ElementTree as ET
# import ET
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

def reposition_il(edge_xml,lane_xml,offset):
    position=Decimal(lane_xml.attrib["length"])
    if position>offset:
        position=position-offset
    else:
        if edge_xml.attrib['id'] not in small_edges:
            small_edges[edge_xml.attrib['id']]=position
        position=5
    induction_loops_dic[lane_xml.attrib["id"]]=str(position)

init_traci()

networkModel = RoadNetworkModel(Constants["ROOT"], Constants["Network_XML"])
agent_dic=create_agent_dic()

treeNET = ET.parse('./environments/sumo/networks/toronto/toronto.net.xml')
rootNET=treeNET.getroot()
# treeADD=ET.parse('./environments/sumo/networks/toronto/toronto_additional.add.ET')
# rootADD = treeADD.getroot()
induction_loops_dic={}
small_edges={}

for edge_xml in rootNET.findall('edge'):
    edgeID=edge_xml.attrib['id']
    if edgeID not in networkModel.edge_ID_dic:
        continue
    if len(networkModel.get_edge_connections(edgeID))<2:
        continue
    # create edge and edge system objects store, them keyed by id
    for lane_xml in edge_xml.findall('lane'):
        if Decimal(lane_xml.attrib['speed'])>=50:
            reposition_il(edge_xml,lane_xml,240)
        else:
            reposition_il(edge_xml,lane_xml,70)



rootADD = ET.Element("additional")
for lane in induction_loops_dic:
    edge=traci.lane.getEdgeID(lane)
    assert(edge in networkModel.edge_connection_dic)
    assert(len(networkModel.get_edge_connections(edge))>1)
    assert(networkModel.get_edge_head_node(edge) in agent_dic)
    iLoop=ET.Element("e1Detector")
    iLoop.attrib["id"]="Detector_"+lane
    iLoop.attrib["lane"]=lane
    iLoop.attrib["pos"]=induction_loops_dic[lane]
    iLoop.attrib["freq"]="900"
    iLoop.attrib["file"]="Detector"+lane+".xml"
    rootADD.append(iLoop)
        # breakpoint()

treeADD = ET.ElementTree(rootADD)
breakpoint()

with open('./environments/sumo/networks/toronto/toronto_additional.add.xml', 'w') as f:
    treeADD.write(f, encoding='unicode')