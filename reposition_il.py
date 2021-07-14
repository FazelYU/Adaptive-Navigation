import xml.etree.ElementTree as ET
import pdb
from decimal import Decimal

treeADD=ET.parse('./environments/sumo/networks/4x3/4x3_additional.add.xml')
rootADD = treeADD.getroot()

rootNET = ET.parse('./environments/sumo/networks/4x3/4x3.net.xml').getroot()
lane_dic={}

for edge_xml in rootNET.findall('edge'):
    # create edge and edge system objects store, them keyed by id
    for lane_xml in edge_xml.findall('lane'):
    	lane_dic[lane_xml.attrib["id"]]=Decimal(lane_xml.attrib["length"])


for addtional in rootADD.findall('e1Detector'): 
	addtional.attrib['pos']=str(lane_dic[addtional.attrib['lane']]-25)

breakpoint()

with open('./environments/sumo/networks/4x3/4x3_additional.add.xml', 'w') as f:
    treeADD.write(f, encoding='unicode')