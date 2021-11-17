import xml.etree.ElementTree as ET
import pdb
from decimal import Decimal

treeNet=ET.parse('./environments/sumo/networks/toronto/test.net.xml')
rootNet = treeNet.getroot()

for edge_xml in rootNet.findall('edge'):
    # create edge and edge system objects store, them keyed by id
    if len(edge_xml.findall('lane'))<2:
        rootNet.remove(edge_xml)

breakpoint()

with open('./environments/sumo/networks/toronto/test.net.xml', 'w') as f:
    treeNet.write(f, encoding='unicode')