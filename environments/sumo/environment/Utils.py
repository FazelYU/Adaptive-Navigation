import numpy

import torch
import json
import random
import traci
from inspect import currentframe, getframeinfo
import networkx as nx


Constants = {
    "SUMO_PATH" : "/usr/share/sumo", #path to sumo in your system
    "SUMO_GUI_PATH" : "/usr/share/sumo/bin/sumo-gui", #path to sumo-gui bin in your system
    "SUMO_CONFIG" : "./environments/sumo/networks/3x3/network.sumocfg", #path to your sumo config file
    "ROOT" : "./",
    "Network_XML" : "./environments/sumo/networks/3x3/test1.net.xml",
    'LOG' : False,
    'WARNINGS': False,
    'WHERE':False,
    'Simulation_Delay' : '10'
    }



class Utils(object):
    """docstring for Utils"""
    def __init__(self,environment,network,Num_Flows,device,GAT,embed_network):
        super(Utils, self).__init__()
        self.network=network
        self.Num_Flows=Num_Flows

        # self.pressure_matrix=[[0]*dim for i in range(dim)]
        self.network_embd_dim=9
        self.environment=environment
        # self.vc_count_dic=self.creat_vc_count_dic()
        self.Num_Flow_Types=3
        self.slow_vc_speed=7
        self.fast_vc_speed=2*self.slow_vc_speed
        self.device=device
        self.adjacency_list_dict={}
        self.node_features_dic={}

        self.gat=GAT
        self.embed_network=embed_network
        self.edge_ID_dic=self.create_edge_ID_dic()
        self.node_dic=self.create_node_dic()
        self.agent_dic=self.create_agent_dic()
        self.agent_dic_index = {list(self.agent_dic)[idx]: idx for idx in range(len(self.agent_dic))}
        self.sink_edge_list=[self.get_edge_ID(edge) \
                        for edge in self.network.graph.out_edges \
                        if self.network.graph.out_degree(edge[1])==0 and \
                        self.network.graph.in_degree(edge[1])==1]
        self.sink_nodes_list=[self.get_edge_tail_node(sink_edge) for sink_edge in self.sink_edge_list]
        self.sink_nodes_index={self.sink_nodes_list[idx]:idx for idx in range(len(self.sink_nodes_list))}
        self.source_edge_list=[self.get_edge_ID(edge) \
                        for edge in self.network.graph.out_edges \
                        if self.network.graph.out_degree(edge[0])==1 and \
                        self.network.graph.in_degree(edge[0])==0
                        ] 

        self.all_pairs_shortest_path= dict(nx.all_pairs_dijkstra_path_length(self.network.graph))
        self.sink_embed_dim=len(self.sink_nodes_list)
        self.network_embed_dim=1
        self.state_dim=self.sink_embed_dim+self.network_embed_dim

    
    def get_state(self,source,sink):
        dest_embed=[0]*len(self.sink_nodes_list)
        dest_embed[self.sink_nodes_index[sink]]=1

        embeding=dest_embed+self.get_network_state()
        return {"agent_id": source,
                "embeding": embeding}

    def get_network_state(self):
        speed=traci.lane.getMaxSpeed('gneE6_0')
        if speed==2:
            return [0]
        else:
            return[1]
    # def get_state(self,origin,destination,vehicle_id):
    #     reshaped_origin=self.res_road(origin)
    #     reshaped_destination=self.res_road(destination)

    #     one_hot_flow_type=[0]*self.Num_Flow_Types
    #     flow_type=self.flow_types_dic[self.get_flow_id(vehicle_id)]
    #     one_hot_flow_type[flow_type]=1
    #     state=self.state2torch(reshaped_origin+reshaped_destination+one_hot_flow_type)

    #     if self.embed_network:
    #         network_embd=self.evolved_node_features.view(1,-1)
    #         return torch.cat((state,network_embd),1)
    #     else:
    #         return state

    def get_state_diminsion(self): 
        return self.state_dim 
    
    def get_flow_id(self,vehicle_id):
        splited=vehicle_id.split('_')
        assert(len(splited)==2)
        return int(splited[1])




    def state2torch(self,state):
        state=torch.tensor(state, device=self.device, dtype=torch.float)
        return state.unsqueeze(0)

# GAT--------------------------------------------------
    def update_vc_count_dic(self):
        lane_vc_count_dic=self.environment.eng.get_lane_vehicle_count()
        self.refresh_vc_count_dic()
        for lane in lane_vc_count_dic:
            road= self.road2int(self.lane2road(lane))
            # if road==10:
            #   breakpoint()
            self.vc_count_dic[road]+=lane_vc_count_dic[lane]

    def creat_vc_count_dic(self):
        lane_vc_count_dic=self.environment.eng.get_lane_vehicle_count()
        vc_count_dic={}
        for lane in lane_vc_count_dic:
            road= self.road2int(self.lane2road(lane))
            if not road in vc_count_dic: vc_count_dic[road]=0
        return vc_count_dic

    def refresh_vc_count_dic(self):
        
        for road in self.vc_count_dic:self.vc_count_dic[road]=0
    
    def update_pressure_matrix(self):
        self.update_vc_count_dic()
        for row in range(0,self.dim):
            for column in range(0,self.dim):
                try:
                    self.pressure_matrix[column][row]=self.get_pressure(column,row)
                except Exception as e:
                    print(e)
                    breakpoint()

    def get_press_embd(self,road):
        press_embd=[0]*self.press_embd_dim

        nroad=self.move(road,0)
        nroad=[x-1 for x in nroad]
        if nroad[0]<self.dim and nroad[1]<self.dim:
            press_embd[0]=self.pressure_matrix[nroad[0]][nroad[1]]

        nroad=self.move(road,1)
        nroad=[x-1 for x in nroad]
        if nroad[0]<self.dim and nroad[1]<self.dim:
            press_embd[0]=self.pressure_matrix[nroad[0]][nroad[1]]

        nroad=self.move(road,2)
        nroad=[x-1 for x in nroad]
        if nroad[0]<self.dim and nroad[1]<self.dim:
            press_embd[0]=self.pressure_matrix[nroad[0]][nroad[1]]


        return press_embd

    def get_pressure(self,column,row):
        # column and rows are 1-indexed
        row+=1
        column+=1

        in_roads=[
            [column-1,row,0],
            [column+1,row,2],
            [column,row-1,1],
            [column,row+1,3]
        ]
        out_roads=[
            [column,row,0],
            [column,row,1],
            [column,row,2],
            [column,row,3],
        ]
        pressure=0

        for road in in_roads:
            pressure+=self.vc_count_dic[self.road2int(road)]

        for road in out_roads:
            pressure-=self.vc_count_dic[self.road2int(road)]

        return pressure

    def get_edge_index(self,add_self_edges=True):
        num_of_nodes=len(self.adjacency_list_dict)
        source_nodes_ids, target_nodes_ids = [], []
        seen_edges = set()

        for src_node, neighboring_nodes in self.adjacency_list_dict.items():
            for trg_node in neighboring_nodes:
                # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
                if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                    source_nodes_ids.append(src_node)
                    target_nodes_ids.append(trg_node)
                    seen_edges.add((src_node, trg_node))

        # shape = (2, E), where E is the number of edges in the graph
        edge_index = numpy.row_stack((source_nodes_ids, target_nodes_ids))

        return torch.tensor(edge_index,dtype=torch.long,device=self.device)

    def update_and_evolve_node_features(self):
        self.update_node_features()
        edge_index=self.get_edge_index()
        node_features=self.get_node_features()
        self.evolved_node_features = self.gat((node_features, edge_index))[0]

    def update_node_features(self):
        self.update_pressure_matrix()
        for row in range(0,self.dim):
            for column in range(0,self.dim):
                node_id=(column+row*3)
                try:
                    self.node_features_dic[node_id][0]=(self.pressure_matrix[column][row])
                except Exception as e:
                    print(e)
                    breakpoint()
    
    def get_node_features(self):
        return torch.tensor([*self.node_features_dic.values()],dtype=torch.float,device=self.device)




 # traci utils-------------------------------------------------

        
    def create_edge_ID_dic(self):
        edge_ID_dic={self.get_edge_ID(edge): edge 
        for edge in self.network.graph.edges() if edge != (None,None)}
        return edge_ID_dic

    def create_node_dic(self):
        """
        input: void
        return: a dictionary for all intersections in the map: 
                                the keys : intersection ids
                                the values : list of out going roads that are connected to the intersection
        """    
        node_dic={ \
                    node: \
                            [self.get_edge_ID(edge) \
                            for edge in self.network.graph.out_edges(node)]\
                    for node in self.network.graph.nodes() \
                    if node!=None \
                    }
        return node_dic

    def create_agent_dic(self):
        return {\
                node: len(self.node_dic[node]) \
                for node in self.node_dic if \
                len(self.node_dic[node])>1\
        }

    def generate_random_trips(self,num_trips,num_vehiles_per_trip=None):
        if num_vehiles_per_trip==None:
            num_vehiles_per_trip=[1 for i in range(0,num_trips)]
        
        for i in range(0,num_trips):
            source=random.choice(self.source_edge_list)
            sink_edge=random.choice(self.sink_edge_list)
            traci.route.add("trip_{}".format(i),[source,sink_edge])
            # stage=traci.simulation.findRoute(source,sink_edge)
            # breakpoint()
            # edgeTT=sum([traci.edge.getTraveltime(edgeID) for edgeID in traci.route.getEdges("trip_{}".format(i)) ])
            # breakpoint()
            for j in range(0,num_vehiles_per_trip[i]):
                traci.vehicle.add("vehicle_{}_{}".format(i,j),"trip_{}".format(i))
    
    def generate_random_trip(self,id):
        source_edge=random.choice(self.source_edge_list)
        sink_edge=random.choice(self.sink_edge_list)
        traci.route.add("trip_{}".format(id),[source_edge,sink_edge])
        traci.vehicle.add("vehicle_{}".format(id),"trip_{}".format(id))
        return "vehicle_{}".format(id),source_edge,self.get_edge_tail_node(sink_edge)

    def get_destination(self,vc):
        route_tail=traci.vehicle.getRoute(vc)[-1]
        return self.get_edge_tail_node(route_tail)
    
    def get_edge_tail_node(self,edge):
        return self.edge_ID_dic[edge][1]

    def get_edge_path(self,edgeID):
        """receives edge ID returns edge"""
        path=[edgeID]
        while self.get_edge_tail_node(path[-1]) not in list(self.agent_dic)+self.sink_nodes_list:
            path.append(self.get_out_edges(self.get_edge_tail_node(path[-1]))[0])
        return path

    def get_edge_path_tail_node(self,edge):
        return self.get_edge_tail_node(self.get_edge_path(edge)[-1])
  

    def get_next_road_IDs(self,node,action_edge_index):
        action_edge_ID=self.get_out_edges(node)[action_edge_index]
        return self.get_edge_path(action_edge_ID)

    def get_out_edges(self,node):
        return self.node_dic[node]
    
    def get_edge_ID(self,edge):
        return self.network.graph.get_edge_data(*edge)['edge'].id
   
    def get_edge(self,edgeID):
        return self.edge_ID_dic[edgeID]

    def is_valid(self,source):
        return len(self.get_out_edges(source))!=0
   
    def get_time(self):
        return traci.simulation.getTime()

    def get_edge_weight(self,edge):
        return self.network.graph.get_edge_data(*edge)['weigh']
    

    def get_shortest_path_time(self,source,destination):
        return self.all_pairs_shortest_path[source][destination]
    #helper-------------------------------------------------

    def log(self, log_str, type='info'):
        if Constants['LOG']:
            if type == 'info':
                print('-Info- ' + log_str)
            if type=='err':
                if Constants['WHERE']:
                    print(self._where())
                print('-Error- ' + log_str)

        if type== 'warn' and Constants['WARNINGS']:
            if Constants['WHERE']:
                print(self._where())
            print('-Warning- ' + log_str)   


    def _where(self):
        cf=currentframe()
        return "@ file:"+getframeinfo(cf).filename+" line:"+cf.f_back.f_lineno