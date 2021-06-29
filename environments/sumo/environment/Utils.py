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
    'LOG' : True,
    'WARNINGS': False,
    'WHERE':False,
    'Simulation_Delay' : '100'
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
        self.stat_dimintion=4
        self.nodes=list(self.network.graph.nodes())
        self.nodesIndex = {self.nodes[idx]: idx for idx in range(len(self.nodes))}
        self.edge_ID_dic=self.create_edge_ID_dic()
        self.node_dic=self.create_node_dic()
        self.sink_list=[self.get_edge_ID(edge) \
                        for edge in self.network.graph.out_edges \
                        if self.network.graph.out_degree(edge[1])==0 and \
                        self.network.graph.in_degree(edge[1])==1]
        self.source_list=[self.get_edge_ID(edge) \
                        for edge in self.network.graph.out_edges \
                        if self.network.graph.out_degree(edge[0])==1 and \
                        self.network.graph.in_degree(edge[0])==0
                        ] 

        self.all_pairs_shortest_path= dict(nx.all_pairs_dijkstra_path_length(self.network.graph))

    
    def get_state(self,source,destination):
        dest_embed=[0]*len(self.nodes)
        dest_embed[self.nodesIndex[destination]]=1
        return {"router": source,
                "embed": dest_embed}

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
        
        return self.stat_dimintion
            
    def get_flow_id(self,vehicle_id):
        splited=vehicle_id.split('_')
        assert(len(splited)==2)
        return int(splited[1])

    def generate_random_trips(self,num_trips,num_vehiles_per_trip=None):
        if num_vehiles_per_trip==None:
            num_vehiles_per_trip=[1 for i in range(0,num_trips)]
        
        for i in range(0,num_trips):
            source=random.choice(self.source_list)
            sink=random.choice(self.sink_list)
            traci.route.add("trip_{}".format(i),[source,sink])
            # stage=traci.simulation.findRoute(source,sink)
            # breakpoint()
            # edgeTT=sum([traci.edge.getTraveltime(edgeID) for edgeID in traci.route.getEdges("trip_{}".format(i)) ])
            # breakpoint()
            for j in range(0,num_vehiles_per_trip[i]):
                traci.vehicle.add("vehicle_{}_{}".format(i,j),"trip_{}".format(i))
    def generate_random_trip(self,id):
        source=random.choice(self.source_list)
        sink=random.choice(self.sink_list)
        traci.route.add("trip_{}".format(id),[source,sink])
        traci.vehicle.add("vehicle_{}".format(id),"trip_{}".format(id))



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


    def get_destination(self,vc):
        route_tail=traci.vehicle.getRoute(vc)[-1]
        return self.get_edge_tail_node(route_tail)
    
    def get_edge_tail_node(self,edge):
        return self.edge_ID_dic[edge][1]

    def get_edge_path_ID(self,edgeID):
        """receives edge ID returns edge"""
        path=[edgeID]
        while self.network.graph.out_degree(self.get_edge_tail_node(path[-1]))==1:
            path.append(self.get_out_edges(self.get_edge_tail_node(path[-1]))[0])
        return path

    def get_next_road_IDs(self,source,action):
        action_edge_ID=self.get_out_edges(source)[action]
        return self.get_edge_path_ID(action_edge_ID)


    def get_edge_path_tail_node(self,edge):
        return self.get_edge_tail_node(self.get_edge_path_ID(edge)[-1])

    def get_out_edges(self,intersection):
        return self.node_dic[intersection]
    
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