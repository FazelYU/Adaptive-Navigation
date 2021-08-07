import numpy

import torch
import json
import random
import traci
from inspect import currentframe, getframeinfo
import networkx as nx
import pymorton as pm

import math


Constants = {
    "EXP":"3x3",
    "SUMO_PATH" : "/usr/share/sumo", #path to sumo in your system
    "SUMO_GUI_PATH" : "/usr/share/sumo/bin/sumo-gui", #path to sumo-gui bin in your system
    "SUMO_SHELL_PATH":"/usr/share/sumo/bin/sumo",
    "SUMO_CONFIG" : "./environments/sumo/networks/toronto/network.sumocfg", #path to your sumo config file
    "ROOT" : "./",
    "Network_XML" : "./environments/sumo/networks/toronto/toronto.net.xml",
    'Analysis_Mode': True,
    'LOG' : True,
    'WARNINGS': False,
    'WHERE':False,
    'Simulation_Delay' : '0',

    'il_lane_ID_subscribtion_code': 0x51,
    'il_last_step_vc_IDs_subscribtion_code': 0x12,
    'il_last_step_vc_count_subscribtion_code': 0x10,

    'vc_road_ID_subscribtion_code': 0x50,
    'vc_lane_ID_subscribtion_code':0x51,
    # # 'congestion_prone_arterials':{1:['207223487#0','124690744#0','30430773#0','4281719#0','43128290','gneE18'],
    # #                             -1:['-gneE18','204493904#0','-4281719#2','-234733018#0','-354354332#2','-207223487#1','-33910433#2'],
    # #                             2:['695753527#1','277877819#0','263506114#5','33023072#0.23','33023075#0','445695121','33024556#0','33025451#0','33025456','449550551#0']
    # #                             -2:['-695753527#1','-277877819#2','-263506114#9.27','-33023072#3','-33023075#4','-33024553#3','-33024556#1','-33025451#1','-33025456','-449550551#2']
    # #                             3:['445865881','4651746#0','222151618#5','5686883#0','220918438#0','446971622#0','4321248#0','35159934#0','248489460#0','35155500','35155501','-446994466#4']
    # #                             -3:[]
    # #                             4:[]
    # #                             -4:[]
    # #                             5:[]
    # #                             -5:[]
    # #                             6:[]
    # #                             -6:[]
    # #                                 }
    # 'congestion_prone_arterials':{
    # 1:['21631714','24959524','20979763','20964581','20964462','20964582','20953780','gneJ0']
    # }

    }

class Utils(object):
    """docstring for Utils"""
    def __init__(self,config,environment,network,Num_Flows,GAT,embed_network):
        super(Utils, self).__init__()
        self.config=config
        torch.autograd.set_detect_anomaly(Constants["Analysis_Mode"])
        self.seeded_random_generator=numpy.random.RandomState(config.envm_seed)
        # breakpoint()
        self.network=network
        self.Num_Flows=Num_Flows

        # self.pressure_matrix=[[0]*dim for i in range(dim)]
        self.network_embd_dim=9
        self.environment=environment
        # self.vc_count_dic=self.creat_vc_count_dic()
        self.Num_Flow_Types=3
        self.slow_vc_speed=7
        self.fast_vc_speed=2*self.slow_vc_speed

        self.gat=GAT
        self.embed_network=embed_network
        

        self.agent_dic=self.create_agent_dic()#{node:[len(in edges), len(out edges)]}
        self.agent_id_embedding_dic,self.agnet_id_embedding_size=self.create_agent_id_embedding_dic()
        self.agent_list=list(self.agent_dic)
        self.agent_index_dic={self.agent_list[idx]:idx for idx in range(len(self.agent_list))}
        self.agent_path_dic=self.create_agent_path_dic()
        self.agent_adjacency_list_dict=self.create_agent_adjacency_list_dic()
        self.max_len_out_edges=max([self.agent_dic[agent_id][1] for agent_id in self.agent_dic])
        self.edge_index=self.create_edge_index()
        self.agents_state=self.create_agents_state()
        if  self.config.routing_mode=='Q_routing_1_hop' or\
            self.config.routing_mode=='Q_routing_2_hop':
            self.aggregated_agents_state=self.graph_attention_network(self.edge_index,self.agents_state)

        self.edge_action_mask_dic=self.create_edge_action_mask_dic()
        self.induction_loops=[il for il in traci.inductionloop.getIDList() if "TLS" not in il]

        for il in self.induction_loops:
            traci.inductionloop.subscribe(il,[\
                                            Constants['il_last_step_vc_count_subscribtion_code'],\
                                            Constants['il_last_step_vc_IDs_subscribtion_code']
                                            ])


    def get_state(self,source_edge,source_node,sink_node):
        action_mask,action_mask_index=self.get_edge_action_mask(source_edge,source_node)
        dest_embed=self.agent_id_embedding_dic[sink_node]
        source_node_state=self.get_agent_state(source_node)
        embeding=torch.cat((dest_embed,source_node_state),0)

        if Constants['Analysis_Mode']:
            try:
                assert(len(embeding)==self.get_state_diminsion(source_node))
            except Exception as e:
                breakpoint()
            
        return {"agent_id": source_node,
                "action_mask": action_mask,
                "action_mask_index":action_mask_index,
                "embeding": embeding}


    def get_agent_state(self,agent_id):
        if self.config.routing_mode=='Q_routing_0_hop':
            return self.agents_state[self.agent_index_dic[agent_id]]
        if  self.config.routing_mode=='Q_routing_1_hop' or\
            self.config.routing_mode=='Q_routing_2_hop':
            return self.aggregated_agents_state[self.agent_index_dic[agent_id]]

        return [] 

    def set_network_state(self):
        # the network state changes randomly. However, the random changes are the same among the benchmarks.
        self.agents_state=self.create_agents_state()
        for agent_id in self.agent_path_dic:
            out_edeges_list=list(self.agent_path_dic[agent_id])
            for edge_number in range(len(out_edeges_list)):
                path_key=out_edeges_list[edge_number]
                if self.seeded_random_generator.random()>self.config.congestion_epsilon:
                    # no congestion for the edge
                    self.agents_state[self.agent_index_dic[agent_id]][edge_number]=0
                    for edge in self.agent_path_dic[agent_id][path_key]:
                        traci.edge.setMaxSpeed(edge,self.network.edge_speed_dic[edge]['speed'])
                        self.network.edge_speed_dic[edge]['is_congested']=False
                else:
                    #congestion 
                    self.agents_state[self.agent_index_dic[agent_id]][edge_number]=1
                    for edge in self.agent_path_dic[agent_id][path_key]:
                        traci.edge.setMaxSpeed(edge,self.network.edge_speed_dic[edge]['speed']*self.config.congestion_speed_factor)
                        self.network.edge_speed_dic[edge]['is_congested']=True
        if  self.config.routing_mode=='Q_routing_1_hop' or\
            self.config.routing_mode=='Q_routing_2_hop':
            self.aggregated_agents_state=self.graph_attention_network(self.edge_index,self.agents_state)


    def get_state_diminsion(self,node_id): 
        if  self.config.routing_mode=='Q_routing_0_hop' or\
            self.config.routing_mode=='Q_routing_1_hop' or\
            self.config.routing_mode=='Q_routing_2_hop':
            return self.agnet_id_embedding_size+self.max_len_out_edges

        return self.agnet_id_embedding_size
    


    def state2torch(self,state):
        state=torch.tensor(state, device=self.config.device, dtype=torch.float)
        return state.unsqueeze(0)
# ---------------------------------------------------------------------------  
    def generate_uniform_demand(self,sim_time):
        if self.config.training_mode:
            trip=self.create_sample_uniform_trip()
        else:
            trip=self.config.uniform_demands[self.config.next_uniform_demand_index]

        self.config.next_uniform_demand_index+=1
        self.config.num_uniform_vc_dispatched+=1
        source_edge=trip[0]
        sink_edge=trip[1]
        destinatino_node=self.network.get_edge_head_node(sink_edge)

        new_vcs=[]
        trip_id="trip_{}".format(sim_time)
        traci.route.add(trip_id,[source_edge,sink_edge])
        deadline=4*self.get_shortest_path_time(self.network.get_edge_tail_node(source_edge),self.network.get_edge_head_node(sink_edge))\
                    +sim_time

        for i in range(0,self.config.demand_scale):
            vid="vehicle_{}_{}_{}".format(sim_time,i,destinatino_node)
            traci.vehicle.add(vid,trip_id)
            new_vcs.append(vid)
            # self.subscribe_vehicle(vid)
        # traci.vehicle.setColor("vehicle_{}".format(sim_time),(255,0,255))
        # traci.vehicle.setShapeClass("vehicle_{}".format(sim_time),"truck")
        return new_vcs,source_edge,self.network.get_edge_head_node(sink_edge),deadline
    
    def generate_biased_demand(self,sim_time,trip):                  
        source_edge=trip[0]
        sink_edge=trip[1]
        destinatino_node=self.network.get_edge_head_node(sink_edge)
        trip_id="biased_trip_{}".format(sim_time)
        self.config.num_biased_vc_dispatched+=1

        new_vcs=[]
        traci.route.add(trip_id,[source_edge,sink_edge])
        deadline=4*self.get_shortest_path_time(self.network.get_edge_tail_node(source_edge),self.network.get_edge_head_node(sink_edge))\
                    +sim_time

        for i in range(0,self.config.demand_scale):
            vid="biased_vehicle_{}_{}_{}".format(sim_time,i,destinatino_node)
            traci.vehicle.add(vid,trip_id)
            new_vcs.append(vid)
            # self.subscribe_vehicle(vid)
        return new_vcs,source_edge,self.network.get_edge_head_node(sink_edge),deadline
    
    def subscribe_vehicle(self,vc):
        traci.vehicle.subscribe(vc,[\
            Constants['vc_lane_ID_subscribtion_code'],
            Constants['vc_road_ID_subscribtion_code']
            ])
    def create_sample_uniform_trip(self):
        source_node=random.choice(self.agent_list)
        sink_node=random.choice(self.agent_list)
        while (sink_node==source_node):
            sink_node=random.choice(self.agent_list)

        source_edge=random.choice(self.network.get_out_edges(source_node))
        sink_edge=random.choice(self.network.get_in_edges(sink_node))
        return [source_edge,sink_edge]
# ------------------------------------------------------------------ 
    def create_agent_dic(self):
        """dictionary of all agents, 
        agent_dic[0]:#in edges
        agent_dic[1]:#out edges
        agent_dic[2]: state of the out-going edges, 1 if an edge is congested 0 O.W."""
        return {\
                node: [
                        len(self.network.get_in_edges(node)),
                        len(self.network.get_out_edges(node)),
                        ] \
                for node in self.network.graph.nodes() if \
                self.does_need_agent(node)
        }

    def does_need_agent(self,node):
        if node==None: 
            return False
        
        if len(self.network.get_out_edges(node))<2:
            return False
        
        for edge in self.network.get_in_edges(node):
            if len(self.network.get_edge_connections(edge))>1:
                return True

        return False
    
    def create_agent_id_embedding_dic(self):
        z_order_dic={}
        agent_embedding_dic={}
        for agent_id in self.agent_dic:
            position=traci.junction.getPosition(agent_id)
            unique_Z_ID=pm.interleave(int(position[0]),int(position[1]))
            if Constants['Analysis_Mode']:
                try:
                    assert(unique_Z_ID not in z_order_dic)
                except Exception as e:
                    breakpoint()

            z_order_dic[unique_Z_ID]=agent_id
        sorted_z_vals=list(z_order_dic)
        sorted_z_vals.sort()
        
        ID_size=len(format(len(sorted_z_vals)-1,'b'))
        for index in range(0,len(sorted_z_vals)):
            z_val=sorted_z_vals[index]
            agent_id=z_order_dic[z_val]
            agent_id_embedding=[0]*ID_size
            index_bin=format(index,'b')
            for i in range(len(index_bin)):
                agent_id_embedding[-i-1]=int(index_bin[-i-1])
            agent_embedding_dic[agent_id]=torch.tensor(agent_id_embedding,dtype=torch.float,device=self.config.device)

        return agent_embedding_dic,ID_size

    def create_agents_state(self):
        return  torch.vstack([torch.zeros(self.max_len_out_edges,device=self.config.device) for agent_id in self.agent_dic])


    def create_agent_path_dic(self):
        agent_paths={}
        for agent in self.agent_dic:
            agent_paths[agent]={}
            for out_edge in self.network.get_out_edges(agent):
                if Constants['Analysis_Mode']:
                    assert(out_edge not in agent_paths)
                agent_paths[agent][out_edge]=self.create_edge_path(out_edge)
        return agent_paths

    
    def create_edge_path(self,edgeID):
        """receives edgeID of the first edge returns edgeID of the path until there is only one connection"""
        path=[edgeID]
        path_head_connections=self.network.get_edge_connections(edgeID)

        while len(path_head_connections)==1:
            path.append(path_head_connections[0])
            path_head_connections=self.network.get_edge_connections(path[-1])


        return path
    
    def create_edge_action_mask_dic(self):
        edge_action_mask_dic={}
        for agent_id in self.agent_dic:
            for in_edge_id in self.network.get_in_edges(agent_id):
                if Constants['Analysis_Mode']:
                    assert(in_edge_id not in edge_action_mask_dic)
                edge_action_mask_dic[in_edge_id]=self.create_edge_action_mask(in_edge_id)

        return edge_action_mask_dic

    def create_edge_action_mask(self,edge_id):
        node_id=self.network.get_edge_head_node(edge_id)
        node_out_edges=self.network.get_out_edges(node_id)
        edge_connections=self.network.get_edge_connections(edge_id)
        action_mask=torch.tensor([-math.inf if edge not in  edge_connections else 0 for edge in node_out_edges],device=self.config.device)
        action_mask_index=[i for i in range(len(node_out_edges)) if node_out_edges[i] in edge_connections]
        return action_mask,action_mask_index

    def get_edge_path(self,node_id,edge_id):
        return self.agent_path_dic[node_id][edge_id]

    # def get_edge_path_head_node(self,edge):
    #     return self.network.get_edge_head_node(self.get_edge_path(edge)[-1])

    def get_next_road_IDs(self,node,action_edge_number):
        action_edge_ID=self.network.get_out_edges(node)[action_edge_number]
        return self.agent_path_dic[node][action_edge_ID]

    
    def get_destination(self,vc):
        route_tail=traci.vehicle.getRoute(vc)[-1]
        return self.network.get_edge_head_node(route_tail)

    def is_valid(self,source):
        self.log("validity check may be wrong",type='warn')
        return len(self.network.get_out_edges(source))!=0
   
    def get_time(self):
        return traci.simulation.getTime()

    def get_edge_weight(self,edge):
        return self.network.graph.get_edge_data(*edge)['weigh']

    def get_shortest_path_time(self,source,destination):
        return self.network.all_pairs_shortest_path[source][destination]
 


    def get_edge_action_mask(self,edge_id,node_id):
        if Constants['Analysis_Mode']:
            assert(node_id==self.network.get_edge_head_node(edge_id))
        return self.edge_action_mask_dic[edge_id]

# GAT--------------------------------------------------
    def create_agent_adjacency_list_dic(self):
        agent_adjacency_list_dic={}
        for agent_id in self.agent_path_dic:
            agent_adjacency_list_dic[agent_id]=[]
            for path in self.agent_path_dic[agent_id]:
                path_head=self.agent_path_dic[agent_id][path][-1]
                path_head_node=self.network.get_edge_head_node(path_head)
                agent_adjacency_list_dic[agent_id].append(path_head_node)
        return agent_adjacency_list_dic

    def create_edge_index(self,add_self_edges=True):
        num_of_nodes=len(self.agent_adjacency_list_dict)
        source_nodes_ids, target_nodes_ids = [], []
        seen_edges = set()

        for src_node, neighboring_nodes in self.agent_adjacency_list_dict.items():

            if Constants['Analysis_Mode']:
                try:
                    assert(src_node==list(self.agent_dic.keys())[self.agent_index_dic[src_node]])
                except Exception as e:
                    breakpoint()
            
            src_node=self.agent_index_dic[src_node]            
            source_nodes_ids.append(src_node)
            target_nodes_ids.append(src_node)
            seen_edges.add((src_node, src_node))

            for trg_node in neighboring_nodes:
                trg_node=self.agent_index_dic[trg_node]
                if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                    source_nodes_ids.append(src_node)
                    target_nodes_ids.append(trg_node)
                    seen_edges.add((src_node, trg_node))

        # shape = (2, E+V), 
        # where E is the number of edges in the graph
        # and V is the number of vertices in the graph
        edge_index = numpy.row_stack((source_nodes_ids, target_nodes_ids))

        return torch.tensor(edge_index,dtype=torch.long,device=self.config.device)

    def get_node_features(self):
        return self.agents_state

    def graph_attention_network(self,edge_index,node_features):
        return self.gat((node_features, edge_index))[0]

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
# deprecated-------------------------------------------------------------------------------------------------------------------
    def update_node_features(self):
        self.update_pressure_matrix()
        for row in range(0,self.dim):
            for column in range(0,self.dim):
                node_id=(column+row*3)
                try:
                    self.agent_state_dic[node_id][0]=(self.pressure_matrix[column][row])
                except Exception as e:
                    print(e)
                    breakpoint()

    def creat_vc_count_dic(self):
        lane_vc_count_dic=self.environment.eng.get_lane_vehicle_count()
        vc_count_dic={}
        for lane in lane_vc_count_dic:
            road= self.road2int(self.lane2road(lane))
            if not road in vc_count_dic: vc_count_dic[road]=0
        return vc_count_dic

    def update_vc_count_dic(self):
        lane_vc_count_dic=self.environment.eng.get_lane_vehicle_count()
        self.refresh_vc_count_dic()
        for lane in lane_vc_count_dic:
            road= self.road2int(self.lane2road(lane))
            # if road==10:
            #   breakpoint()
            self.vc_count_dic[road]+=lane_vc_count_dic[lane]


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
    def get_edge_index_among_node_out_edges(self,edge_id,node_id):
        return self.network.get_out_edges(node_id).index(edge_id)    

    def get_edge_index_among_node_in_edges(self,edge_id,node_id):
        return self.network.get_in_edges(node_id).index(edge_id)