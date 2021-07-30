import numpy

import torch
import json
import random
import traci
from inspect import currentframe, getframeinfo
import networkx as nx
import pymorton as pm


Constants = {
    "EXP":"3x3",
    "SUMO_PATH" : "/usr/share/sumo", #path to sumo in your system
    "SUMO_GUI_PATH" : "/usr/share/sumo/bin/sumo-gui", #path to sumo-gui bin in your system
    "SUMO_SHELL_PATH":"/usr/share/sumo/bin/sumo",
    "SUMO_CONFIG" : "./environments/sumo/networks/toronto/network.sumocfg", #path to your sumo config file
    "ROOT" : "./",
    "Network_XML" : "./environments/sumo/networks/toronto/toronto.net.xml",
    'LOG' : False,
    'WARNINGS': False,
    'WHERE':False,
    'Simulation_Delay' : '0'
    }

class Utils(object):
    """docstring for Utils"""
    def __init__(self,config,environment,network,Num_Flows,device,GAT,embed_network):
        super(Utils, self).__init__()
        self.config=config
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
        

        self.agent_dic=self.create_agent_dic()#{node:[len(in edges), len(out edges)]}
        self.agent_id_embedding_dic,self.agnet_id_embedding_size=self.create_agent_embedding_dic()
        self.agent_list=list(self.agent_dic)
        self.agent_index={self.agent_list[idx]:idx for idx in range(len(self.agent_list))}
        
        self.path_dic=self.create_path_dic()

        self.induction_loops=[il for il in traci.inductionloop.getIDList() if "TLS" not in il]

        self.network_state=[]
        self.dynamic_paths=self.path_dic #all edges are prone to congestion

        self.network_embed_dim=0
        # sum([len(self.dynamic_edges[edge]) for edge in self.dynamic_edges])
        
    def get_state(self,source_edge,source_node,sink_node):
        source_embed=[0]*self.agent_dic[source_node][0]
        source_embed[self.get_edge_index_among_node_in_edges(source_edge,source_node)]=1
        dest_embed=self.agent_id_embedding_dic[sink_node]
        embeding=source_embed+dest_embed+self.get_network_state_embeding()
        try:
            assert(len(embeding)==self.get_state_diminsion(source_node))
        except Exception as e:
            breakpoint()
            
        return {"agent_id": source_node,
                "embeding": embeding}

    def get_network_state_embeding(self):
        return []
        # try:
        #     assert(self.network_state!=[])
        # except Exception as e:
        #     self.log('network_state is null',type='err')
        #     # breakpoint()
        
        return self.network_state
            # return[1]

    def set_network_state(self):
        self.network_state=[]
        for path_key in self.dynamic_paths:
            if random.random()>self.config.congestion_epsilon:
                # no congestion for the edge
                for edge in self.dynamic_paths[path_key]:
                    traci.edge.setMaxSpeed(edge,self.network.edge_speed_dic[edge])

            else:
                #congestion 
                for edge in self.dynamic_paths[path_key]:
                    traci.edge.setMaxSpeed(edge,self.network.edge_speed_dic[edge]*self.config.congestion_speed_factor)


    def get_state_diminsion(self,agent_id): 
        return self.agent_dic[agent_id][0]+self.agnet_id_embedding_size+self.network_embed_dim
    
    def get_flow_id(self,vehicle_id):
        splited=vehicle_id.split('_')
        assert(len(splited)==2)
        return int(splited[1])

    def state2torch(self,state):
        state=torch.tensor(state, device=self.device, dtype=torch.float)
        return state.unsqueeze(0)
# ---------------------------------------------------------------------------  
    def generate_uniform_demand(self,sim_time):
        trip=self.config.uniform_demands[self.config.next_uniform_demand_index]
        self.config.next_uniform_demand_index+=1

        source_edge=trip[0]
        sink_edge=trip[1]

        new_vcs=[]
        traci.route.add("trip_{}".format(sim_time),[source_edge,sink_edge])
        deadline=4*self.get_shortest_path_time(self.network.get_edge_tail_node(source_edge),self.network.get_edge_head_node(sink_edge))\
                    +sim_time

        for i in range(0,self.config.demand_scale):
            traci.vehicle.add("vehicle_{}_{}".format(sim_time,i),"trip_{}".format(sim_time))
            new_vcs.append("vehicle_{}_{}".format(sim_time,i))
        # traci.vehicle.setColor("vehicle_{}".format(sim_time),(255,0,255))
        # traci.vehicle.setShapeClass("vehicle_{}".format(sim_time),"truck")
        return new_vcs,source_edge,self.network.get_edge_head_node(sink_edge),deadline
    
    def generate_biased_demand(self,sim_time,trip):                  
        source_edge=trip[0]
        sink_edge=trip[1]

        new_vcs=[]
        traci.route.add("biased_trip_{}".format(sim_time),[source_edge,sink_edge])
        deadline=4*self.get_shortest_path_time(self.network.get_edge_tail_node(source_edge),self.network.get_edge_head_node(sink_edge))\
                    +sim_time

        for i in range(0,self.config.demand_scale):
            traci.vehicle.add("biased_vehicle_{}_{}".format(sim_time,i),"biased_trip_{}".format(sim_time))
            new_vcs.append("biased_vehicle_{}_{}".format(sim_time,i))
            # traci.vehicle.setColor("vehicle_{}".format(sim_time),(255,0,255))
        # traci.vehicle.setShapeClass("vehicle_{}".format(sim_time),"truck")
        return new_vcs,source_edge,self.network.get_edge_head_node(sink_edge),deadline
# ------------------------------------------------------------------ 
    def create_agent_dic(self):
        """dictionary of all agents, 
        agent_dic[0]:#in edges
        agent_dic[1]:#out edges"""
        return {\
                node: [
                        len(self.network.get_in_edges(node)),
                        len(self.network.get_out_edges(node)),
                        ] \
                for node in self.network.graph.nodes() if \
                self.does_need_agent(node)
        }

    def create_agent_embedding_dic(self):
        z_order_dic={}
        agent_embedding_dic={}
        for agent_id in self.agent_dic:
            position=traci.junction.getPosition(agent_id)
            unique_Z_ID=pm.interleave(int(position[0]),int(position[1]))
            
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
            agent_embedding_dic[agent_id]=agent_id_embedding

        return agent_embedding_dic,ID_size



    def does_need_agent(self,node):
        if node==None: 
            return False
        
        if len(self.network.get_out_edges(node))<2:
            return False
        
        for edge in self.network.get_in_edges(node):
            if len(self.network.get_edge_connections(edge))>1:
                return True

        return False

    def create_path_dic(self):
        paths={}
        for agent in self.agent_dic:
            for out_edge in self.network.get_out_edges(agent):
                assert(out_edge not in paths)
                paths[out_edge]=self.create_edge_path(out_edge)
        return paths

    
    def create_edge_path(self,edgeID):
        """receives edgeID of the first edge returns edgeID of the path until there is only one connection"""
        path=[edgeID]
        path_head_connections=self.network.get_edge_connections(edgeID)

        while len(path_head_connections)==1:
            path.append(path_head_connections[0])
            path_head_connections=self.network.get_edge_connections(path[-1])


        return path

    def get_edge_path(self,edge_id):
        return self.path_dic[edge_id]

    def get_edge_path_head_node(self,edge):
        return self.network.get_edge_head_node(self.get_edge_path(edge)[-1])

    def get_next_road_IDs(self,node,action_edge_index):
        action_edge_ID=self.network.get_out_edges(node)[action_edge_index]
        return self.get_edge_path(action_edge_ID)

    
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
 
    def get_induction_loop_edge(self,inductionloop):
        return inductionloop.split('_')[1]

    def get_lane_edge(self,lane_ID):
        return traci.lane.getEdgeID(lane_ID)

    def get_edge_index_among_node_out_edges(self,edge_id,node_id):
        return self.network.get_out_edges(node_id).index(edge_id)    

    def get_edge_index_among_node_in_edges(self,edge_id,node_id):
        return self.network.get_in_edges(node_id).index(edge_id)
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