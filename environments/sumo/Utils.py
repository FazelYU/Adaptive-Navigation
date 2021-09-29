import numpy

import torch
import json
import random
import traci
from inspect import currentframe, getframeinfo
import networkx as nx
import pymorton as pm
import pandas as pd
from sklearn.manifold import TSNE
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.decomposition import PCA
import igraph as ig
from scipy.stats import entropy


NETWORK="5x6"
Constants = {
    "NETWORK":NETWORK,
    "SUMO_PATH" : "/usr/share/sumo", #path to sumo in your system
    "SUMO_GUI_PATH" : "/usr/share/sumo/bin/sumo-gui", #path to sumo-gui bin in your system
    "SUMO_SHELL_PATH":"/usr/share/sumo/bin/sumo",
    "SUMO_CONFIG" : "./environments/sumo/networks/{}/network.sumocfg".format(NETWORK), #path to your sumo config file
    "ROOT" : "./",
    "Network_XML" : "./environments/sumo/networks/{}/{}.net.xml".format(NETWORK,NETWORK),
    'Additional_XML':"./environments/sumo/networks/{}/{}_additional.add.xml".format(NETWORK,NETWORK),
    'Analysis_Mode': True,
    'LOG' : False,
    'WARNINGS': False,
    'WHERE':False,
    'Simulation_Delay' : '0',

    'il_lane_ID_subscribtion_code': 0x51,
    'il_last_step_vc_IDs_subscribtion_code': 0x12,
    'il_last_step_vc_count_subscribtion_code': 0x10,

    'vc_road_ID_subscribtion_code': 0x50,
    'vc_lane_ID_subscribtion_code':0x51,


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
        # self.network_embd_dim=9
        self.environment=environment
        # self.vc_count_dic=self.creat_vc_count_dic()
        self.Num_Flow_Types=3
        self.slow_vc_speed=7
        self.fast_vc_speed=2*self.slow_vc_speed

        self.gat=GAT
        self.embed_network=embed_network
        

        self.agent_dic=self.create_agent_dic()#{node:[len(in edges), len(out edges)]}
        self.agent_list=list(self.agent_dic)
        
        self.agent_id_embedding_dic,self.agnet_id_embedding_size=self.create_agent_id_embedding_dic()
        self.agent_label_dic=self.create_agent_labels_dic()
        # self.tSNE_plot()

        self.agent_index_dic={self.agent_list[idx]:idx for idx in range(len(self.agent_list))}
        self.agent_path_dic=self.create_agent_path_dic()
        self.agent_adjacency_list_dict=self.create_agent_adjacency_list_dic()
        self.max_len_out_edges=max([self.agent_dic[agent_id][1] for agent_id in self.agent_dic])
        self.config.edge_index=self.create_edge_index()
        self.config.network_state=self.create_network_state()
        # if  self.config.routing_mode=='Q_routing_1_hop' or\
        #     self.config.routing_mode=='Q_routing_2_hop':
        #     self.aggregated_network_state=self.graph_attention_network(self.edge_index,self.config.network_state)

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
        # if dest_embed.device.type!='cuda':
        #     breakpoint()
        # source_node_state=self.get_agent_state(source_node)
        # embeding=torch.cat((dest_embed,source_node_state),0)

        # if Constants['Analysis_Mode']:
        #     try:
        #         assert(len(embeding)==self.get_state_diminsion(source_node))
        #     except Exception as e:
        #         breakpoint()
        return {
                "agent_id": source_node,
                "agent_idx": self.agent_index_dic[source_node],
                "action_mask": action_mask,
                "action_mask_index":action_mask_index,
                "embeding": torch.cat((dest_embed,self.config.network_state.detach().clone().view(-1)),-1)
                # "network_state":self.config.network_state.detach().clone()
                }

    def get_state_diminsion(self,node_id): 
        if  self.config.does_need_network_state:
            return self.agnet_id_embedding_size+self.max_len_out_edges

        return self.agnet_id_embedding_size

    def get_network_embed_size(self):
        if  self.config.does_need_network_state:
            return self.max_len_out_edges
        return 0

    def get_intersection_id_size(self):
        return self.agnet_id_embedding_size
    
    def create_network_state(self):
        return  torch.vstack([torch.zeros(self.max_len_out_edges,device=self.config.device) for agent_id in self.agent_dic]).detach()

    def set_network_state(self):
        # the network state changes randomly. However, the random changes are the same among the benchmarks.
        for agent_id in self.agent_path_dic:
            out_edeges_list=list(self.agent_path_dic[agent_id])
            for edge_number in range(len(out_edeges_list)):
                path_key=out_edeges_list[edge_number]
                if self.seeded_random_generator.random()>self.config.congestion_epsilon:
                    # no congestion for the edge
                    self.config.network_state[self.agent_index_dic[agent_id]][edge_number]=0
                    for edge in self.agent_path_dic[agent_id][path_key]:
                        traci.edge.setMaxSpeed(edge,self.network.edge_speed_dic[edge]['speed'])
                        self.network.edge_speed_dic[edge]['is_congested']=False
                else:
                    #congestion 
                    self.config.network_state[self.agent_index_dic[agent_id]][edge_number]=1
                    for edge in self.agent_path_dic[agent_id][path_key]:
                        traci.edge.setMaxSpeed(edge,self.network.edge_speed_dic[edge]['speed']*self.config.congestion_speed_factor)
                        self.network.edge_speed_dic[edge]['is_congested']=True
        self.visualize_gat_properties(self.agent_list)
        # breakpoint()

    


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

    def create_agent_labels_dic(self):
        agent_label_dic={}
        positions=numpy.array([list(traci.junction.getPosition(agent_id)) for agent_id in self.agent_list])
        max_X,max_Y=positions.max(0)
        min_X,min_Y=positions.min(0)
        range_X=max_X-min_X
        range_Y=max_Y-min_Y
        for agent_id in self.agent_dic:
            x,y=traci.junction.getPosition(agent_id)
            # if x==min_X:
            #     breakpoint()
            # if y==min_Y:
            #     breakpoint()
            if x==max_X:
                # breakpoint()
                x-=0.001
            if y==max_Y:
                # breakpoint()
                y-=0.001

            i=math.floor((x-min_X)/range_X*4)
            j=math.floor((y-min_Y)/range_Y*4)
            label=j*4+i
            agent_label_dic[agent_id]=label

        y=numpy.array([agent_label_dic[agent_id] for agent_id in self.agent_dic])
        # breakpoint()
        return agent_label_dic



    def vis_intersec_id_embedding(self,agent_id,transform_func):
        X=torch.vstack([self.agent_id_embedding_dic[agent_id] for agent_id in self.agent_dic])
        X_trns=transform_func(agent_id,X)
        
        X=X.detach().cpu().numpy()
        X_trns=X_trns.detach().cpu().numpy()
        
        y=numpy.array([self.agent_label_dic[agent_id] for agent_id in self.agent_dic])
        
        # self.tSNE_plot(X,y)
        # self.tSNE_plot(X_trns,y)
        self.pca_plot(X,y)
        self.pca_plot(X_trns,y)
        plt.show()
        breakpoint()

    # def tSNE_plot(self,X,y):
    #     df=pd.DataFrame(X)
    #     df['label-class']=y
    #     df['label']=[int(lbl) for lbl in y]
    #     breakpoint()
    #     df.groupby('label-class', as_index=False).size().plot(kind='bar',x='label')
    #     breakpoint()
    #     tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=400)
    #     tsne_results = tsne.fit_transform(df)
    #     df['tsne-2d-one'] = tsne_results[:,0]
    #     df['tsne-2d-two'] = tsne_results[:,1]
    #     plt.figure(figsize=(16,10))
    #     sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",hue="label",size="label",data=df,legend="full")
    #     # alpha=0.3
    
    def pca_plot(self,X,y):
        df=pd.DataFrame(X)
        df['label']=y
        # df.groupby('label', as_index=False).size().plot(kind='bar',x='label')
        # breakpoint()

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df)
        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1] 
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        # tsne = TSNE(n_components=2, verbose=1, perplexity=4, n_iter=400)
        # tsne_results = tsne.fit_transform(df)
        # df['tsne-2d-one'] = tsne_results[:,0]
        # df['tsne-2d-two'] = tsne_results[:,1]
        plt.figure()
        sns.set(font_scale=2)
        sns.scatterplot(x="pca-one", y="pca-two",hue="label",style="label",data=df,legend="full",s=400)
  

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
        return self.config.network_state

    # def graph_attention_network(self,edge_index,node_features):
    #     return self.gat((node_features, edge_index))[0]

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


    def visualize_gat_properties(self,nodes_of_interest,visualization_type="ATTENTION"):
        gat=self.config.GAT
        node_features=self.config.network_state
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

        # config = {
        #     'dataset_name': dataset_name,
        #     'layer_type': LayerType.IMP3,
        #     'should_visualize': False,  # don't visualize the dataset
        #     'batch_size': 2,  # used only for PPI
        #     'ppi_load_test_only': True  # used only for PPI (optimization, we're loading only test graphs)
        # }

        # Step 1: Prepare the data
        gat.eval()  # some layers like nn.Dropout behave differently in train vs eval mode so this part is important
        with torch.no_grad():
            # Step 3: Run predictions and collect the high dimensional data
            all_nodes_unnormalized_scores = gat(node_features.view(1,-1,self.config.network_state_size)).view(-1,self.config.network_embed_size) # shape = (N, num of classes)
            all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()
        # We'll need the edge index in different for multiple visualization types
        edge_index = self.config.edge_index

        
        # Step 4: Perform a specific visualization
        if visualization_type == "ATTENTION":
            # The number of nodes for which we want to visualize their attention over neighboring nodes
            # (2x this actually as we add nodes with highest degree + random nodes)
            # num_nodes_of_interest = 4  # 4 is an arbitrary number you can play with these numbers
            head_to_visualize = 0  # plot attention from this multi-head attention's head
            gat_layer_id = 0  # plot attention from this GAT layer

            # Build up the complete graph
            # node_features shape = (N, FIN), where N is the number of nodes and FIN number of inumpyut features
            total_num_of_nodes = len(node_features)
            complete_graph = ig.Graph()
            complete_graph.add_vertices(total_num_of_nodes)  # igraph creates nodes with ids [0, total_num_of_nodes - 1]
            edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))  # igraph requires this format
            complete_graph.add_edges(edge_index_tuples)

            # Pick the target nodes to plot (nodes with highest degree + random nodes)
            # Note: there could be an overlap between random nodes and nodes with highest degree - but highly unlikely
            # nodes_of_interest_idx = numpy.argpartition(complete_graph.degree(), -num_nodes_of_interest)[-num_nodes_of_interest:]
            # random_node_ids = numpy.random.randint(low=0, high=total_num_of_nodes, size=num_nodes_of_interest)
            # nodes_of_interest_idx = numpy.append(nodes_of_interest_idx, random_node_ids)
            # numpy.random.shuffle(nodes_of_interest_idx)
            # breakpoint()
            nodes_of_interest_idx=[self.agent_index_dic[agent_id] for agent_id in nodes_of_interest]

            target_node_ids = edge_index[1]
            source_nodes = edge_index[0]
            for target_node_id in nodes_of_interest_idx:
                # Step 1: Find the neighboring nodes to the target node
                # Note: self edges are included so the target node is it's own neighbor (Alexandro yo soy tu madre)
                src_nodes_indices = torch.eq(target_node_ids, target_node_id)
                source_node_ids = source_nodes[src_nodes_indices].cpu().numpy()
                size_of_neighborhood = len(source_node_ids)

                # Step 2: Fetch their labels
                # labels = node_labels[source_node_ids].cpu().numpy()

                # Step 3: Fetch the attention weights for edges (attention is logged during GAT's forward pass above)
                # attention shape = (N, NH, 1) -> (N, NH) - we just squeeze the last dim it's superfluous
                # breakpoint()
                all_attention_weights = gat.gat_net[gat_layer_id].attention_weights.squeeze(dim=-1).squeeze(0)
                attention_weights = all_attention_weights[src_nodes_indices, head_to_visualize].cpu().numpy()
                # This part shows that for CORA what GAT learns is pretty much constant attention weights! Like in GCN!
                # On the other hand PPI's attention pattern is much less uniform.
                print(f'Max attention weight = {numpy.max(attention_weights)} and min = {numpy.min(attention_weights)}')
                attention_weights /= numpy.max(attention_weights)  # rescale the biggest weight to 1 for nicer plotting

                # Build up the neighborhood graph whose attention we want to visualize
                # igraph constraint - it works with contiguous range of ids so we map e.g. node 497 to 0, 12 to 1, etc.
                id_to_igraph_id = dict(zip(source_node_ids, range(len(source_node_ids))))
                ig_graph = ig.Graph()
                ig_graph.add_vertices(size_of_neighborhood)
                ig_graph.add_edges([(id_to_igraph_id[neighbor], id_to_igraph_id[target_node_id]) for neighbor in source_node_ids])

                # Prepare the visualization settings dictionary and plot
                # breakpoint()
                visual_style = {
                    "vertex_size":30,
                    "vertex_label_size":25,
                    "edge_width": 5*attention_weights,  # make edges as thick as the corresponding attention weight
                    "layout": ig_graph.layout_reingold_tilford_circular(),  # layout for tree-like graphs
                    "margin":100,
                    "vertex_label_dist":1,
                    # "layout": ig_graph.layout(layout='layout_grid').scale(5),
                    # "layout": ig_graph.layout_grid(),
                    "vertex_label": [self.agent_list[idx] for idx in source_node_ids]
                }
                # This is the only part that's Cora specific as Cora has 7 labels
                # if dataset_name.lower() == DatasetType.CORA.name.lower():
                #     visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in labels]
                # else:
                #     print('Add custom color scheme for your specific dataset. Using igraph default coloring.')


                ig.plot(ig_graph, **visual_style)
                # fig,ax=plt.subplots()
                # # ig.plot(ig_graph,
                # #     layout=ig_graph.layout_reingold_tilford_circular(),
                # #     vertex_size=40,
                # #     vertex_label=[self.agent_list[idx] for idx in source_node_ids],
                # #     edge_width=5*attention_weights,
                # #     # target=ax
                # #     )
                # plt.show()


            # breakpoint()
        # We want our local probability distributions (attention weights over the neighborhoods) to be
        # non-uniform because that means that GAT is learning a useful pattern. Entropy histograms help us visualize
        # how different those neighborhood distributions are from the uniform distribution (constant attention).
        # If the GAT is learning const attention we could well be using GCN or some even simpler models.
        elif visualization_type == "ENTROPY":
            num_heads_per_layer = [layer.num_of_heads for layer in gat.gat_net]
            num_layers = len(num_heads_per_layer)
            num_of_nodes = len(node_features)

            target_node_ids = edge_index[1].cpu().numpy()

            # For every GAT layer and for every GAT attention head plot the entropy histogram
            for layer_id in range(num_layers):
                # Fetch the attention weights for edges (attention is logged during GAT's forward pass above)
                # attention shape = (N, NH, 1) -> (N, NH) - we just squeeze the last dim it's superfluous
                all_attention_weights = gat.gat_net[layer_id].attention_weights.squeeze(dim=-1).squeeze(0).cpu().numpy()

                # tmp fix for PPI there are some numerical problems and so most of attention coefficients are 0
                # and thus we can't plot entropy histograms
                # if dataset_name == DatasetType.PPI.name and layer_id > 0:
                #     print(f'Entropy histograms for {dataset_name} are available only for the first layer.')
                #     break

                for head_id in range(num_heads_per_layer[layer_id]):
                    uniform_dist_entropy_list = []  # save the ideal uniform histogram as the reference
                    neighborhood_entropy_list = []

                    # This can also be done much more efficiently via scatter_add_ (no for loops)
                    # pseudo: out.scatter_add_(node_dim, -all_attention_weights * log(all_attention_weights), target_index)
                    for target_node_id in range(num_of_nodes):  # find every the neighborhood for every node in the graph
                        # These attention weights sum up to 1 by GAT design so we can treat it as a probability distribution
                        neigborhood_attention = all_attention_weights[target_node_ids == target_node_id].flatten()
                        # Reference uniform distribution of the same length
                        ideal_uniform_attention = numpy.ones(len(neigborhood_attention))/len(neigborhood_attention)

                        # Calculate the entropy, check out this video if you're not familiar with the concept:
                        # https://www.youtube.com/watch?v=ErfnhcEV1O8 (Aurélien Géron)
                        neighborhood_entropy_list.append(entropy(neigborhood_attention, base=2))
                        uniform_dist_entropy_list.append(entropy(ideal_uniform_attention, base=2))

                    title = f'{self.config.network_name} entropy histogram layer={layer_id}, attention head={head_id}'
                    draw_entropy_histogram(uniform_dist_entropy_list, title, color='orange', uniform_distribution=True)
                    draw_entropy_histogram(neighborhood_entropy_list, title, color='dodgerblue')

                    fig = plt.gcf()  # get current figure
                    plt.show()
                    # fig.savefig(os.path.join(DATA_DIR_PATH, f'layer_{layer_id}_head_{head_id}.jpg'))
                    plt.close()
        else:
            raise Exception(f'Visualization type {visualization_type} not supported.')



def draw_entropy_histogram(entropy_array, title, color='blue', uniform_distribution=False, num_bins=30):
    max_value = numpy.max(entropy_array)
    bar_width = (max_value / num_bins) * (1.0 if uniform_distribution else 0.75)
    histogram_values, histogram_bins = numpy.histogram(entropy_array, bins=num_bins, range=(0.0, max_value))

    plt.bar(histogram_bins[:num_bins], histogram_values[:num_bins], width=bar_width, color=color)
    plt.xlabel(f'entropy bins')
    plt.ylabel(f'# of node neighborhoods')
    plt.title(title)




















































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