from environments.sumo.Utils import Constants
from environments.sumo.Utils import Utils
from environments.sumo.model.network import RoadNetworkModel
import os, sys
import traci
import random
import math
import gym
from gym import spaces
from torch.utils.tensorboard import SummaryWriter


class AR_SUMO_Environment(gym.Env):

		environment_name = "Adaptive Routing SUMO"
		def __init__(self,config, device):
			#create the road network
			
			self.config=config
			self.init_traci()		
			self.network = RoadNetworkModel(Constants["ROOT"], Constants["Network_XML"])
			# breakpoint()
			self.utils=Utils(environment=self,network=self.network,Num_Flows=1,device=device,GAT=config.GAT,embed_network=False)
			# breakpoint()
			# traci.edge.adaptTraveltime('gneE6',4*traci.edge.getTraveltime('gneE6'))

			self.reloadCmd = []


			self.vehicles={} # holds a copy of the info of the engine vehicles. I use it to find the vehicles that change road. it may be redundent with SUMO (SUMO might have an API that keeps track of the vehicles that change road)
			self.routing_queries=[]
			
			# -----------------------------------------------------
			self.gat=config.GAT
			self.routing_queries_states=[]
			self.states=[]
			self.acts=[]
			self.next_states=[]
			self.rewds=[]
			self.dones=[]
			# self.Max_Sim_Time=Max_Sim_Time
			self.device=device
			self.lanes_dic={}

			# self.stochastic_actions_probability = 0
			# self.actions = set(range(3))
			self.id = "Adaptive Routing"
			self.action_space = spaces.Discrete(3)
			# self.state_size=self.utils.get_state_diminsion()
			self.seed()
			self.reward_threshold = -10000
			self.trials = 10
			# self.Log=Log
			# self.skip_routing=skip_routing
			# self.iteration=-1
			# self.random_trips=random_trips
			# self.embed_network=embed_network
			self.current_step=-1
			self.cummulative_tt=0
			self.cummulative_n_success_v=0
			self.cummulative_n_failed_v=0
			self.cummulative_n_invalid_actions=0
			self.should_log_data=True
			if self.should_log_data:
				self.summ_writer = SummaryWriter()
						
		def reset(self,episode,config):
			self.episode_number=episode
			if episode==config.num_episodes_to_run-1:
				Constants["LOG"]=True
				breakpoint()
			# self.vehicles={}
			# self.refresh_routing_queries()
			# self.refresh_exp()
			# for vc in traci.vehicle.getIDList():
			# 	traci.vehicle.remove(vc)

		def refresh_routing_queries(self):
			self.routing_queries=[]
			self.routing_queries_states=[]
		
		def refresh_exp(self):
			self.states=[]
			self.acts=[]
			self.next_states=[]
			self.rewds=[]
			self.dones=[]	
			
																		# traci.edge.getLastStepMeanSpeed('gneE6')
																		# traci.vehicle.getLastActionTime(vid)
																		# traci.vehicle.getBestLanes(self, vehID)
																		# traci.vehicle.changeTarget()
																		# traci.inductionloop.getLastStepVehicleIDs("id induction loop")

		def step(self,actions=[]):
			sim_time=traci.simulation.getTime()
			self.utils.log("sim_time:{} ".format(sim_time))
			self.change_traffic_condition(sim_time)
			# ------------------------------------------
			# actions=[self.get_random_action(trans_vehicle) for trans_vehicle in self.routing_queries]				
			try:
				assert(len(self.routing_queries)==len(actions))
			except Exception as e:
				breakpoint()

			self.set_route(self.routing_queries, actions)
			#-------------------------------------------- 
			traci.simulationStep()

				# self.utils.update_and_evolve_node_features()
			# -------------------------------------------

			self.refresh_exp()
			self.refresh_routing_queries()
			# -------------------------------------------
			# for vc in traci.vehicle.getIDList():
			# 	if sim_time>self.vehicles[vc]["dead_line"]:
			# 		self.fail_routing(vc,sim_time)
			routing_queries,exiting_queries=self.get_queries()

			for vc in routing_queries:
				# self.add_to_next_trans_for_routing(vc)
				road=traci.vehicle.getRoadID(vc)
				try:
					assert(road in self.network.edge_ID_dic)
				except:
					self.utils.log("{} has entered an internal edge. Too late for making a routing decision".format(vc))
					breakpoint()

				agent_id=self.utils.get_edge_path_head_node(road)	

				try:
					assert(agent_id in self.utils.agent_dic)
				except:
					self.utils.log("invalid agent ID: {}".format(agent_id))
					breakpoint()		
				next_state=self.utils.get_state(road,agent_id,self.vehicles[vc]["destination"])
				
				if not self.vehicles[vc]["is_new"]:
					try:
						assert(self.vehicles[vc]["action"]!=None)
						assert(self.vehicles[vc]["state"]!=None)
						if not self.vehicles[vc]["is_action_valid"]:
							assert(self.vehicles[vc]["substitute_action"]!=None)
					except Exception as e:
						self.utils.log(str(e),type='err')
						breakpoint()
					
					reward=self.get_reward(self.vehicles[vc]["time"],sim_time)

					# try:
					# 	if not self.vehicles[vc]["is_action_valid"]:
					# 		assert(self.vehicles[vc]["substitute_action"]<self.utils.agent_dic[agent_id][1])
					# 	assert(self.vehicles[vc]["action"]<self.utils.agent_dic[agent_id][1])
					# except Exception as e:
					# 	breakpoint()

					if self.vehicles[vc]["is_action_valid"]:
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],reward,next_state,done=False)
					else:
						try:
							assert(self.vehicles[vc]["action"]<self.utils.agent_dic[self.vehicles[vc]["state"]['agent_id']][1])
						except Exception as e:
							breakpoint()
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],100*reward,next_state,done=False)
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["substitute_action"],reward,next_state,done=False)


				try:
					assert(next_state!=None)
				except Exception as e:
					breakpoint()
				
				self.update_env_vc_info(vc, sim_time,road,agent_id,next_state)
				self.routing_queries.append(vc)
				self.routing_queries_states.append(next_state)

			for vc in exiting_queries:
				assert(not self.vehicles[vc]["is_new"])
				reward=self.get_reward(self.vehicles[vc]["time"],sim_time)

				if self.vehicles[vc]["is_action_valid"]:
					self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],reward,next_state=None,done=True)
				else:
					self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],10*reward,next_state=None,done=True)
					self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["substitute_action"],reward,next_state=None,done=True)

				self.success_routing(vc,sim_time)
			# ----------------------------------------
			return self.close(sim_time)


		def set_route(self,VCs,ACTs):
			
			for vc, ac in zip(VCs, ACTs):
				assert(ac!=None)
				self.vehicles[vc]["action"]=ac
				self.vehicles[vc]["substitute_action"]=None
				self.vehicles[vc]["is_action_valid"]=True
				agent_id=self.vehicles[vc]["agent_id"]

				try:
					assert(self.vehicles[vc]["action"]<self.utils.agent_dic[agent_id][1])
				except Exception as e:
					breakpoint()

				next_roads = self.utils.get_next_road_IDs(agent_id,ac)
				current_road = self.vehicles[vc]["road"]
				assert(traci.vehicle.getRoadID(vc)==current_road)
				current_lane=traci.vehicle.getLaneID(vc)
				self.utils.log("agnet {} generated routing response {} for {}".format(agent_id,[current_road]+ next_roads,vc))

				if self.config.routing_mode=="TTSP":
					self.utils.log("Routing mode TTSP. Action discarded.")
					continue

				if self.config.routing_mode=="TTSPWRR":
					self.utils.log("Routing mode TTSPWRR. Action discarded.")
					traci.vehicle.rerouteTraveltime(vc)
					continue

				try:
					traci.vehicle.setRoute(vc, [current_road]+ next_roads)
					self.utils.log("there may be other reasons for failure of setRoute!",type='warn')
					self.utils.log('Sucess! {0} route changed. current_road:{1}, next_road:{2})'.format(vc, current_road, next_roads))

				except Exception as e:
					self.utils.log('Failed! Action {} is not valid @ road {}'.format(ac,current_road))
					self.set_substitue_action(vc,current_road,current_lane,agent_id)


		


		def add_env_vc(self,vc,road,time,destination,dead_line):
			self.utils.log(vc+" entered simulation @ road: {}".format(road))
			self.vehicles[vc]={
				"destination": destination,
				"start_time":time,
				"dead_line":dead_line,
				"time": None,
				"road":None,
				"agent_id":None,
				"action":None,
				"is_action_valid":None,
				"substitute_action":None,
				"reward":None,
				"state":None,
				"is_new":True
				}

		def update_env_vc_info(self,vc,time,road,agent_id,state):	
			self.utils.log(" {} sent RQ to agent {}".format(vc,agent_id))
			self.vehicles[vc]["time"]=time
			self.vehicles[vc]["road"]=road
			self.vehicles[vc]["agent_id"]=agent_id
			self.vehicles[vc]["state"]=state
			self.vehicles[vc]["is_new"]=False

		def get_reward(self,last_time,current_time):
			return last_time-current_time
		
		def save_expirence(self,state,action,reward,next_state,done):

			self.states.append(state)
			self.acts.append(action)
			self.rewds.append(reward)
			self.next_states.append(next_state)
			self.dones.append(done)
			self.utils.log("S:{},A:{},R:{},NS:{},D:{}".format(state,action,reward,next_state,done))

		def success_routing(self, vc,sim_time):
			self.utils.log("{} exited.".format(vc))
			self.cummulative_tt+=sim_time-self.vehicles[vc]["start_time"]
			self.cummulative_n_success_v+=1
			self.exit(vc)

		def fail_routing(self,vc,time):
			self.utils.log("Routing for {} failed @ time {}".format(vc,time))
			self.cummulative_n_failed_v+=1
			self.exit(vc)

		def exit(self, vc):
			self.vehicles.pop(vc)
			traci.vehicle.remove(vc)			

		def set_substitue_action(self,vc,current_road,current_lane,agent_id):
			self.vehicles[vc]["is_action_valid"]=False
			self.vehicles[vc]["substitute_action"]=self.get_subtitue_action(current_lane,agent_id)
			try:
				assert(self.vehicles[vc]["substitute_action"]<self.utils.agent_dic[agent_id][1])
			except Exception as e:
				breakpoint()
			next_roads = self.utils.get_next_road_IDs(agent_id,self.vehicles[vc]["substitute_action"])
			self.cummulative_n_invalid_actions+=1
			try:
				traci.vehicle.setRoute(vc, [current_road]+ next_roads)	
				self.utils.log("agnet {} generated substitute routing response {} for {}".format(agent_id,[current_road]+ next_roads,vc))
			except Exception as e:
				self.utils.log("error in setting the substitute route",type='err')
				breakpoint()

		def get_subtitue_action(self, lane_ID,agent_id):
			"""
			return a random number between 1 and len(self.utils.get_out_edges(self.g)t_intersec_id(vc_ID)])
			"""
			self.utils.log("I am assuming that each edge has only one lane",type='warn')
			subs_edge=random.choice(traci.lane.getLinks(lane_ID))[0].split('_')[0]
			subs_act=self.utils.get_edge_index_among_node_out_edges(subs_edge,agent_id)
			
			return subs_act
		
		def change_traffic_condition(self,time):
			if time%self.config.traffic_period==1:
				self.utils.set_network_state(epsilon=1)

			if time%self.config.vc_period==1 and traci.vehicle.getIDCount()<self.config.Max_number_vc:
				vid,road,destination,dead_line=self.utils.generate_random_trip(time)
				self.add_env_vc(vid,road,time,destination,dead_line)
		
		
		def log_data(self):
			if self.cummulative_n_success_v!=0:
				self.summ_writer.add_scalar("AVTT:",self.cummulative_tt/self.cummulative_n_success_v,self.episode_number)
			# self.summ_writer.add_scalar("Routing Succcess:",self.cummulative_n_success_v,self.episode_number)
			# self.summ_writer.add_scalar("Routing failure:",self.cummulative_n_failed_v,self.episode_number)
			if self.config.routing_mode=="Q_routing":
				self.summ_writer.add_scalar("Invalid Action:",self.cummulative_n_invalid_actions,self.episode_number)
			
			self.cummulative_tt=0
			self.cummulative_n_success_v=0
			self.cummulative_n_failed_v=0
			self.cummulative_n_invalid_actions=0

		def close(self,sim_time):
			if self.is_terminal(sim_time):
				# self.summ_writer.close()
				if self.log_data:
					self.log_data()
				return self.states,self.acts,self.next_states,self.rewds,self.dones,True
			else:
				return self.states,self.acts,self.next_states,self.rewds,self.dones,False
		
		def is_terminal(self,sim_time):
			"""
			check for some terminal condition. e.g. all vehicles exited the simulation or the time limit has passed

			"""
			# traci.simulation.getMinExpectedNumber() == 0 or
			return sim_time%self.config.episode_period==self.config.episode_period-1

		def init_traci(self):
			sys.path.append(os.path.join(Constants['SUMO_PATH'], os.sep, 'tools'))
			sumoBinary = Constants["SUMO_GUI_PATH"]
			self.sumoCmd = [sumoBinary, '-S', '-d', Constants['Simulation_Delay'], "-c", Constants["SUMO_CONFIG"]]
			traci.start(self.sumoCmd)
		
		def close_traci(self):
			traci.close()


		def get_queries(self):
			routing_queries=[]
			exiting_queries=[]

			for il in self.utils.induction_loops:
				for vc in traci.inductionloop.getLastStepVehicleIDs(il):
					if self.has_exited(vc) or self.has_transited(vc):
						continue
					if self.has_arrived(il,vc):
						exiting_queries.append(vc)
					else:
						routing_queries.append(vc)
			return routing_queries,exiting_queries
		
		def has_exited(self,vc):
			return vc not in self.vehicles

		def has_transited(self, vc):
			if self.vehicles[vc]["is_new"]:
				return False
			return self.vehicles[vc]["road"]==traci.vehicle.getRoadID(vc)

		def has_arrived(self,il,vc):
			il_edge=self.utils.get_induction_loop_edge(il)
			return self.network.get_edge_head_node(il_edge)\
					==self.vehicles[vc]['destination']

		def get_routing_queries(self):
			return self.routing_queries_states