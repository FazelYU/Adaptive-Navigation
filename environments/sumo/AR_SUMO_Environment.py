# from environments.sumo.Utils import self.config.Constants
from environments.sumo.Utils import Utils
from environments.sumo.model.network import RoadNetworkModel
import os, sys
import traci
import math
import gym
from gym import spaces
from torch.utils.tensorboard import SummaryWriter


class AR_SUMO_Environment(gym.Env):

		environment_name = "Adaptive Routing SUMO"
		def __init__(self,config):
			#create the road network
			
			self.config=config

			self.network = config.network
			# self.network = RoadNetworkModel(self.config.Constants["ROOT"], self.config.Constants["Network_XML"])
			# breakpoint()
			self.utils=config.utils
			# self.utils=Utils(config=config,environment=self,network=self.network)
			# breakpoint()
			# traci.edge.adaptTraveltime('gneE6',4*traci.edge.getTraveltime('gneE6'))

			self.reloadCmd = []


			self.vehicles={} # holds a copy of the info of the engine vehicles. I use it to find the vehicles that change road. it may be redundent with SUMO (SUMO might have an API that keeps track of the vehicles that change road)
			self.routing_queries=[]
			
			# -----------------------------------------------------
			# self.gat=config.GAT
			self.routing_queries_states=[]
			self.states=[]
			self.acts=[]
			self.next_states=[]
			self.rewds=[]
			self.dones=[]
			# self.Max_Sim_Time=Max_Sim_Time
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
				self.summ_writer = SummaryWriter(filename_suffix=self.config.routing_mode)
						
		def reset(self,episode,config):
			self.episode_number=episode
			self.cummulative_tt=0
			self.cummulative_n_success_v=0
			self.cummulative_n_failed_v=0
			self.cummulative_n_invalid_actions=0
			if not self.config.training_mode:
				self.config.next_uniform_demand_index=0
				self.config.num_biased_vc_dispatched=0
				self.config.num_uniform_vc_dispatched=0
			# if episode==config.num_episodes_to_run-1:
			# 	self.config.Constants["LOG"]=True
			# 	breakpoint()
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
			self.set_route(self.routing_queries, actions)
			#-------------------------------------------- 
			traci.simulationStep()

			try:
				TelNum=traci.simulation.getStartingTeleportNumber()
				if self.config.Constants['Analysis_Mode']: assert(TelNum==0)
			except Exception as e:
				self.utils.log("{} Telepoting Vehicle Found!".format(TelNum))
				# breakpoint()

				# self.utils.update_and_evolve_node_features()
			# -------------------------------------------

			self.refresh_exp()
			self.refresh_routing_queries()

# ------------------------------------------------------------
			# for vc in traci.vehicle.getIDList():
			# 	if sim_time>self.vehicles[vc]["dead_line"]:
			# 		self.fail_routing(vc,sim_time)
#-------------------------------------------------------------
			routing_queries,exiting_queries=self.get_queries()

			for (vc,road) in routing_queries:
				if self.config.Constants['Analysis_Mode']:
					try:					
						assert(road in self.network.edge_ID_dic)
					except:
						self.utils.log("{} has passed the detector without a decision. Too late for making a routing decision".format(vc))
						breakpoint()

				next_agent_id=self.network.get_edge_head_node(road)	
				if self.config.Constants['Analysis_Mode']:
					try:
						assert(next_agent_id in self.utils.agent_dic)
					except:
						self.utils.log("invalid agent ID: {}".format(next_agent_id))
						breakpoint()
						self.exit(vc)
						continue		
				
				next_state=self.utils.get_state(road,next_agent_id,self.vehicles[vc]["destination"])
				
				if not self.vehicles[vc]["is_new"]:
					if self.config.Constants['Analysis_Mode']:
						try:
							assert(self.vehicles[vc]["action"]!=None)
							assert(self.vehicles[vc]["road"]!=road)
						except Exception as e:
							self.utils.log("vehicle already transited!",type='err')
							breakpoint()
						
						try:
							assert(self.vehicles[vc]["state"]!=None)
						except Exception as e:
							breakpoint()

					reward=self.get_reward(self.vehicles[vc]["time"],sim_time)


					if self.vehicles[vc]["is_action_valid"]:
						if self.config.Constants['Analysis_Mode']:
							try:
								assert(self.vehicles[vc]["action"]<self.utils.agent_dic[self.vehicles[vc]["state"]['agent_id']][1])
								# breakpoint()
							except Exception as e:
								breakpoint()

						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],reward,next_state,done=False)
					else:
						breakpoint()
						if self.config.Constants['Analysis_Mode']:
							try:
								assert(self.vehicles[vc]["substitute_action"]<self.utils.agent_dic[self.vehicles[vc]["state"]['agent_id']][1])
							except Exception as e:
								breakpoint()

						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],100*reward,next_state,done=False)
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["substitute_action"],reward,next_state,done=False)

				if self.config.Constants['Analysis_Mode']:
					try:
						assert(next_state!=None)
					except Exception as e:
						breakpoint()
				
				self.update_env_vc_info(vc, sim_time,road,next_agent_id,next_state)
				self.routing_queries.append(vc)
				self.routing_queries_states.append(next_state)

			for (vc,road) in exiting_queries:
				if not self.vehicles[vc]["is_new"]:
					reward=self.get_reward(self.vehicles[vc]["time"],sim_time)

					if self.vehicles[vc]["is_action_valid"]:
						if self.config.Constants['Analysis_Mode']:
							try:
								assert(self.vehicles[vc]["action"]<self.utils.agent_dic[self.vehicles[vc]["state"]['agent_id']][1])
							except Exception as e:
								breakpoint()
		
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],reward,next_state=None,done=True)
					else:
						if self.config.Constants['Analysis_Mode']:
							try:
								assert(self.vehicles[vc]["substitute_action"]<self.utils.agent_dic[self.vehicles[vc]["state"]['agent_id']][1])
							except Exception as e:
								breakpoint()
						
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],10*reward,next_state=None,done=True)
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["substitute_action"],reward,next_state=None,done=True)

				self.success_routing(vc,sim_time)


			return self.close(sim_time)


		def set_route(self,VCs,ACTs):
			
			for vc, ac in zip(VCs, ACTs):
				self.vehicles[vc]["action"]=ac
				self.vehicles[vc]["substitute_action"]=None
				self.vehicles[vc]["is_action_valid"]=True
				agent_id=self.vehicles[vc]["agent_id"]

				if self.config.Constants['Analysis_Mode']:
					try:
						ac=self.vehicles[vc]["action"]
						assert(ac!=None)
						assert(ac<self.utils.agent_dic[agent_id][1])
					except Exception as e:
						breakpoint()

				next_roads = self.utils.get_next_road_IDs(agent_id,ac)
				current_road = self.vehicles[vc]["road"]
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
					breakpoint()
					# self.set_substitue_action(vc,current_road,current_lane,agent_id)

		def is_action_valid(self,agent_id,action):
			return action<self.utils.agent_dic[agent_id][1]

		def is_action_valid(self,road,action):
			pass

		


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

		
		def change_traffic_condition(self,time):
			if time%self.config.traffic_period==0:
				self.utils.set_network_state()
			
			if self.should_insert_new_vc(time):
				if time%self.config.uniform_demand_period==1:
					vids,road,destination,dead_line=self.utils.generate_uniform_demand(time)
					for vid in vids:
						self.add_env_vc(vid,road,time,destination,dead_line)
				
				if not self.config.training_mode:	
					if self.config.num_uniform_vc_dispatched>0 and \
					self.config.num_biased_vc_dispatched/self.config.num_uniform_vc_dispatched < \
					self.config.biased_demand_2_uniform_demand_ratio:
						for trip in self.config.biased_demand:			
							vids,road,destination,dead_line=self.utils.generate_biased_demand(time,trip)
							for vid in vids:
								self.add_env_vc(vid,road,time,destination,dead_line)
			
		def should_insert_new_vc(self,time):
			if self.config.training_mode:
				#  we are in training mode
				return traci.vehicle.getIDCount()<self.config.Max_number_vc

			# we are testing. don't insert vehicles if reached the end of the test demand data.
			return 	traci.vehicle.getIDCount()<self.config.Max_number_vc and\
					self.config.next_uniform_demand_index<len(self.config.uniform_demands)

		
		def is_terminal(self,sim_time):
			if self.config.training_mode:
				return sim_time%self.config.max_num_sim_time_step_per_episode==self.config.max_num_sim_time_step_per_episode-1

			return 	self.config.next_uniform_demand_index==len(self.config.uniform_demands) \
					and traci.simulation.getMinExpectedNumber() == 0
			
		def log_data(self):
			if self.cummulative_n_success_v!=0:
				self.summ_writer.add_scalar("Routing Success:",self.cummulative_n_success_v,self.episode_number)
				self.summ_writer.add_scalar("AVTT:",self.cummulative_tt/self.cummulative_n_success_v,self.episode_number)
			# self.summ_writer.add_scalar("Routing Succcess:",self.cummulative_n_success_v,self.episode_number)
			# self.summ_writer.add_scalar("Routing failure:",self.cummulative_n_failed_v,self.episode_number)
			


		def close(self,sim_time):
			if self.is_terminal(sim_time):
				# self.summ_writer.close()

				if self.log_data:
					self.log_data()
				return self.states,self.acts,self.next_states,self.rewds,self.dones,True
			else:
				return self.states,self.acts,self.next_states,self.rewds,self.dones,False
		
		def close_traci(self):
			traci.close()

		# def init_traci(self):
		# 	sys.path.append(os.path.join(self.config.Constants['SUMO_PATH'], os.sep, 'tools'))
		# 	sumoBinary = self.config.Constants["SUMO_GUI_PATH"]
		# 	self.sumoCmd = [sumoBinary, '-S', '-d', self.config.Constants['Simulation_Delay'], "-c", self.config.Constants["SUMO_CONFIG"],"--no-warnings","true"]
		# 	traci.start(self.sumoCmd)
		


		def get_queries(self):
			routing_queries=set()
			exiting_queries=set()
			self.utils.log("a vehicle can be detected twice in a single timestep!",type='warn')
			AllSubsResult=traci.inductionloop.getAllSubscriptionResults()
			for il in AllSubsResult:
				tvcs=AllSubsResult[il][self.config.Constants['il_last_step_vc_IDs_subscribtion_code']]
				if len(tvcs)>0:
					split=il.split('_')
					il_edge=split[1]
					il_lane=split[1]+'_'+split[2]
					il_edge_head=self.network.get_edge_head_node(il_edge)
					# il_lane=
					for vc in tvcs:
						# lane_ID=il[self.config.Constants['il_lane_ID_subscribtion_code']]
						# breakpoint()
						if self.has_exited(vc) or self.has_transited(vc,il_edge):
							continue
						
						if self.config.Constants['Analysis_Mode']:
							try:
								assert(self.vehicles[vc]["road"]!=il_edge)
							except Exception as e:
								breakpoint()

						if self.has_arrived(vc,il_edge_head):
							exiting_queries.add((vc,il_edge))
						else:
							routing_queries.add((vc,il_edge))					

			return routing_queries,exiting_queries
		
		def has_exited(self,vc):
			return vc not in self.vehicles

		def has_transited(self, vc,il_road):
			if self.vehicles[vc]["is_new"]:
				return False
			return self.vehicles[vc]["road"]==il_road

		def has_arrived(self,vc,il_road_head):
			return il_road_head==self.vehicles[vc]['destination']

		def get_routing_queries(self):
			return self.routing_queries_states

		
		# def set_substitue_action(self,vc,current_road,current_lane,agent_id):
		# 	self.vehicles[vc]["is_action_valid"]=False
		# 	self.vehicles[vc]["substitute_action"]=self.get_subtitue_action(current_lane,agent_id)
		# 	if self.config.Constants['Analysis_Mode']:
		# 		try:
		# 			assert(self.vehicles[vc]["substitute_action"])!= None
		# 			assert(self.vehicles[vc]["substitute_action"]<self.utils.agent_dic[agent_id][1])
		# 		except Exception as e:
		# 			breakpoint()
		# 	next_roads = self.utils.get_next_road_IDs(agent_id,self.vehicles[vc]["substitute_action"])
		# 	self.cummulative_n_invalid_actions+=1
		# 	if self.config.Constants['Analysis_Mode']:
		# 		try:
		# 			traci.vehicle.setRoute(vc, [current_road]+ next_roads)	
		# 			self.utils.log("agnet {} generated substitute routing response {} for {}".format(agent_id,[current_road]+ next_roads,vc))
		# 		except Exception as e:
		# 			self.utils.log("error in setting the substitute route",type='err')
		# 			breakpoint()

		# def get_subtitue_action(self, lane_ID,agent_id):
		# 	"""
		# 	return a random number between 1 and len(self.utils.get_out_edges(self.g)t_intersec_id(vc_ID)])
		# 	"""
		# 	edge_ID=self.utils.get_lane_edge(lane_ID)
		# 	connections=self.network.get_edge_connections(edge_ID)
		# 	subs_edge=random.choice(connections)
		# 	subs_act=self.utils.get_edge_index_among_node_out_edges(subs_edge,agent_id)
		# 	return subs_act