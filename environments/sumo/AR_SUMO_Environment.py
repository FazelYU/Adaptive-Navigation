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
		def __init__(self,GAT,embed_network,Num_Flows,
					skip_routing,random_trips, Max_Sim_Time,
					 device, Log, rolling_window):
			#create the road network
			

			self.init_traci()
			self.network = RoadNetworkModel(Constants["ROOT"], Constants["Network_XML"])
			self.utils=Utils(environment=self,network=self.network,Num_Flows=Num_Flows,device=device,GAT=GAT,embed_network=embed_network)
			# breakpoint()
			# traci.edge.adaptTraveltime('gneE6',4*traci.edge.getTraveltime('gneE6'))

			self.reloadCmd = []


			self.vehicles={} # holds a copy of the info of the engine vehicles. I use it to find the vehicles that change road. it may be redundent with SUMO (SUMO might have an API that keeps track of the vehicles that change road)
			self.transient_avs=[]
			
			# -----------------------------------------------------
			self.gat=GAT
			self.transient_avs_states=[]
			self.states=[]
			self.acts=[]
			self.next_states=[]
			self.rewds=[]
			self.dones=[]
			self.Max_Sim_Time=Max_Sim_Time
			self.device=device
			self.lanes_dic={}

			self.stochastic_actions_probability = 0
			self.actions = set(range(3))
			self.id = "Adaptive Routing"
			self.action_space = spaces.Discrete(3)
			# self.state_size=self.utils.get_state_diminsion()
			self.seed()
			self.reward_threshold = 0.0
			self.trials = rolling_window
			self.Log=Log
			self.skip_routing=skip_routing
			self.iteration=-1
			self.random_trips=random_trips
			self.embed_network=embed_network
			# self.manual_drive=[1,0,2,0,2,0]
			self.manual_drive=[]
			self.current_step=-1
			self.cummulative_tt=0
			self.cummulative_n_success_v=0
			self.cummulative_n_failed_v=0
			self.cummulative_n_invalid_actions=0
			self.TTSPWRR=False
			self.TTSP=False
			self.should_log_data=True
			if self.should_log_data:
				self.summ_writer = SummaryWriter()
						
		def reset(self,episode,config):
			self.episode_number=episode
			if episode==config.num_episodes_to_run-1:
				Constants["LOG"]=True
				breakpoint()
			# self.vehicles={}
			# self.refresh_trans()
			# self.refresh_exp()
			# for vc in traci.vehicle.getIDList():
			# 	traci.vehicle.remove(vc)

		def refresh_trans(self):
			self.transient_avs=[]
			self.transient_avs_states=[]
		
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
			self.change_traffic_condition(sim_time,50,400)
			# ------------------------------------------
			# actions=[self.get_random_action(trans_vehicle) for trans_vehicle in self.transient_avs]				
			try:
				assert(len(self.transient_avs)==len(actions))
			except Exception as e:
				breakpoint()

			self.set_route(self.transient_avs, actions)
			#-------------------------------------------- 
			traci.simulationStep()

			if self.embed_network:
				self.utils.update_and_evolve_node_features()
			# -------------------------------------------

			self.refresh_exp()
			self.refresh_trans()
			# -------------------------------------------
			# for vc in traci.vehicle.getIDList():
			# 	if sim_time>self.vehicles[vc]["dead_line"]:
			# 		self.fail_routing(vc,sim_time)

			for vc in self.get_transient_avs():
				# self.add_to_next_trans_for_routing(vc)
				road=traci.vehicle.getRoadID(vc)
				try:
					assert(road in self.utils.edge_ID_dic)
				except:
					self.utils.log("{} has entered an internal edge. Too late for making a routing decision".format(vc))
					breakpoint()

				agent_id=self.utils.get_edge_path_head_node(road)			
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
					if self.vehicles[vc]["is_action_valid"]:
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],reward,next_state,done=False)
					else:
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],10*reward,next_state,done=False)
						self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["substitute_action"],reward,next_state,done=False)


				try:
					assert(next_state!=None)
				except Exception as e:
					breakpoint()
				
				self.update_env_vc_info(vc, sim_time,road,agent_id,next_state)
				self.transient_avs.append(vc)
				self.transient_avs_states.append(next_state)

			for vc in self.get_exiting_avs():
				assert(not self.vehicles[vc]["is_new"])
				reward=self.get_reward(self.vehicles[vc]["time"],sim_time)

				if self.vehicles[vc]["is_action_valid"]:
					self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],reward,next_state=None,done=True)
				else:
					self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["action"],100*reward,next_state=None,done=True)
					self.save_expirence(self.vehicles[vc]["state"],self.vehicles[vc]["substitute_action"],reward,next_state=None,done=True)

				self.success_routing(vc,sim_time)
			# ----------------------------------------
			return self.close(sim_time)


		def set_route(self,VCs,ACTs):
			
			for vc, ac in zip(VCs, ACTs):
				assert(ac!=None)
				self.vehicles[vc]["action"]=ac
				agent_id=self.vehicles[vc]["agent_id"]
				next_roads = self.utils.get_next_road_IDs(agent_id,ac)
				current_road = self.vehicles[vc]["road"]
				self.utils.log("agnet {} generated routing response {} for {}".format(agent_id,[current_road]+ next_roads,vc))

				if self.TTSP:
					self.utils.log("Routing mode TTSP. Action discarded.")
					continue

				if self.TTSPWRR:
					self.utils.log("Routing mode TTSPWRR. Action discarded.")
					traci.vehicle.rerouteTraveltime(vc)
					continue

				try:
					traci.vehicle.setRoute(vc, [current_road]+ next_roads)
					self.utils.log("there may be other reasons for failure of setRoute!",type='warn')
					self.utils.log('Sucess! {0} route changed. current_road:{1}, next_road:{2})'.format(vc, current_road, next_roads))
					self.vehicles[vc]["is_action_valid"]=True

				except Exception as e:
					self.utils.log('Failed! Action {} is not valid @ road {}'.format(ac,current_road))
					self.set_substitue_action(vc,current_road,agent_id)


		


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

		def set_substitue_action(self,vc,current_road,agent_id):
			self.vehicles[vc]["is_action_valid"]=False
			self.vehicles[vc]["substitute_action"]=self.get_subtitue_action(current_road,agent_id)
			next_roads = self.utils.get_next_road_IDs(agent_id,self.vehicles[vc]["substitute_action"])
			self.cummulative_n_invalid_actions+=1
			try:
				traci.vehicle.setRoute(vc, [current_road]+ next_roads)	
				self.utils.log("agnet {} generated substitute routing response {} for {}".format(agent_id,[current_road]+ next_roads,vc))
			except Exception as e:
				self.log("error in setting the substitute route",type='err')
				breakpoint()

		def get_subtitue_action(self, road_ID,agent_id):
			"""
			return a random number between 1 and len(self.utils.get_out_edges(self.g)t_intersec_id(vc_ID)])
			"""
			self.utils.log("I am assuming that each edge has only one lane",type='warn')
			subs_edge=random.choice(traci.lane.getLinks(road_ID+"_0"))[0].split('_')[0]
			return self.utils.get_edge_index_among_node_out_edges(subs_edge,agent_id)
		
		def change_traffic_condition(self,time,vc_period,traffic_period):
			if traci.vehicle.getIDCount()>15:
				return
			if time%vc_period==0:
			# if time==0:
				vid,road,destination,dead_line=self.utils.generate_random_trip(time)
				self.add_env_vc(vid,road,time,destination,dead_line)
			# if time%traffic_period==1:
			# 	self.utils.log("speed --")
			# 	traci.edge.setMaxSpeed('gneE6', 2)
			# if time%traffic_period==math.floor(traffic_period/2)+1:
			# 	self.utils.log("speed ++")
			# 	traci.edge.setMaxSpeed('gneE6', 14)
		
		
		def log_data(self):
			# self.summ_writer.add_scalar("AVTT:",self.cummulative_tt/self.cummulative_n_success_v,self.episode_number)
			self.summ_writer.add_scalar("Routing Succcess:",self.cummulative_n_success_v,self.episode_number)
			self.summ_writer.add_scalar("Routing failure:",self.cummulative_n_failed_v,self.episode_number)
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
			return sim_time%4000==3999

		def init_traci(self):
			sys.path.append(os.path.join(Constants['SUMO_PATH'], os.sep, 'tools'))
			sumoBinary = Constants["SUMO_GUI_PATH"]
			self.sumoCmd = [sumoBinary, '-S', '-d', Constants['Simulation_Delay'], "-c", Constants["SUMO_CONFIG"]]
			traci.start(self.sumoCmd)
		
		def close_traci(self):
			traci.close()

		def is_transient(self, vc):

			return  self.vehicles[vc]["is_new"] or self.vehicles[vc]["road"]!=traci.vehicle.getRoadID(vc)

		def get_transient_avs(self):
			transient_avs=[]
			for il in self.utils.transition_induction_loops:
				transient_avs+=[vc for vc in traci.inductionloop.getLastStepVehicleIDs(il) if vc in self.vehicles and self.is_transient(vc)]
			return transient_avs
		
		def get_exiting_avs(self):
			exiting_avs=[]
			for il in self.utils.sink_edge_induction_loops:
				exiting_avs+=[vc for vc in traci.inductionloop.getLastStepVehicleIDs(il) if vc in self.vehicles and self.utils.get_edge_head_node(self.utils.get_induction_loop_edge(il))==self.vehicles[vc]['destination']]
			
			return exiting_avs