from environments.sumo.environment.Utils import Constants
from environments.sumo.environment.Utils import Utils
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
			self.state_size=self.utils.get_state_diminsion()
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
			self.cummulative_n_ex_v=0
			self.summ_writer = SummaryWriter()
			self.TTSPWRR=False
			self.TTSP=False

		def run(self):
			# traci.start(self.sumoCmd)
			# self.reset()
			step = 0
			while not self.step():
				step += 1
			# traci.close()
			

		def reset(self,episode):
			self.utils.log("restart called. Not implemented.",type='warn')
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
			self.change_traffic_condition(sim_time,5,400)
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

			self.transient_avs=[vc for vc in traci.inductionloop.getLastStepVehicleIDs('e1Detector_gneE1_0_0') if self.is_transient(vc)]
			exiting_vcs=[vc for vc in traci.inductionloop.getLastStepVehicleIDs('e1Detector_gneE5_0_1') if vc in self.vehicles]
						
			for vc in self.transient_avs:
				# self.add_to_next_trans_for_routing(vc)
				road=traci.vehicle.getRoadID(vc)
				agent_id=self.utils.get_edge_path_tail_node(road)
				state=self.utils.get_state(agent_id,self.vehicles[vc]["destination"])
				self.transient_avs_states.append(state)
				if not self.vehicles[vc]["is_new"]:
					reward=self.get_reward(self.vehicles[vc]["time"],sim_time)
					self.save_expirence(vc,reward,state,done=False)

				self.update_env_vc_info(vc, sim_time,road,agent_id,state)	


			for vc in exiting_vcs:
				assert(not self.vehicles[vc]["is_new"])
				reward=self.get_reward(self.vehicles[vc]["time"],sim_time)
				self.save_expirence(vc,reward,next_state=None,done=True)
				self.compute_AVTT(vc,sim_time)
				self.exit(vc)
			# ----------------------------------------
			self.log_AVTT(sim_time,period=400)
			return self.close(sim_time)

		def get_random_action(self, vc_ID):
			"""
			return a random number between 1 and len(self.utils.get_out_edges(self.g)t_intersec_id(vc_ID)])
			"""

			if len(self.manual_drive)!=0:
				self.current_step+=1
				return self.manual_drive[self.current_step]

			source = self.vehicles[vc_ID]["agent_id"]
			assert len(self.utils.get_out_edges(source)) > 1 
			action=random.choice(range(0, len(self.utils.get_out_edges(source))))
			self.utils.log("action: " +str(action)+" chosen for vehicle: "+vc_ID+" out of "+str(len(self.utils.get_out_edges(source)))+" actions")
			return action

		def set_route(self,VCs,ACTs):
			if self.TTSP:
				return
			for vc, ac in zip(VCs, ACTs):
				assert(ac!=None)
				self.vehicles[vc]["action"]=ac

				if self.TTSPWRR:
					traci.vehicle.rerouteTraveltime(vc)
					continue

				agent_id=self.vehicles[vc]["agent_id"]
				next_roads = self.utils.get_next_road_IDs(agent_id,ac)
				
				current_road = self.vehicles[vc]["road"]
				try:
					traci.vehicle.setRoute(vc, [current_road]+ next_roads)
					self.utils.log('Sucess! {0} route changed. current_road:{1}, next_road:{2})'.format(vc, current_road, next_roads))

				except Exception as e:
					self.utils.log(str(e),type='err')
					breakpoint()
		
		def is_transient(self, vc):

			return  self.vehicles[vc]["is_new"] or self.vehicles[vc]["road"]!=traci.vehicle.getRoadID(vc)

		def change_traffic_condition(self,time,vc_period,traffic_period):
			if time%vc_period==0:
				vid,road,destination=self.utils.generate_random_trip(time)
				self.add_env_vc(vid,road,time,destination)
			if time%traffic_period==1:
				self.utils.log("speed --")
				traci.edge.setMaxSpeed('gneE6', 2)
			if time%traffic_period==math.floor(traffic_period/2)+1:
				self.utils.log("speed ++")
				traci.edge.setMaxSpeed('gneE6', 14)


		def add_env_vc(self,vc,road,time,destination):
				self.utils.log(vc+" entered simulation @ road:"+road)
				self.vehicles[vc]={
					"destination": destination,
					"start_time":time,
					"time": None,
					"road":None,
					"agent_id":None,
					"action":None,
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
		

		def save_expirence(self,vc,reward,next_state,done):
			try:
				assert(self.vehicles[vc]["action"]!=None)
				assert(self.vehicles[vc]["state"]!=None)
			except Exception as e:
				self.utils.log(str(e),type='err')
				breakpoint()
			self.states.append(self.vehicles[vc]["state"])
			self.acts.append(self.vehicles[vc]["action"])
			self.next_states.append(next_state)
			self.rewds.append(reward)
			self.dones.append(done)
			self.utils.log("S:{},A:{},R:{},NS:{},D:{}".format(self.vehicles[vc]["state"],\
															self.vehicles[vc]["action"],\
															reward,next_state,done))



		def get_engine_vehicles_dic(self):
			"""
			query to the SUMO API and return a dictionary of the vehicles that are currently in the simulation

			"""
			vehicleIDs = traci.vehicle.getIDList()
			return {id: traci.vehicle.getRoadID(id) for id in vehicleIDs}

		def exit(self, vc):
			self.utils.log("{} exited.".format(vc))
			self.vehicles.pop(vc)


		def __is_internal(self, edgeID):
			return edgeID not in self.utils.edge_ID_dic

			

		# def add_to_next_trans_for_routing(self,vc):
		# 	try:
		# 		self.utils.log("an assertin is needed in <add_to_next_trans_for_routing>", type='warn')
		# 		# assert(self.utils.road2int(self.vehicles[vc]["road"]) in self.lanes_dic)
		# 	except:
		# 		breakpoint()
		# 	self.transient_avs.append(vc)
		# 	# self.transient_avs_states.append(self.vehicles[vc]["state"])		
		
		def is_manual_on(self,vc):
			return self.utils.get_flow_id(vc) in self.skip_routing

		def compute_AVTT(self,vc,sim_time):
			self.cummulative_tt+=sim_time-self.vehicles[vc]["start_time"]
			self.cummulative_n_ex_v+=1
		
		def log_AVTT(self,sim_time,period):
			if sim_time%period==period-1:
				self.summ_writer.add_scalar("AVTT:",self.cummulative_tt/self.cummulative_n_ex_v,int(sim_time))
				self.cummulative_tt=0
				self.cummulative_n_ex_v=0

		def close(self,sim_time):
			if self.is_terminal(sim_time):
				# self.summ_writer.close()
				return self.states,self.acts,self.next_states,self.rewds,self.dones,True
			else:
				return self.states,self.acts,self.next_states,self.rewds,self.dones,False
		
		def is_terminal(self,sim_time):
			"""
			check for some terminal condition. e.g. all vehicles exited the simulation or the time limit has passed

			"""
			return traci.simulation.getMinExpectedNumber() == 0 or sim_time%4000==3999


		def init_traci(self):
			sys.path.append(os.path.join(Constants['SUMO_PATH'], os.sep, 'tools'))
			sumoBinary = Constants["SUMO_GUI_PATH"]
			self.sumoCmd = [sumoBinary, '-S', '-d', Constants['Simulation_Delay'], "-c", Constants["SUMO_CONFIG"]]
			traci.start(self.sumoCmd)
		
		def close_traci(self):
			traci.close()