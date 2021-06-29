from environments.sumo.environment.Utils import Constants
from environments.sumo.environment.Utils import Utils
from environments.sumo.model.network import RoadNetworkModel
import os, sys
import traci
import random
import gym
from gym import spaces


class AR_SUMO_Environment(gym.Env):
		environment_name = "Adaptive Routing SUMO"
		def __init__(self,GAT,embed_network,Num_Flows,
					skip_routing,random_trips, Max_Sim_Time,
					 device, Log, rolling_window):
			#create the road network
			sys.path.append(os.path.join(Constants['SUMO_PATH'], os.sep, 'tools'))
			sumoBinary = Constants["SUMO_GUI_PATH"]
			self.sumoCmd = [sumoBinary, '-S', '-d', Constants['Simulation_Delay'], "-c", Constants["SUMO_CONFIG"]]
		
			traci.start(self.sumoCmd)

			self.network = RoadNetworkModel(Constants["ROOT"], Constants["Network_XML"])
			self.utils=Utils(environment=self,network=self.network,Num_Flows=Num_Flows,device=device,GAT=GAT,embed_network=embed_network)
			
			# breakpoint()
			# traci.edge.adaptTraveltime('gneE6',4*traci.edge.getTraveltime('gneE6'))

			self.reloadCmd = []


			self.vehicles={} # holds a copy of the info of the engine vehicles. I use it to find the vehicles that change road. it may be redundent with SUMO (SUMO might have an API that keeps track of the vehicles that change road)
			self.trans_vehicles=[]
			
			# -----------------------------------------------------
			self.gat=GAT
			self.trans_vehicles_states=[]
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

			

		def run(self):
			# traci.start(self.sumoCmd)
			# self.reset()
			step = 0
			while not self.step():
				step += 1
			# traci.close()
			

		def reset(self,episode):
			self.utils.log("restart called. Not implemented.")
		

		def refresh(self):
			self.vehicles={}
			self.refresh_trans()
			self.refresh_exp()

		def refresh_trans(self):
			self.trans_vehicles=[]
			self.trans_vehicles_states=[]
		
		def refresh_exp(self):
			self.states=[]
			self.acts=[]
			self.next_states=[]
			self.rewds=[]
			self.dones=[]	
			
		def step(self,actions=[]):
			self.utils.log("sim_time: "+str(traci.simulation.getTime()))
			# traci.edge.getLastStepMeanSpeed('gneE6')
			if traci.simulation.getTime()%5==0:
				self.utils.generate_random_trip(traci.simulation.getTime())
			if traci.simulation.getTime()%200==1:
				self.utils.log("speed --")
				traci.edge.setMaxSpeed('gneE6', 2)
			if traci.simulation.getTime()%200==101:
				self.utils.log("speed ++")
				traci.edge.setMaxSpeed('gneE6', 14)

			# ------------------------------------------
			# self.trans_vehicles is an array of the vehicle_IDs that changed road in the last call to step()
			actions=[self.get_random_action(trans_vehicle) for trans_vehicle in self.trans_vehicles]	
			# if len(actions)!=0:
			# 	breakpoint()
			
			self.set_route(self.trans_vehicles, actions)
			#-------------------------------------------- 
			traci.simulationStep()

			if self.embed_network:
				self.utils.update_and_evolve_node_features()

			self.refresh_trans()	
			self.refresh_exp()

			# -------------------------------------------
			eng_vehicles_dic=self.get_engine_vehicles_dic()
			transient_vcs=[vc for vc in eng_vehicles_dic if self.is_trans(vc)]
			exiting_vcs=[vc for vc in self.vehicles if vc not in eng_vehicles_dic]
						
			for vc in transient_vcs:
				# if self.is_manual_on(vc):
				# 	continue
				self.transit_env_vc(vc, eng_vehicles_dic[vc])	
				if self.vehicles[vc]["memory0"]["valid"]:
					self.add_to_next_trans_for_routing(vc)
				if self.vehicles[vc]["memory2"]!=None:
					self.save_expirence(vc,self.get_reward(vc),done=False)				

			for vc in exiting_vcs:
				assert(vc not in transient_vcs)
				self.transit_env_vc(vc, None)
				assert(self.vehicles[vc]["memory2"]!=None)
				self.save_expirence(vc,self.get_reward(vc),done=True)
				self.exit(vc)	
			
			if self.is_terminal():
				traci.close()
				return True
				return self.states,self.acts,self.next_states,self.rewds,self.dones,True
			else:
				return False
				return self.states,self.acts,self.next_states,self.rewds,self.dones,False

		def get_random_action(self, vc_ID):
			"""
			return a random number between 1 and len(self.utils.get_out_edges(self.g)t_intersec_id(vc_ID)])
			"""

			if len(self.manual_drive)!=0:
				self.current_step+=1
				return self.manual_drive[self.current_step]

			self.utils.log("we are selecting a neighboring node as action. we are bette choose an out going edge",type='warn')
			source = self.vehicles[vc_ID]["memory0"]["source"]
			assert len(self.utils.get_out_edges(source)) > 1 
			action=random.choice(range(0, len(self.utils.get_out_edges(source))))
			self.utils.log("action: " +str(action)+" chosen for vehicle: "+vc_ID+" out of "+str(len(self.utils.get_out_edges(source)))+" actions")
			return action

		def set_route(self,VCs,ACTs):
			for vc, ac in zip(VCs, ACTs):
				assert(ac!=None)
				self.vehicles[vc]["memory0"]["action"]=ac
				source=self.vehicles[vc]["memory0"]["source"]
				next_roads = self.utils.get_next_road_IDs(source,ac)
				
				current_road = self.vehicles[vc]["memory0"]["road"]
				try:
					traci.vehicle.setRoute(vc, [current_road]+ next_roads)
					self.utils.log('Sucess! {0} route changed. current_road:{1}, next_road:{2})'.format(vc, current_road, next_roads))

				except Exception as e:
					self.utils.log(str(e),type='err')
					breakpoint()
		
		def is_trans(self, vc):
			"""
			return true if the vehicle is new or it has changed road.
			I don't know if sumo provides API for this. 
			One way to get around with this is to compare the current road of vc in the engine with its road in the copy that we hold in self.vehicles
			"""
			self.utils.log("with current mechanism of finding transient vehicles, we detect it after hapening. this may not work in case of traffic signals",type='warn')
			return  self.is_new(vc) or \
					( \
						not self.__is_internal(traci.vehicle.getRoadID(vc)) and \
						self.vehicles[vc]["memory0"]["source"] != \
						self.utils.get_edge_path_tail_node(traci.vehicle.getRoadID(vc)) \
					)

		def transit_env_vc(self, vc, road):
			if self.is_new(vc):
				self.utils.log(vc+" entered simulation @ road:"+road)
				destination=self.utils.get_destination(vc)
				source=self.utils.get_edge_path_tail_node(road)
				

				self.vehicles[vc]={

				"destination": destination,
				"memory0":{
					"time": self.utils.get_time(),
					"road":road,
					"source":source,
					"action":None,
					"reward":None,
					"state":self.utils.get_state(source,destination),
					"valid": True
					# self.utils.check_valid(source)
				},
				"memory1":None,
				"memory2":None
				}
				return

			self.vehicles[vc]["memory2"]=None if self.vehicles[vc]["memory1"]==None else self.vehicles[vc]["memory1"].copy()
			self.vehicles[vc]["memory1"]=self.vehicles[vc]["memory0"].copy()
			if road==None:
				self.utils.log(vc+" exited.")
				self.vehicles[vc]["memory0"]={"time": self.utils.get_time()}
				return
			
			
			self.utils.log(vc+" changed its line to road:"+road)
			try:
				self.vehicles[vc]["memory0"]["time"]=self.utils.get_time()
				source=self.utils.get_edge_path_tail_node(road)
				destination=self.vehicles[vc]["destination"]
				self.vehicles[vc]["memory0"]["road"]=road
				self.vehicles[vc]["memory0"]["source"]=source
				self.vehicles[vc]["memory0"]["state"]=self.utils.get_state(source,destination)
				self.vehicles[vc]["memory0"]["valid"]=self.utils.is_valid(source)
			except Exception as e:
				self.utils.log(str(e),type='err')
				breakpoint()

		def get_reward(self,vc):
			t_prime=self.vehicles[vc]["memory1"]["time"]
			t_zgond=self.vehicles[vc]["memory0"]["time"]
			return t_prime-t_zgond
		

		def save_expirence(self,vc,reward,done):
			state=self.vehicles[vc]["memory2"]["state"]
			action=self.vehicles[vc]["memory2"]["action"]
			next_state=self.vehicles[vc]["memory1"]["state"]
			self.states.append(state)
			self.acts.append(action)
			self.next_states.append(next_state)
			self.rewds.append(reward)
			self.dones.append(done)
			self.utils.log("S:{},A:{},R:{},NS:{},D:{}".format(state,action,reward,next_state,done))


		def is_terminal(self):
			"""
			check for some terminal condition. e.g. all vehicles exited the simulation or the time limit has passed

			"""
			return traci.simulation.getMinExpectedNumber() == 0

		def get_engine_vehicles_dic(self):
			"""
			query to the SUMO API and return a dictionary of the vehicles that are currently in the simulation

			"""
			vehicleIDs = traci.vehicle.getIDList()
			return {id: traci.vehicle.getRoadID(id) for id in vehicleIDs}

		def exit(self, vc):
			self.vehicles.pop(vc)


		def __is_internal(self, edgeID):
			return edgeID not in self.utils.edge_ID_dic

			

		def add_to_next_trans_for_routing(self,vc):
			try:
				self.utils.log("an assertin is needed in <add_to_next_trans_for_routing>", type='warn')
				# assert(self.utils.road2int(self.vehicles[vc]["memory0"]["road"]) in self.lanes_dic)
			except:
				breakpoint()
			self.trans_vehicles.append(vc)
			# self.trans_vehicles_states.append(self.vehicles[vc]["memory0"]["state"])		
		
		
		def is_manual_on(self,vc):
			return self.utils.get_flow_id(vc) in self.skip_routing
		
		def is_new(self,vc):
			return vc not in self.vehicles

