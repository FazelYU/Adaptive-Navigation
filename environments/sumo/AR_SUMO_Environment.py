from environment.Utils import Utils
from model.network import RoadNetworkModel
import os, sys
import traci
import random


class AR_SUMO_Environment():

	def __init__(self):
		#create the road network
		self.network = RoadNetworkModel(Utils["ROOT"], Utils["Network_XML"])
		# print(self.network.junctions, self.network.edges)
		#traci sumo init
		sys.path.append(os.path.join(Utils['SUMO_PATH'], os.sep, 'tools'))
		sumoBinary = Utils["SUMO_GUI_PATH"]
		self.sumoCmd = [sumoBinary, "-c", Utils["SUMO_CONFIG"]]
		
		# for id, edge in self.network.edges.items():
		# 	print(id, edge.type)
		# for edge in self.network.graph.edges():
		# 	print(self.network.graph.get_edge_data(*edge)['edge'].id)

		self.vehicles={} # holds a copy of the info of the engine vehicles. I use it to find the vehicles that change road. it may be redundent with SUMO (SUMO might have an API that keeps track of the vehicles that change road)
		self.trans_vehicles=[]

	def run(self):
		traci.start(self.sumoCmd)
		# self.reset()
		step = 0
		while not self.step():
			step += 1
		traci.close()

	# def reset(self,episode):
	# 	self.eng.reset()
	# 	# set up replay saving and other file mangements. 
	# 	self.eng.set_save_replay(True)
	# 	self.eng.set_replay_file("replays/episode"+str(episode)+".txt")
	# 	self.refresh()
	# 	#CityFlow needed one extra step at the begining. IDK about SUMO.
	# 	self.eng.next_step()
	# 	return False
	# 	# return self.state
	# 	# assuming that there is no TVs at the beginning
	
	def refresh(self):
		self.vehicles={}
		self.refresh_trans()

	def refresh_trans(self):
		self.trans_vehicles=[]
		
	def step(self):
		# self.trans_vehicles is an array of the vehicle_IDs that changed road in the last call to step()
		actions = [0 for i in range(len(self.trans_vehicles))]
		for i in range(len(actions)): actions[i] = self.get_random_action(self.trans_vehicles[i])
			
		self.set_route(self.trans_vehicles, actions)
			
		traci.simulationStep()
		self.refresh_trans()

		if self.is_terminal():
			return True

		eng_vehicles_dic=self.get_engine_vehicles_dic() 
		self.update_env_vehicles(eng_vehicles_dic)
		for vc in eng_vehicles_dic:
			if self.is_trans(vc):
				self.transit_vc(vc, eng_vehicles_dic[vc][0])	
	
		return False

	def set_route(self,VCs,ACTs):
		"""
		input: 
			VCs: an array of vehicle IDs
			ACTs: an array of actions, each action represents one of the roads connected to the next-intersection of the vc
		Out: 
			void

		Function:
			for vc,ac in zip(VCs,ACTs):
				next_intersection_id = self.get_intersec_id(vc)
				roads= self.intersec_dic[next_intersection_id]
						(roads can be an array)

				selected-road=roads[ac]
				route vc to selected-road:
					set up the simulator to route the vc to selected-road accordingly in the next step()
				
				Note: e.g. at a intersection "ij" action 1 should always route vc to "road1" connected to "ij"

		"""
		for vc, ac in zip(VCs, ACTs):
			if ac is None:
				continue
			next_intersection_id = self.get_intersec_id(vc)
			roads = self.get_intersec_dic()[next_intersection_id]
			selected_road = roads[ac - 1]
			current_road = self.vehicles[vc][0]
			# print(current_road, selected_road)
			traci.vehicle.setRoute(vc, [current_road, selected_road])
		pass


	def get_intersec_dic(self):
		"""
		input: void
		return: a dictionary for all intersections in the map: 
								the keys : intersection ids
								the values : list of out going roads that are connected to the intersection
		"""
		return {node: [self.network.graph.get_edge_data(*edge)['edge'].id for edge in self.network.graph.out_edges(node)] for node in self.network.graph.nodes()}
	
	def get_intersec_id(self,vc_ID):
		"""
		return the next intersection of the vehicle with id= "vc_ID"
		"""
		roadID = self.vehicles[vc_ID][0]
		edge = list(filter(lambda e: self.network.graph.get_edge_data(*e)['edge'].id == roadID, self.network.graph.edges()))[0]
		return edge[1]

	
	def get_random_action(self, vc_ID):
		"""
		return a random number between 1 and len(self.intersec_dic[self.get_intersec_id(vc_ID)])
		"""
		roadID = self.vehicles[vc_ID][0]
		if len(self.network.connectionGraph.out_edges(roadID)) == 0:
			return None
		choices = list(range(1, len(self.get_intersec_dic()[self.get_intersec_id(vc_ID)]) + 1))
		choice = random.choice(choices)
		while self.get_intersec_dic()[self.get_intersec_id(vc_ID)][choice - 1] not in [pair[1] for pair in self.network.connectionGraph.out_edges(roadID)]:
			choice = random.choice(choices)
		return choice

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
		return {id: (traci.vehicle.getRoadID(id), False) for id in vehicleIDs}

	def update_env_vehicles(self, eng_vehicles_dic):
		removable=[]
		for vc in self.vehicles:
			if not vc in eng_vehicles_dic:
				removable.append(vc)
		for vc in removable:
			self.vehicles.pop(vc)



	def is_trans(self, vc):
		"""
		return true if the vehicle is new or it has changed road.
		I don't know if sumo provides API for this. 
		One way to get around with this is to compare the current road of vc in the engine with its road in the copy that we hold in self.vehicles
		"""
		return  (vc not in self.vehicles.keys()) or ((self.vehicles[vc][0] != traci.vehicle.getRoadID(vc)) and (not self.__is_internal(traci.vehicle.getRoadID(vc))))


	def __is_internal(self, roadID):
		edges = list(filter(lambda e: self.network.graph.get_edge_data(*e)['edge'].id == roadID, self.network.graph.edges()))
		if len(edges) == 0:
			return True
		edge = edges[0]
		edge = self.network.graph.get_edge_data(*edge)['edge']
		return edge.type == 'internal'
		

	def transit_vc(self, vc, road):
		self.vehicles[vc]= road, False
		self.trans_vehicles.append(vc)