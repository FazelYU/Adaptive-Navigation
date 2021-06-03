#""the goal is to migrate from CityFlow to SUMO""
import gym
# import cityflow 
# from environments.Utils import Utils
# import SUMO instead. It is a good practice to create your own Utils.py


class AR_SUMO_Environment():

	def __init__(self):
		# replace the engine with sumo API
		self.eng = cityflow.Engine("environments/3x3/config.json", thread_num=8)
		self.intersec_dic=self.get_intersec_dic()
		self.vehicles={} # holds a copy of the info of the engine vehicles. I use it to find the vehicles that change road. it may be redundent with SUMO (SUMO might have an API that keeps track of the vehicles that change road)
		self.trans_vehicles=[]

	def run(self):
		self.reset()
		while self.step():
			pass

	def reset(self,episode):
		self.eng.reset()
		# set up replay saving and other file mangements. 
		self.eng.set_save_replay(True)
		self.eng.set_replay_file("replays/episode"+str(episode)+".txt")
		self.refresh()
		#CityFlow needed one extra step at the begining. IDK about SUMO.
		self.eng.next_step()
		return False
		# return self.state
		# assuming that there is no TVs at the beginning
	
	def refresh(self):
		self.vehicles={}
		self.refresh_trans()

	def refresh_trans(self):
		self.trans_vehicles=[]
		
	def step(self):
		# self.trans_vehicles is an array of the vehicle_IDs that changed road in the last call to step()
		actions=[0]*len(self.trans_vehicles)
		for i in range(len(actions)): actions[i]=self.get_random_action(self.trans_vehicles[i])
			
		self.set_route(self.trans_vehicles,actions)
			
		self.eng.next_step() #this is a call to the CityFlow API. SUMO must have a similar API. Replace this line with SUMO version
		self.refresh_trans()

		if self.is_terminal():
			return True

		eng_vehicles_dic=self.get_engine_vehicles_dic() 
		self.update_env_vehicles(eng_vehicles_dic)
		
		for vc in eng_vehicles_dic:
			if self.is_trans(vc):
				self.transit_vc(vc)	
	
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
		pass


	def get_intersec_dic(self):
		"""
		input: void
		return: a dictionary for all intersections in the map: 
								the keys : intersection ids
								the values : list of out going roads that are connected to the intersection
		"""
		pass
	
	def get_intersec_id(self,vc_ID):
		"""
		return the next intersection of the vehicle with id= "vc_ID"
		"""
	
	def get_random_action(self,vc_ID):
		"""
		return a random number between 1 and len(self.intersec_dic[self.get_intersec_id(vc_ID)])
		"""
		pass

	def is_terminal(self):
		"""
		check for some terminal condition. e.g. all vehicles exited the simulation or the time limit has passed

		"""
		pass

	def get_engine_vehicles_dic(self):
		"""
		query to the SUMO API and return a dictionary of the vehicles that are currently in the simulation

		"""
		pass

	def update_env_vehicles(self,eng_vehicles_dic):
		removable=[]
		for vc in self.vehicles:
			if not vc in eng_vehicles_dic:
				removable.append(vc)
		for vc in removable:
			self.vehicles.pop(vc)



	def is_trans(self,vc):
		"""
		return true if the vehicle is new or it has changed road.
		I don't know if sumo provides API for this. 
		One way to get around with this is to compare the current road of vc in the engine with its road in the copy that we hold in self.vehicles
		"""
		# return self.is_new(vc) or self.get_road(vc) != self.vehicles[vc]["memory0"]["road"]
		pass

	def transit_vc(self,vc):
		self.vehicles[vc]=self.get_road(vc)
		self.trans_vehicles.append(vc)