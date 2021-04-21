import copy
import random
from collections import namedtuple
import gym
from gym import wrappers
import numpy as np
import matplotlib as mpl
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot
from random import randint

import cityflow
from environments.Utils import Utils
import torch
import itertools
import numpy
import math
# eng = cityflow.Engine("3x3/config.json", thread_num=1)
# for i in range(1000):
# 	eng.next_step()
class Adaptive_Routing_Environment(gym.Env):
	environment_name = "Adaptive Routing"

	def __init__(self,dim,encode,Num_Flows,skip_routing,Max_Sim_Time,device,Log):
		self.eng = cityflow.Engine("environments/3x3/config.json", thread_num=8)
		self.vehicles={}
		self.trans_vehicles=[]
		self.trans_vehicles_states=[]
		self.Max_Sim_Time=Max_Sim_Time
		self.device=device
		self.Cnst_Rew=1
		self.scale_factor=100
		self.valid_set=[110,111,112,113,
						210,211,212,213,
						310,311,312,313,
						120,121,122,123,
						220,221,222,223,
						320,321,322,323,
						130,131,132,133,
						230,231,232,233,
						330,331,332,333]
		self.utils=Utils(dim=dim,encode=encode,Num_Flows=Num_Flows,valid_set=self.valid_set)
		self.stochastic_actions_probability = 0
		self.actions = set(range(3))
		self.id = "Adaptive Routing"
		self.action_space = spaces.Discrete(3)
		self.state_size=self.utils.get_state_diminsion()
		self.seed()
		self.reward_threshold = 0.0
		self.trials = 100
		self.Log=Log
		self.skip_routing=skip_routing
	# def step(self, desired_action):
	# 	return self.s, self.reward, self.done, {}
	# 	# must satisfy base agent conduct_action function line 200
	# 	# self.next_state, self.reward, self.done, _ = self.environment.step(action)
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def reset(self,episode):
		self.eng.reset()
		# breakpoint()
		self.eng.set_save_replay(True)
		self.eng.set_replay_file("replays/episode"+str(episode)+".txt")
		self.vehicles={}
		self.trans_vehicles=[]
		self.trans_vehicles_states=[]
		self.eng.next_step()
		return []
		# return self.state
		# assuming that there is no TVs at the beginning

	def step(self,actions):
		if self.Log:
			print("Simulation Time: :{:.2f}".format(self.eng.get_current_time()))
		try:
			assert(len(self.trans_vehicles)==len(actions))
		except Exception as e:
			breakpoint()
		
		try:
			self.set_route(self.trans_vehicles,actions)
		except Exception as e:
			print(e)
			breakpoint()
		
		self.eng.next_step()
		self.trans_vehicles=[]
		self.trans_vehicles_states=[]
		states=[]
		acts=[]
		next_states=[]
		rewds=[]
		dones=[]

		# we don't have to check if all the vehicles have arrived at their destination, 
		# instead set an appropriate Max_Sim_Time
		# in case we need it : 
		# eng.get_vehicle_count==0 ==> done

		if self.eng.get_current_time()==self.Max_Sim_Time:
			return states,acts,next_states,rewds,dones,True

		vehicles=self.eng.get_vehicle_speed().keys()
		# vehicles=list(itertools.chain.from_iterable(self.eng.get_lane_vehicles().values()))
		# if len(vehicles)!=0:
		# breakpoint()

		removable=[]
		for vc in self.vehicles.keys():
			if not vc in vehicles:
				removable.append(vc)

		for vc in removable:
			if self.Log:
				print("vehicle "+vc+" exited the simulation")	
			self.vehicles.pop(vc)
		
		for vc in vehicles:

			if self.utils.get_flow_id(vc) in self.skip_routing:
				# breakpoint()
				continue

			if vc not in self.vehicles.keys():
				if self.Log:
					print("vehicle "+vc+" entered simulation")
				
				self.trans_vehicles.append(vc)
				self.vehicles[vc]={
				"road": self.get_road(vc),
				"destination": self.get_destination(vc),
				"memory_2": None,
				"memory_1": None,
				"enter time": self.eng.get_current_time(),
				"last_trans_time": self.eng.get_current_time(),
				"last_trans_dur": 0,
				}
				self.vehicles[vc]["memory_0"]=[self.get_state(vc),None,False] # set the action later in set_route
			else:
				if self.get_road(vc) != self.vehicles[vc]["road"]:
					if self.Log:
						print("vehicle "+vc+" changed its line")

					self.trans_vehicles.append(vc)

					# breakpoint()
					self.vehicles[vc]["road"]= self.get_road(vc)
					# self.vehicles[vc]["memory_3"]=list(self.vehicles[vc]["memory_2"]) if self.vehicles[vc]["memory_2"]!=None else None
					self.vehicles[vc]["memory_2"]=list(self.vehicles[vc]["memory_1"]) if self.vehicles[vc]["memory_1"]!=None else None
					self.vehicles[vc]["memory_1"]=list(self.vehicles[vc]["memory_0"]) if self.vehicles[vc]["memory_0"]!=None else None
					self.vehicles[vc]["memory_0"][0]=self.get_state(vc)
					self.vehicles[vc]["memory_0"][1]=None
					self.vehicles[vc]["memory_0"][2]=False
					self.vehicles[vc]["last_trans_dur"]=self.eng.get_current_time()-self.vehicles[vc]["last_trans_time"]
					self.vehicles[vc]["last_trans_time"]=self.eng.get_current_time()
					# self.vehicles[vc]["last_act"]=self.vehicles[vc]["act"]
					# breakpoint()
					
					if self.vehicles[vc]["memory_2"]!=None:	
						done,reward=self.get_reward(vc)

						if self.Log:
							print(self.vehicles[vc]["memory_2"][0],
								self.vehicles[vc]["memory_2"][1],
								self.vehicles[vc]["memory_1"][0],
								reward)

						states.append(self.utils.reshape(self.vehicles[vc]["memory_2"][0],vc))
						acts.append(self.vehicles[vc]["memory_2"][1])
						next_states.append(self.utils.reshape(self.vehicles[vc]["memory_1"][0],vc))
						rewds.append(reward)
						dones.append(done)

		for tv in self.trans_vehicles:
			state=self.vehicles[tv]["memory_0"][0]
			state=self.utils.reshape(state,tv)
			state=torch.tensor(state, device=self.device, dtype=torch.float)
			self.trans_vehicles_states.append(state)
				
		return states,acts,next_states,rewds,dones,False

	def get_road(self,vc):

		return self.eng.get_vehicle_info(vc)["route"].split()[0]
	
	def get_destination(self,vc):
		path=self.eng.get_vehicle_info(vc)["route"].split()
		return path[len(path)-1]	

	def get_state(self,vc):
		derivable=list(map(int,self.eng.get_vehicle_info(vc)["drivable"].split('_')[1:5]))
		road=self.utils.derivable2road(derivable)
		destination=list(map(int,self.vehicles[vc]["destination"].split('_')[1:4]))
		state=road+destination
		return state

	def get_reward(self,vc):
	
		if not self.vehicles[vc]["memory_2"][2]:
			road=self.vehicles[vc]["memory_1"][0][0:3]
			roadD=self.vehicles[vc]["memory_1"][0][3:6]
			if road==roadD:
				
				TT=self.eng.get_current_time()-self.vehicles[vc]["enter time"]
				SPTT=self.utils.get_Shoretest_Path_Travel_Time(vc)
				# print()
				# print(vc)
				# print(TT)

				reward=1+math.exp(SPTT/TT)

				if self.Log:
					print("goal reached: {:.2f}".format(reward))
			else:
				reward=-2
				if self.Log:
					print("dead-end")
			
			return True,reward

		time=self.vehicles[vc]["last_trans_dur"]
		state1=self.vehicles[vc]["memory_1"][0]
		state2=self.vehicles[vc]["memory_0"][0]
		intersec1=state1[0:2]
		intersec2=state2[0:2]
		intersecD=state1[3:5]
		# road2=state2[0:3]
		# roadD=state1[3:6]		


		Dist_1_D=self.utils.get_distance(intersec1,intersecD)
		Dist_2_D=self.utils.get_distance(intersec2,intersecD)

		# reward= (Dist_1_D-Dist_2_D)*10

		if Dist_1_D>Dist_2_D:
				reward=0
		else:
				reward=-1
		
		if self.Log:
				print("moving reward: "+ str(reward))

		return False,reward

	def set_route(self,vehicles,acts):
		for vc,act in zip(vehicles,acts):
			current_road=self.vehicles[vc]["memory_0"][0][0:3]
			new_road=self.utils.move(current_road,act)
			road="road_"+str(new_road[0])+"_"+str(new_road[1])+"_"+str(new_road[2])
			
			if self.Log:	
				print("########## setting route for vc="+str(vc)+" act="+str(act)+"#########")
				print("current_road: "+str(current_road)+","+"new road: "+str(new_road))
				print(self.eng.get_vehicle_info(vc)["route"])
			
			routing_complete=self.eng.set_vehicle_route(vc,[road])
			
			if self.Log:
				print(routing_complete)
				print(self.eng.get_vehicle_info(vc)["route"])
			# assert(routing_complete)			
			# breakpoint()
			self.vehicles[vc]["memory_0"][1]=act
			self.vehicles[vc]["memory_0"][2]=routing_complete and self.utils.check_valid(new_road)
			