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

class Adaptive_Routing_Environment(gym.Env):
	environment_name = "Adaptive Routing"

	def __init__(self,dim,encode,Num_Flows,skip_routing,Max_Sim_Time,device,Log):
		self.eng = cityflow.Engine("environments/3x3/config.json", thread_num=8)
		self.vehicles={}
		self.trans_vehicles=[]
		self.trans_vehicles_states=[]
		self.states=[]
		self.acts=[]
		self.next_states=[]
		self.rewds=[]
		self.dones=[]
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

		self.lanes_dic={}
		# TODO
		
		self.utils=Utils(dim=dim,encode=encode,Num_Flows=Num_Flows,valid_set=self.valid_set)
		self.stochastic_actions_probability = 0
		self.actions = set(range(3))
		self.id = "Adaptive Routing"
		self.action_space = spaces.Discrete(3)
		self.state_size=self.utils.get_state_diminsion()
		self.seed()
		self.reward_threshold = 0.0
		self.trials = 5
		self.Log=Log
		self.skip_routing=skip_routing
		self.manual_drive_route=[0,2,2,0,1,1]
		self.iteration=-1
		self.manual_drive=False

	def reset(self,episode):
		self.eng.reset()
		# breakpoint()
		self.eng.set_save_replay(True)
		self.eng.set_replay_file("replays/episode"+str(episode)+".txt")
		self.refresh()
		self.eng.next_step()
		return []
		# return self.state
		# assuming that there is no TVs at the beginning
	
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
		

	def step(self,actions):
		if self.Log:
			print("Simulation Time: :{:.2f}".format(self.eng.get_current_time()))
		self.refresh_exp()

		if self.manual_drive:
			for i in range(len(actions)):
				actions[i]=self.manual_drive_route[self.iteration]
			

		for vc,act in zip(self.trans_vehicles,actions):
			self.set_route(vc,act)
			if self.vehicles[vc]["memory2"]!=None:
				done,reward=self.get_reward(vc)
				self.save_expirence(vc,reward,done)
			
		
		self.eng.next_step()
		self.refresh_trans()

		if self.is_terminal():
			return self.states,self.acts,self.next_states,self.rewds,self.dones,True

		
		eng_vehicles_dic=self.get_engine_vehicles_dic()
		self.update_env_vehicles(eng_vehicles_dic)
		
		for vc in eng_vehicles_dic:
			if self.is_autonomus_off(vc):
				continue
			
			if self.is_trans(vc):
				if self.manual_drive:
					self.iteration+=1

				self.transit_vc(vc)	
	
		return self.states,self.acts,self.next_states,self.rewds,self.dones,False

	def set_route(self,vc,act):
		current_road=self.vehicles[vc]["memory0"]["road"]
		next_road=self.get_next_road(vc,act)
		self.vehicles[vc]["memory0"]["action"]=act
		# self.vehicles[vc]["memory0"]["next_road"]=next_road

		road="road_"+str(next_road[0])+"_"+str(next_road[1])+"_"+str(next_road[2])
		
		if self.Log:	
			print("########## setting route for vc="+str(vc)+" act="+str(act)+"#########")
			print("current_road: "+str(current_road)+","+"new road: "+str(next_road))
			print(self.eng.get_vehicle_info(vc)["route"])
		
		routing_complete=self.eng.set_vehicle_route(vc,[road])
		
		if self.Log:
			print(routing_complete)
			print(self.eng.get_vehicle_info(vc)["route"])

		
		self.vehicles[vc]["memory0"]["valid"]=routing_complete and self.utils.check_valid(next_road)


	def save_expirence(self,vc,reward,done):
		if self.Log:
			print(self.vehicles[vc]["memory2"]["road"],
				self.vehicles[vc]["memory2"]["action"],
				self.vehicles[vc]["memory1"]["road"],
				reward)
		self.states.append(self.utils.reshape(self.get_state(vc,"memory2"),vc))
		self.acts.append(self.vehicles[vc]["memory2"]["action"])
		# bug here
		self.next_states.append(self.utils.reshape(self.get_state(vc,"memory1"),vc))
		self.rewds.append(reward)
		self.dones.append(done)


	def update_env_vehicles(self,eng_vehicles_dic):
		removable=[]
		for vc in self.vehicles:
			if not vc in eng_vehicles_dic:
				removable.append(vc)
		for vc in removable:
			if self.Log:
				print("vehicle "+vc+" exited the simulation")	
			self.vehicles.pop(vc)

	def transit_vc(self,vc):
		
		if self.is_new(vc):
			if self.Log:
				print("vehicle "+vc+" entered simulation")
			
			self.vehicles[vc]={
			# "previous_road": None,
			"destination": self.get_destination(vc),
			"enter time": self.eng.get_current_time(),
			"memory0":{
				"road":self.get_road(vc),
				"action":None,
				"reward":None,
				"valid":True
			},
			"memory1":None,
			"memory2":None
			}
		else:
			if self.Log:
					print("vehicle "+vc+" changed its line")
			# self.vehicles[vc]["previous_road"]=self.vehicles[vc]["road"]
			try:
				self.vehicles[vc]["memory2"]=None if self.vehicles[vc]["memory1"]==None else self.vehicles[vc]["memory1"].copy()
				self.vehicles[vc]["memory1"]=self.vehicles[vc]["memory0"].copy()
				self.vehicles[vc]["memory0"]["road"]=self.get_road(vc)
			except Exception as e:
				breakpoint()


		self.trans_vehicles.append(vc)
		state=self.state2torch(self.get_state(vc,"memory0"),vc)
		self.trans_vehicles_states.append(state)


	def state2torch(self,state,vc):
		state=torch.tensor(self.utils.reshape(state,vc), device=self.device, dtype=torch.float)
		return state

	def is_route_finished(self):
		pass
	
	def get_reward(self,vc):
		road=self.vehicles[vc]["memory2"]["road"]
		next_road=self.vehicles[vc]["memory1"]["road"]
		next_next_road=self.vehicles[vc]["memory0"]["road"]
		roadD=self.vehicles[vc]["destination"]		
		
		if not self.vehicles[vc]["memory2"]["valid"]:
			# breakpoint()
			if next_road==roadD:
				TT=self.eng.get_current_time()-self.vehicles[vc]["enter time"]
				SPTT=self.utils.get_Shoretest_Path_Travel_Time(vc)
				reward=1+math.exp(SPTT/TT)
				if self.Log:
					print("goal reached: {:.2f}".format(reward))
			else:
				reward=-2
				if self.Log:
					print("dead-end")
			
			return True,reward

		intersec1=next_road[0:2]
		intersec2=next_next_road[0:2]
		intersecD=roadD[0:2]

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


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def is_rewardable(self,vc):
		
		return vc in self.vehicles and "action" in self.vehicles[vc]
	
	def is_terminal(self):
		
		return self.eng.get_vehicle_count==0 or self.eng.get_current_time()==self.Max_Sim_Time

	def is_autonomus_off(self,vc):
		
		return self.utils.get_flow_id(vc) in self.skip_routing
	
	def is_new(self,vc):
		
		return vc not in self.vehicles

	def is_trans(self,vc):
		return self.is_new(vc) or self.get_road(vc) != self.vehicles[vc]["memory0"]["road"]

	def get_engine_vehicles_dic(self):
		
		return self.eng.get_vehicle_speed()

	def get_road(self,vc):
		derivable=list(map(int,self.eng.get_vehicle_info(vc)["drivable"].split('_')[1:5]))
		road=self.utils.derivable2road(derivable)
		return road
	
	def get_next_road(self,vc,action):
		derivable=self.get_road(vc)+[action]
		road=self.utils.derivable2road(derivable)
		return road

	def get_destination(self,vc):
		path=self.eng.get_vehicle_info(vc)["route"].split()
		destination=path[len(path)-1]
		destination=list(map(int,destination.split('_')[1:4]))
		return destination

	def get_state(self,vc,memory):
		source=self.vehicles[vc][memory]["road"]
		destination=self.vehicles[vc]["destination"]
		state=source+destination
		return state

	def get_next_state(self,vc):
		source=self.vehicles[vc]["next_road"]
		destination=self.vehicles[vc]["destination"]
		state=source+destination
		return state