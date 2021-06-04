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

	def __init__(self,GAT,dim,encode,embed_press,Num_Flows,skip_routing,random_trips,Max_Sim_Time,device,Log,rolling_window):

		self.gat=GAT
		Adaptive_Routing_Environment.eng = cityflow.Engine("environments/3x3/config.json", thread_num=8)
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
		Adaptive_Routing_Environment.lanes_dic={
						110:3,111:3,
						210:3,211:3,212:3,
						311:3,312:3,
						120:3,121:3,123:3,
						220:3,221:3,222:3,223:3,
						321:3,322:3,323:3,
						130:3,133:3,
						230:3,232:3,233:3,
						332:3,333:3
		}
		self.dim=dim
		
		self.utils=Utils(dim=dim,environment=self,encode=encode,Num_Flows=Num_Flows,valid_dic=self.lanes_dic,device=device)


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
		self.manual_drive_route=[2,2,2,0,1,1]
		self.iteration=-1
		self.manual_drive=False
		self.random_trips=random_trips
		self.embed_press=embed_press


	def reset(self,episode):
		if self.random_trips:
			self.utils.generate_random_trips()
			self.eng = cityflow.Engine("environments/3x3/config.json", thread_num=8)

		
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

		if self.manual_drive:
			for i in range(len(actions)):
				actions[i]=self.manual_drive_route[self.iteration]
		
		self.set_route(self.trans_vehicles,actions)	
		# ---------------------------------------------

		self.eng.next_step()
		if self.is_terminal():
			return self.states,self.acts,self.next_states,self.rewds,self.dones,True

		self.utils.update_pressure_matrix()
		edge_index=self.utils.get_edge_index()
		node_features=self.utils.get_node_features()
		evolved_node_features = self.gat((node_features, edge_index))[0]
		breakpoint()

		self.refresh_trans()	
		self.refresh_exp()
		eng_vehicles_dic=self.get_engine_vehicles_dic()
		self.update_env_vehicles(eng_vehicles_dic)
		# ---------------------------------------------

		for vc in eng_vehicles_dic:
			if self.is_autonomus_off(vc) or not self.is_trans(vc):
				continue

			self.transit_env_vc(vc)

			if self.vehicles[vc]["memory0"]["valid"]:
				self.add_to_next_trans_for_routing(vc)

			if self.vehicles[vc]["memory2"]!=None:
				done,reward=self.get_reward(vc)
				self.save_expirence(vc,reward,done)	

			if self.manual_drive:
				self.iteration+=1	


		return self.states,self.acts,self.next_states,self.rewds,self.dones,False

	def set_route(self,VCs,ACTs):
		for vc,act in zip(VCs,ACTs):
			current_road=self.vehicles[vc]["memory0"]["road"]
			next_road=self.utils.get_next_road(vc,act)
			road="road_"+str(next_road[0])+"_"+str(next_road[1])+"_"+str(next_road[2])
			routing_complete=self.eng.set_vehicle_route(vc,[road])
			self.vehicles[vc]["memory0"]["action"]=act
			self.vehicles[vc]["memory0"]["valid"]=routing_complete and self.utils.check_valid(next_road)

			if self.Log:	
				print("########## setting route for vc="+str(vc)+" act="+str(act)+"#########")
				print("current_road: "+str(current_road)+","+"new road: "+str(next_road))
				print("routing_complete:"+ str(routing_complete))
				print("next_road_valid:"+str(self.utils.check_valid(next_road)))
				print(self.eng.get_vehicle_info(vc)["route"])	

	def save_expirence(self,vc,reward,done):
		if self.Log:
			print(self.vehicles[vc]["memory2"]["road"],
				self.vehicles[vc]["memory2"]["action"],
				self.vehicles[vc]["memory1"]["road"],
				reward)
		
		state=self.vehicles[vc]["memory2"]["state"]
		action=self.vehicles[vc]["memory2"]["action"]
		next_state=self.vehicles[vc]["memory1"]["state"]

		road_id=self.utils.state2road(state)
		try:
			assert(road_id in self.lanes_dic)
		except:
			breakpoint()
		
		self.states.append(state)
		self.acts.append(action)
		self.next_states.append(next_state)
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

	def transit_env_vc(self,vc):
		
		if self.is_new(vc):
			if self.Log:
				print("vehicle "+vc+" entered simulation")
			destination=self.utils.get_destination(vc)
			source=self.utils.get_road(vc)
			self.vehicles[vc]={

			"destination": destination,
			"enter time": self.eng.get_current_time(),
			"memory0":{
				"road":source,
				"action":None,
				"reward":None,
				"state":self.utils.get_state(source,destination,self.embed_press,vehicle_id=vc),
				"valid": self.utils.check_valid(source)
			},
			"memory1":None,
			"memory2":None
			}
		else:
			if self.Log:
				print("vehicle "+vc+" changed its line")
				# breakpoint()
			# self.vehicles[vc]["previous_road"]=self.vehicles[vc]["road"]
			try:
				self.vehicles[vc]["memory2"]=None if self.vehicles[vc]["memory1"]==None else self.vehicles[vc]["memory1"].copy()
				self.vehicles[vc]["memory1"]=self.vehicles[vc]["memory0"].copy()
				source=self.utils.get_road(vc)
				destination=self.vehicles[vc]["destination"]
				self.vehicles[vc]["memory0"]["road"]=source
				self.vehicles[vc]["memory0"]["state"]=self.utils.get_state(source,destination,self.embed_press,vehicle_id=vc)
			except Exception as e:
				breakpoint()

	def add_to_next_trans_for_routing(self,vc):
		try:
			assert(self.utils.road2int(self.vehicles[vc]["memory0"]["road"]) in self.lanes_dic)
		except:
			breakpoint()
		self.trans_vehicles.append(vc)
		state=self.vehicles[vc]["memory0"]["state"]
		self.trans_vehicles_states.append(self.utils.state2torch(state))

	def get_reward(self,vc):
		road=self.vehicles[vc]["memory2"]["road"]
		next_road=self.vehicles[vc]["memory1"]["road"]
		next_next_road=self.vehicles[vc]["memory0"]["road"]
		roadD=self.vehicles[vc]["destination"]		
		
		if not self.vehicles[vc]["memory2"]["valid"]:
			# breakpoint()
			if next_road==roadD:
				# TT=self.eng.get_current_time()-self.vehicles[vc]["enter time"]
				# SPTT=self.utils.get_Shoretest_Path_Travel_Time(vc)
				# reward=1+math.exp(SPTT/TT)
				reward=0
				if self.Log:
					print("goal reached: {:.2f}".format(reward))
			else:
				reward=-10000
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
				reward=1
		else:
				reward=-1.5
		
		if self.Log:
				print("moving reward: "+ str(reward))

		return False,reward

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def is_terminal(self):
		
		return self.eng.get_vehicle_count==0 or self.eng.get_current_time()==self.Max_Sim_Time

	def is_autonomus_off(self,vc):
		
		return self.utils.get_flow_id(vc) in self.skip_routing
	
	def is_new(self,vc):
		
		return vc not in self.vehicles

	def is_trans(self,vc):
		return self.is_new(vc) or self.utils.get_road(vc) != self.vehicles[vc]["memory0"]["road"]

	def get_engine_vehicles_dic(self):
		
		return self.eng.get_vehicle_speed()

