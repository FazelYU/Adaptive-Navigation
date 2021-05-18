import numpy
import torch
import json
import random

class Utils(object):
	"""docstring for Utils"""
	def __init__(self,encode,dim,Num_Flows,valid_dic):
		super(Utils, self).__init__()
		self.valid_dic=valid_dic
		self.dim=dim
		self.encode=encode
		self.Num_Flows=Num_Flows
		self.destination_list=[
								"road_3_1_0",
								"road_3_2_0",
								"road_3_3_0",
								"road_1_1_2",
								"road_1_2_2",
								"road_1_3_2",

								"road_1_3_1",
								"road_2_3_1",
								"road_3_3_1",
								"road_1_1_3",
								"road_2_1_3",
								"road_3_1_3",
							]
		self.source_list=[
							"road_0_1_0",
							"road_0_2_0",
							"road_0_3_0",

							"road_4_1_2",
							"road_4_2_2",
							"road_4_3_2",


							"road_1_0_1",
							"road_2_0_1",
							"road_3_0_1",

							"road_1_4_3",
							"road_2_4_3",
							"road_3_4_3",
		]

	def change_row(self,intersec,val):
		intersec[1]+=val
		return intersec

	def change_col(self,intersec,val):
		intersec[0]+=val
		return intersec


	def turn(self,dir,lane):
		if lane==0:
			dir+=1
		if lane==2:
			dir-=1
		return dir%4
		
	def change_position(self,intersec,dir):
		if dir==0:
			intersec=self.change_col(intersec,1)

		if dir==1:
			intersec=self.change_row(intersec,1)
		
		if dir==2:
			intersec=self.change_col(intersec,-1)
		
		if dir==3:
			intersec=self.change_row(intersec,-1)

		return intersec		
	
	def move(self,road,lane):
		intersec=road[0:2]
		dir=road[2]
		intersec=self.change_position(intersec,dir)
		dir=self.turn(dir,lane)
		# intersec=change_position(intersec,dir)
		new_road=intersec+[dir]
		return new_road

	def check_valid(self,road):
		# TODO: instead of checking a valid set chek an invalid setS
		# road2valid=self.move(road,1)
		road=road[0]*100+road[1]*10+road[2]		
		# valid=road in self.valid_dic
		# breakpoint()
		return road in self.valid_dic

	def derivable2road(self,derivable):		
		return self.move(derivable[0:3],derivable[3])

	def get_distance(self,inter1,inter2):
		return numpy.linalg.norm(numpy.array(inter1)-numpy.array(inter2))
		# return torch.norm(inter1-inter2)

	def res_road(self,road):
		
		if self.encode=="full_one_hot":
			diminsion_center=4*self.dim*self.dim
			diminsion_edge=4*self.dim
			one_hot_center=[0]*diminsion_center
			one_hot_edge=[0]*diminsion_edge
			[r1,r2,r3]=road
			dim=self.dim
			if r1==0:
				one_hot_edge[r2-1]=1
			elif r1==4:
				one_hot_edge[dim+r2-1]=1		
			elif r2==0:
				one_hot_edge[2*dim+r1-1]=1
			elif r2==4:
				one_hot_edge[3*dim+r1-1]=1
			else:
				one_hot_center[12*(r1-1)+4*(r2-1)+r3]=1
			return one_hot_center+one_hot_edge
		
		elif self.encode=="one_hot":
			one_hot_colum=[0]*(self.dim+2)
			one_hot_colum[road[0]]=1

			one_hot_row=[0]*(self.dim+2)
			one_hot_row[road[1]]=1

			one_hot_road=[0]*4
			one_hot_road[road[2]]=1

			return one_hot_colum+one_hot_row+one_hot_road

		else:
			intersec=road[0:2]
			dir=road[2]
			intersec=self.change_position(intersec,dir)
			return intersec+[dir]
			
	def reshape(self,state,vehicle_id):

		road1=self.res_road(state[:3])
		road2=self.res_road(state[3:])

		if self.Num_Flows==0:
			return road1+road2
		
		one_hot_vid=[-100]*self.Num_Flows
		one_hot_vid[self.get_flow_id(vehicle_id)]=100

		return road1+road2+one_hot_vid

	def get_state_diminsion(self):
		if self.encode=="full_one_hot":
			ret=((self.dim*self.dim+self.dim)*8)*2
		
		elif self.encode=="one_hot":
			ret=(2*(self.dim+2)+4)*2
		
		else:
			ret=self.dim
		
		return ret+self.Num_Flows

	def get_flow_id(self,flow):
		vid=list(map(int,flow.split('_')[1:3]))
		return vid[0]

	def get_Shoretest_Path_Travel_Time(self,flow):
		flow_id=self.get_flow_id(flow)
		if flow_id==0:
			return 165
		if flow_id==1:
			return 500

		return None
		
	def state2road(self,state):
		# breakpoint()
		# column=state[0][0]*0+state[0][1]*1+state[0][2]*2+state[0][3]*3+state[0][4]*4
		# row=state[0][5]*0+state[0][6]*1+state[0][7]*2+state[0][8]*3+state[0][9]*4
		# lane=state[0][10]*0+state[0][11]*1+state[0][12]*2
		try:
			if torch.is_tensor(state):
				state=state[0].cpu().detach().numpy().tolist()
			column=state[0]*0+state[1]*1+state[2]*2+state[3]*3+state[4]*4
			row=state[5]*0+state[6]*1+state[7]*2+state[8]*3+state[9]*4
			lane=state[10]*0+state[11]*1+state[12]*2+state[13]*3
		except:
			breakpoint()

		# if torch.is_tensor(state):
		# 	state=state[0].cpu().detach().numpy().tolist()
		# column=state[0]*0+state[1]*1+state[2]*2+state[3]*3+state[4]*4
		# row=state[5]*0+state[6]*1+state[7]*2+state[8]*3+state[9]*4
		# lane=state[10]*0+state[11]*1+state[12]*2
		# breakpoint()
		return int(column*100+row*10+lane)

	def road2int(self,road):
		return road[0]*100+road[1]*10+road[2]

	def generate_random_trips(self):
		data=[]
		with open("environments/3x3/flow.json","r") as read_file:
			data=json.load(read_file)
			# breakpoint()
			for vc in data:
				vc["route"][0]=random.choice(self.source_list)
				vc["route"][1]=random.choice(self.destination_list)
			# breakpoint()

		with open("environments/3x3/flow.json","w") as write_file:
			json.dump(data,write_file,sort_keys=True, indent=4, separators=(',', ': '))
		# breakpoint()