import numpy
import torch
import json
import random

class Utils(object):
	"""docstring for Utils"""
	def __init__(self,environment,encode,dim,Num_Flows,valid_dic,device):
		super(Utils, self).__init__()
		self.valid_dic=valid_dic
		self.dim=dim
		self.encode=encode
		self.Num_Flows=Num_Flows
		self.destination_list=[
								"road_3_1_0",
								"road_3_2_0",
								"road_3_3_0"
								# ,
								# "road_1_1_2",
								# "road_1_2_2",
								# "road_1_3_2",

								# "road_1_3_1",
								# "road_2_3_1",
								# "road_3_3_1",
								# "road_1_1_3",
								# "road_2_1_3",
								# "road_3_1_3",
							]
		self.source_list=[
							"road_0_1_0",
							"road_0_2_0",
							"road_0_3_0"

							# ,

							# "road_4_1_2",
							# "road_4_2_2",
							# "road_4_3_2",


							# "road_1_0_1",
							# "road_2_0_1",
							# "road_3_0_1",

							# "road_1_4_3",
							# "road_2_4_3",
							# "road_3_4_3",
		]
		self.pressure_matrix=[[0]*dim for i in range(dim)]
		self.dim=dim
		self.press_embd_dim=3
		self.environment=environment
		self.vc_count_dic=self.creat_vc_count_dic()
		self.Num_Flow_Types=3
		self.slow_vc_speed=7
		self.fast_vc_speed=2*self.slow_vc_speed
		self.device=device



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
			
	def get_state(self,origin,destination,embed_press,vehicle_id):
		reshaped_origin=self.res_road(origin)
		reshaped_destination=self.res_road(destination)
		if embed_press:
			press_embd=self.get_press_embd(origin)
		else:
			press_embd=[0]* self.press_embd_dim
		

		one_hot_flow_type=[0]*self.Num_Flow_Types
		flow_type=self.flow_types_dic[self.get_flow_id(vehicle_id)]
		one_hot_flow_type[flow_type]=1

		return reshaped_origin+reshaped_destination+press_embd+one_hot_flow_type

	def get_state_diminsion(self):
		if self.encode=="full_one_hot":
			ret=((self.dim*self.dim+self.dim)*8)*2
		
		elif self.encode=="one_hot":
			ret=(2*(self.dim+2)+4)*2
		
		else:
			ret=self.dim
		
		return ret+self.press_embd_dim+self.Num_Flow_Types

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
		self.flow_types_dic={}
		data=[]
		with open("environments/3x3/flow.json","r") as read_file:
			data=json.load(read_file)
			assert(len(data)!=0)
			while len(data)<self.Num_Flows:
				data.append(data[0].copy())
			while len(data)>self.Num_Flows:
				data.pop(len(data)-1)

			for flow_id in range(0,len(data)):
				
				vc=data[flow_id]
				vc["route"][0]=random.choice(self.source_list)
				vc["route"][1]=random.choice(self.destination_list)
				
				vc_max_speed=vc["vehicle"]["maxSpeed"]
				self.flow_types_dic[flow_id]= 2 if vc_max_speed>=self.fast_vc_speed else (0 if vc_max_speed<self.slow_vc_speed else 1)
			
			

		with open("environments/3x3/flow.json","w") as write_file:
			json.dump(data,write_file,sort_keys=True, indent=4, separators=(',', ': '))


	def get_press_embd(self,road):
		press_embd=[0]*self.press_embd_dim

		nroad=self.move(road,0)
		nroad=[x-1 for x in nroad]
		if nroad[0]<self.dim and nroad[1]<self.dim:
			press_embd[0]=self.pressure_matrix[nroad[0]][nroad[1]]

		nroad=self.move(road,1)
		nroad=[x-1 for x in nroad]
		if nroad[0]<self.dim and nroad[1]<self.dim:
			press_embd[0]=self.pressure_matrix[nroad[0]][nroad[1]]

		nroad=self.move(road,2)
		nroad=[x-1 for x in nroad]
		if nroad[0]<self.dim and nroad[1]<self.dim:
			press_embd[0]=self.pressure_matrix[nroad[0]][nroad[1]]


		return press_embd

	def update_pressure_matrix(self):
		self.update_vc_count_dic()
		for row in range(0,self.dim):
			for column in range(0,self.dim):
				try:
					self.pressure_matrix[column][row]=self.get_pressure(column,row)
				except Exception as e:
					print(e)
					breakpoint()

	def update_vc_count_dic(self):
		lane_vc_count_dic=self.environment.eng.get_lane_vehicle_count()
		self.refresh_vc_count_dic()
		for lane in lane_vc_count_dic:
			road= self.road2int(self.lane2road(lane))
			# if road==10:
			# 	breakpoint()
			self.vc_count_dic[road]+=lane_vc_count_dic[lane]

	def creat_vc_count_dic(self):
		lane_vc_count_dic=self.environment.eng.get_lane_vehicle_count()
		vc_count_dic={}
		for lane in lane_vc_count_dic:
			road= self.road2int(self.lane2road(lane))
			if not road in vc_count_dic: vc_count_dic[road]=0
		return vc_count_dic

	def refresh_vc_count_dic(self):
		for road in self.vc_count_dic:self.vc_count_dic[road]=0


	def lane2road(self,lane):
		road=list(map(int,lane.split('_')[1:4]))
		return road		

	def get_pressure(self,column,row):
		# column and rows are 1-indexed
		row+=1
		column+=1

		in_roads=[
			[column-1,row,0],
			[column+1,row,2],
			[column,row-1,1],
			[column,row+1,3]
		]
		out_roads=[
			[column,row,0],
			[column,row,1],
			[column,row,2],
			[column,row,3],
		]
		pressure=0

		for road in in_roads:
			pressure+=self.vc_count_dic[self.road2int(road)]

		for road in out_roads:
			pressure-=self.vc_count_dic[self.road2int(road)]

		return pressure

	def get_next_road(self,vc,action):
		derivable=self.get_road(vc)+[action]
		road=self.derivable2road(derivable)
		return road


	def get_road(self,vc):
		derivable=list(map(int,self.environment.eng.get_vehicle_info(vc)["drivable"].split('_')[1:5]))
		road=self.derivable2road(derivable)
		return road
	


	def get_destination(self,vc):
		path=self.environment.eng.get_vehicle_info(vc)["route"].split()
		destination=path[len(path)-1]
		destination=list(map(int,destination.split('_')[1:4]))
		return destination

	def state2torch(self,state):
		state=torch.tensor(state, device=self.device, dtype=torch.float)
		return state