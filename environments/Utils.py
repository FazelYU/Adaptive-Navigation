import numpy

class Utils(object):
	"""docstring for Utils"""
	def __init__(self,encode,dim,Num_Flows):
		super(Utils, self).__init__()
		self.valid_set=[110,111,112,113,
						210,211,212,213,
						310,311,312,313,
						120,121,122,123,
						220,221,222,223,
						320,321,322,323,
						130,131,132,133,
						230,231,232,233,
						330,331,332,333]
		self.dim=dim
		self.encode=encode
		self.Num_Flows=Num_Flows

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
		road2valid=self.move(road,1)
		road2valid=road2valid[0]*100+road2valid[1]*10+road2valid[2]		
		valid=road2valid in self.valid_set
		return valid

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
		if self.Num_Flows!=0:
			one_hot_vid=[-100]*self.Num_Flows
			one_hot_vid[self.get_flow_id(vehicle_id)]=100

		road1=self.res_road(state[:3])
		road2=self.res_road(state[3:])
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
		