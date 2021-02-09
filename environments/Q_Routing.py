import time
import gym
import math
import random
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import Adaptive_Routing_Environment as cf
import Utils
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# is_ipython = 'inline' in matplotlib.get_backend() # this is false for now
# if is_ipython:
#     from IPython import display

# plt.ion() #pyplot interactive on

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device is cpu

# Replay Memory
# We’ll be using experience replay memory for training our DQN. 
# It stores the transitions that the agent observes, allowing us to reuse this data later. 
# By sampling from it randomly, the transitions that build up a batch are decorrelated. 
# It has been shown that this greatly stabilizes and improves the DQN training procedure.

Transition = namedtuple('Transition',
						('state', 'action','next_state', 'reward'))

class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		# breakpoint()
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		transitions=random.sample(self.memory, batch_size)
		batch = Transition(*zip(*transitions))
		state_batch = torch.cat(batch.state)
		next_state_batch=torch.cat(batch.next_state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		return state_batch,next_state_batch,action_batch,reward_batch

	def dump(self):

		batch = Transition(*zip(*self.memory))
		state_batch = torch.cat(batch.state)
		next_state_batch=torch.cat(batch.next_state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		return [state_batch,next_state_batch,action_batch,reward_batch]

	def __len__(self):
		return len(self.memory)

# class ReplayMemory(object):

	# def __init__(self, capacity):
	# 	self.capacity = capacity
	# 	self.memory = []
	# 	self.position = 0

	# def push(self, *args):
	# 	"""Saves a transition."""
	# 	if len(self.memory) < self.capacity:
	# 		self.memory.append(None)

	# 	# breakpoint()
	# 	self.memory[self.position] = Transition(*args)
	# 	self.position = (self.position + 1) % self.capacity

	# def sample(self, batch_size):
	# 	return random.sample(self.memory, batch_size)

	# def __len__(self):
	# 	return len(self.memory)

class DQN(nn.Module):

	def __init__(self, state_size, outputs):
		super(DQN, self).__init__()
		self.fc1 = nn.Linear(state_size, 6)
		self.fc2 = nn.Linear(6, 6)
		self.fc3 = nn.Linear(6, outputs)
		self.softmax=nn.Softmax(dim=-1)


	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		# breakpoint()
		x = self.softmax(x)
		return x

# global coutner
# coutner=0
# actions=[[torch.tensor(0)],
# 		[torch.tensor(1)],
# 		[torch.tensor(2)],
# 		[torch.tensor(1)],
# 		[torch.tensor(1)],
# 		[torch.tensor(2)],
# 		[torch.tensor(1)],
# 		[torch.tensor(2)],
# 		[torch.tensor(0)],
# 		[torch.tensor(2)],
# 		[torch.tensor(2)]]

# def select_action(TVs,env,steps_done):
# 	global coutner
# 	# assert(coutner<5)
# 	act= actions[coutner]
# 	coutner+=1
# 	return act
	


def select_action(states,steps_done):
	acts=[]

	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY) + Bias
	
	for state in states:

		if sample < eps_threshold: 
			# Explore
			act=torch.tensor([random.randrange(n_actions)], device=device, dtype=torch.long)
			acts.append(act)
		else:
			# Exploite
			with torch.no_grad():
            
            	# t.max(1) will return largest column value of each row.
            	# second column on max result is index of where max element was
           		# found, so we pick action with the larger expected reward.
				# act=policy_net(state).max(0)[1]
				act=policy_net(state).max(0)[1]

				# if act==2:
				# 	breakpoint()
				
				acts.append(act.reshape(1))		

	return eps_threshold,acts
	

def optimize_model():

	if len(memory)<BATCH_SIZE:
		return -999

	[state_batch,next_state_batch,action_batch,reward_batch]=memory.sample(BATCH_SIZE)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
 #    # for each batch state according to policy_net
	# state_action_values = policy_net(state_batch).gather(1, action_batch)
	state_action_values = policy_net(state_batch).gather(-1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
	# next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values = target_net(next_state_batch).max(1)[0].detach().view(len(next_state_batch),1)
    # Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
	# loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
	loss=MSE_loss(state_action_values,expected_state_action_values)
    # Optimize the model
	optimizer.zero_grad()
	loss.backward()
	# for param in policy_net.parameters():
	# 	param.grad.data.clamp_(-1, 1)
	# breakpoint()
	optimizer.step()
	# breakpoint()
	return loss.item()


Num_Episodes = 1000

# Exploration Hyper-Params
EPS_START = 0.999
EPS_END = 0
EPS_DECAY = Num_Episodes*0.4
Bias=-0.079

# Q Learning Hyper-Params
TARGET_UPDATE = Num_Episodes*0.01
GAMMA = 0
		# small gama prefers immidiate rewards

# NN Hyper-Params
n_actions = 3
state_size=6
policy_net = DQN(state_size, n_actions).to(device)
target_net = DQN(state_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
		# policy_net=policy_net.int() ????????????????????????????????????????


# Optimizer Hyper-Params
# Memory_Size=int(Num_Episodes*0.1)
Memory_Size=100
BATCH_SIZE = int(Memory_Size*0.1)
utils=Utils.Utils()	
MSE_loss=nn.MSELoss()
optimizer=optim.Adam(policy_net.parameters(), lr=0.01)
# optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(Memory_Size)

#Debugging
LOG=True
TIME=False
DEBUG=True

# def test(state):
# 	state_batch=torch.tensor(
# 		[state],
# 		device='cuda:0'
# 		)
# 	print(state_batch)
# 	print(policy_net(state_batch))
# 	print(target_net(state_batch))

def test(arg):
	state_batch=torch.tensor([
		[2., 1., 0., 3., 3., 0.],
		[3., 1., 0., 3., 3., 0.],
		[2., 2., 1., 3., 3., 0.],
		[3., 3., 1., 3., 3., 0.],
		[3., 2., 1., 3., 3., 0.],
		[3., 3., 1., 3., 3., 0.],
		[3., 3., 0., 3., 3., 0.]
		]
		,
		device=device
		)
	print(state_batch)
	print(policy_net(state_batch))
	print(target_net(state_batch))

def main():
	
	env = cf.Adaptive_Routing_Environment(600,device)
	env.eng.set_save_replay(True)
	for i_episode in range(Num_Episodes):
		env.reset()		
		total_reward=0
		acc_reward=[]
		acc_action=[]
		env.reset()
		env.eng.set_replay_file("replays/episode"+str(i_episode)+".txt")
		
		for t in count():


			if LOG and TIME:
				print("epsd"+str(i_episode)+" time"+ str(t))

			explore_rate,actions_todo=select_action(env.tvcs_states,i_episode)			
			states,acts,next_states,rews,done=env.step(actions_todo)
			assert(len(states)==len(acts)==len(rews))

			for st,act,nst,rew in zip(states,acts,next_states,rews):
				# breakpoint()
				st=utils.reshape(st)
				nst=utils.reshape(nst)
				# breakpoint()
				state= torch.tensor([st], device=device,dtype=torch.float)
					# state= torch.tensor([st], device=device)

				next_state= torch.tensor([nst], device=device,dtype=torch.float)
					# next_state= torch.tensor([nst], device=device)
				action = torch.tensor([[act]], device=device)
				reward = torch.tensor([[rew]], device=device,dtype=torch.float)
					# reward = torch.tensor([[rew]], device=device)
				memory.push(state, action,next_state, reward)
				# breakpoint()
				total_reward+=rew
				acc_reward.append(rew)
				acc_action.append(act.tolist())			
			
			if done:
				break

			if len(env.tvcs)==0:
				continue			
			
			

			if LOG and TIME:
				print("acc_action"+ str(acc_action) +" acc_reward: "+ str(acc_reward))
				# TODO: warning in the next line
			
			# steps_done+=len(TVs)


		loss=optimize_model()

		if LOG:
			print('Completed Episode' + str(i_episode)+ \
				" memory lenght: "+str(memory.__len__())+ \
				" exploration rate: {:.2f}".format(round(explore_rate,2))+ \
				" acc_action"+ str(acc_action)+ \
				" acc_reward: "+ str(acc_reward)+ \
				" total reward: "+ str(total_reward)+ \
				" loss: {:.2f}".format(round(loss,2)))

		if i_episode % TARGET_UPDATE == 0:
			# breakpoint()
			target_net.load_state_dict(policy_net.state_dict())
			if LOG:
				print("Target net updated")



	if LOG:
		print('Simulation Completed')



if __name__=="__main__":
	main()



