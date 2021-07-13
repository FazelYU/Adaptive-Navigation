from collections import Counter

import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
# from utilities.data_structures.Replay_Buffer import Replay_Buffer
# TODO:reduce state diminsion  -> optimize convergence
class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.agent_dic = self.create_agent_dic()
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"])

    def pick_action(self, states):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""

        actions=[]
        for state in states:
            # if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
            agent_id=self.get_agent_id(state)
            q_network_local=self.agent_dic[agent_id]["NN"]
            embeding=torch.tensor(state['embeding'],dtype=torch.float).unsqueeze(0).to(self.device)
            
            q_network_local.eval() #puts network in evaluation mode
            with torch.no_grad():
                action_values = q_network_local(embeding)
            q_network_local.train() #puts network back in training mode
            
            action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.agent_dic[agent_id]["episode_number"]})
            
            self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
            actions.append(action)   
        return actions

    def learn(self,agent_id):
        """Runs a learning iteration for the Q network on each agent"""
        memory=self.agent_dic[agent_id]["memory"]
        for _ in range(self.hyperparameters["learning_iterations"]):
            states, actions, rewards, next_states, dones = self.sample_experiences(memory) #Sample experiences
            loss = self.compute_loss(agent_id, states, next_states, rewards, actions, dones)
            # self.summ_writer.add_scalar('Loss/'+str(agent_id),loss,self.env_episode_number)
            
            # writer.add_scalar('Loss/train', np.random.random(), n_iter)
            actions_list = [action_X.item() for action_X in actions ]
            self.logger.info("Action counts {}".format(Counter(actions_list)))
            self.take_optimisation_step(agent_id, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, agent_id, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        
        Q_expected = self.compute_expected_q_values(agent_id, states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states,dones)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states,dones):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
                
        batch_size=dones.size()[0]
        Q_targets_next=torch.zeros(batch_size,1).to(self.device)
        
        masks_dic={}
        for i in range(0,batch_size):
            if dones[i]==1:
                continue
            agent_id=self.get_agent_id(next_states[i][0])
            # state=torch.unsqueeze(state,0)

            if not agent_id in masks_dic:
                masks_dic[agent_id]={} 
                masks_dic[agent_id]["mask"]=[False]*batch_size
                masks_dic[agent_id]["index"]=[]
            
            masks_dic[agent_id]["mask"][i]=True
            masks_dic[agent_id]["index"].append(i)
                

        for agent_id in masks_dic:
            agent_mask=torch.Tensor(masks_dic[agent_id]["mask"]).unsqueeze(1).to(self.device,dtype=torch.bool)
            agent_states=next_states[masks_dic[agent_id]["index"]]
            agent_states_embedings=[agent_state[0]['embeding'] for agent_state in agent_states]
            agent_states_embedings=torch.tensor(agent_states_embedings,dtype=torch.float).to(self.device)
            try:
                agent_Q_targets_next=self.agent_dic[agent_id]["NN"](agent_states_embedings).detach().max(1)[0].unsqueeze(1)
            except Exception as e:
                breakpoint()
            Q_targets_next.masked_scatter_(agent_mask,agent_Q_targets_next)

        return Q_targets_next

                                                                                                        # max(1): find the max in every row of the batch
                                                                                                        # max(0): find the max in every column of the batch
                                                                                                        # max(1)[0]: value of the max in every row of the batch
                                                                                                        # max(1)[1]: index of the max in every row of the batch
        
    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        # TODO: why (1-dones)?
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, agent_id, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        
        Q_expected = self.agent_dic[agent_id]["NN"](states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected
