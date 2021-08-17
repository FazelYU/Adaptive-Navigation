from collections import Counter
import collections
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
# from utilities.data_structures.Replay_Buffer import Replay_Buffer
class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.agent_dic = self.create_agent_dic()
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

    def reset_game(self):
        super(DQN, self).reset_game()
        # self.update_learning_rate(self.hyperparameters["learning_rate"])

    def pick_action(self, states):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        if len(states)==0:
            return []
        
        if self.config.does_need_network_state:
            if self.config.does_need_network_state_embeding:
                self.config.GAT.eval()
                with torch.no_grad():
                    network_state_embeding=\
                    self.config.GAT(self.config.network_state.view(1,-1,4)).view(-1,4)
                self.config.GAT.train()
            else:
                network_state_embeding=self.config.network_state
        else:
            size=self.config.network_state.size()
            network_state_embeding=torch.empty(size[0],0).to(self.device)

        actions=[]

        for state in states:
            agent_id=self.get_agent_id(state)
            intersection_state_embeding=network_state_embeding[state['agent_idx']]
            destination_id=state['embeding'][0:self.intersection_id_size]
            destination_id_embeding=self.get_intersection_id_embedding(agent_id,destination_id,eval=True)            
            embeding=torch.cat((destination_id_embeding,intersection_state_embeding),0)
            action_values=self.get_action_values(agent_id,embeding.unsqueeze(0),eval=True)
            action_data={
                "action_values": action_values,
                "state": state,
                "turn_off_exploration": self.turn_off_exploration,
                "episode_number": self.env_episode_number}
            action = self.exploration_strategy.perturb_action_for_exploration_purposes(action_data)
                
            self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
            actions.append(action)   
        
        return actions
   
    def learn(self):
        """Runs a learning iteration for the Q network on each agent"""
        for _ in range(self.hyperparameters["learning_iterations"]):
            agents_losses=[self.compute_loss(agent_id) for agent_id in self.agent_dic if self.time_for_q_network_to_learn(agent_id)]
            try:
                self.take_optimisation_step(agents_losses, self.hyperparameters["gradient_clipping_norm"],retain_graph=True)            
            except Exception as e:
                breakpoint()

    def compute_loss(self, agent_id):
        """Computes the loss required to train the Q network"""
        memory=self.agent_dic[agent_id]["memory"]
        states, actions, rewards, next_states, dones = self.sample_experiences(memory) #Sample experiences

        with torch.no_grad():
            Q_values_next_states = self.compute_q_values_for_next_states(next_states,dones)
            Q_targets = rewards + (self.hyperparameters["discount_rate"] * Q_values_next_states * (1 - dones))
        
        Q_expected = self.compute_expected_q_values(agent_id, states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return (agent_id,loss)

    def compute_q_values_for_next_states(self, next_states,dones):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        batch_size=dones.size()[0]
        Q_targets_next=torch.zeros(batch_size,1).to(self.device)
        
        for state in next_states:
            # find a dummy embeding to replace for none states!
            if state!=None:
                dummy_embed=state['embeding']
        
        next_states_embedings=[state['embeding'] if state!=None else dummy_embed for state in next_states]
        # not_Non_next_states_batch_index_dic={id(not_Non_next_states[idx]):idx for idx in range(len(not_Non_next_states))}
        next_states_embedings_batch=torch.vstack(next_states_embedings)
        network_states_batch=next_states_embedings_batch[:,self.intersection_id_size:]

        if self.config.does_need_network_state:
            if self.config.does_need_network_state_embeding:
                network_state_embeding_batch=self.config.GAT(network_states_batch)
            else:
                network_state_embeding_batch=network_states_batch.view(batch_size,-1,4)
                # breakpoint()
        else:
            size=network_states_batch.view(batch_size,-1,4).size()
            network_state_embeding_batch=torch.empty(size[0],size[1],0).to(self.device)
            # breakpoint()

        
        masks_dic={}
        
        for i in range(0,batch_size):
            if dones[i]==1:
                continue
            agent_id=self.get_agent_id(next_states[i])
            if not agent_id in masks_dic:
                masks_dic[agent_id]={} 
                masks_dic[agent_id]["mask"]=[False]*batch_size
                masks_dic[agent_id]["batch_indexs"]=[]           
                masks_dic[agent_id]["network_index"]=next_states[i]['agent_idx']
            masks_dic[agent_id]["mask"][i]=True
            masks_dic[agent_id]["batch_indexs"].append(i)
                
        for agent_id in masks_dic:
            agent_mask=torch.Tensor(masks_dic[agent_id]["mask"]).unsqueeze(1).to(self.device,dtype=torch.bool)
            agent_states_action_mask=torch.vstack([agent_state['action_mask'] for agent_state in next_states[masks_dic[agent_id]["batch_indexs"]]])
            destination_ids=next_states_embedings_batch[masks_dic[agent_id]["batch_indexs"],0:self.intersection_id_size]
            destination_ids_embedings=self.get_intersection_id_embedding(agent_id,destination_ids)
            intersec_states_embeding=network_state_embeding_batch[masks_dic[agent_id]["batch_indexs"],masks_dic[agent_id]["network_index"]]
            agent_states_embedings=torch.cat((destination_ids_embedings,intersec_states_embeding),1)
            try:
                agent_Q_targets_next=(self.agent_dic[agent_id]["policy"](agent_states_embedings)+agent_states_action_mask).detach().max(1)[0].unsqueeze(1)
            except Exception as e:
                breakpoint()
            Q_targets_next.masked_scatter_(agent_mask,agent_Q_targets_next)
        
        return Q_targets_next

                                                                                                        # max(1): find the max in every row of the batch
                                                                                                        # max(0): find the max in every column of the batch
                                                                                                        # max(1)[0]: value of the max in every row of the batch
                                                                                                        # max(1)[1]: batch_indexs of the max in every row of the batch

    def compute_expected_q_values(self, agent_id, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        network_index=states[0]['agent_idx']
        states_batch=torch.vstack([state['embeding'] for state in states])
        network_states_batch=states_batch[:,self.intersection_id_size:]

        if self.config.does_need_network_state:
            if self.config.does_need_network_state_embeding:
                network_state_embeding_batch=self.config.GAT(network_states_batch)
            else:
                network_state_embeding_batch=network_states_batch.view(network_states_batch.size()[0],-1,4)
        else:
            size=network_states_batch.view(network_states_batch.size()[0],-1,4).size()
            network_state_embeding_batch=torch.empty(size[0],size[1],0).to(self.device)

        destination_ids=states_batch[:,0:self.intersection_id_size]
        destination_ids_embedings=self.get_intersection_id_embedding(agent_id,destination_ids)
        intersec_states_embeding=network_state_embeding_batch[:,network_index]
        states_embedings=torch.cat((destination_ids_embedings,intersec_states_embeding),1)
        try:
            Q_expected = self.agent_dic[agent_id]["policy"](states_embedings).gather(1, actions.long()) #must convert actions to long so can be used as batch_indexs
        except Exception as e:
            breakpoint()
        return Q_expected
