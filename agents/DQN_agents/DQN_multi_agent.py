from collections import Counter

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
        self.agent_dic = self.create_agent_dic(input_dim=self.state_size)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"])

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.actions = self.pick_action()
            self.conduct_action(self.actions)

            steps=len(self.environment.trans_vehicles_states)

            if steps==0:
                continue
            
            if self.right_amount_of_steps_taken():
                for agent_id in self.agent_dic:
                    if self.enough_experiences_to_learn_from(agent_id):
                        for _ in range(self.hyperparameters["learning_iterations"]):
                            self.learn(agent_id)
            
            self.save_experience()

            # self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += steps
        self.episode_number += 1

    def pick_action(self, states=None):
        actions=[]
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if states is None: states = self.environment.trans_vehicles_states
            
        for state in states:
            if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
            state = state.float().unsqueeze(0).to(self.device)
            agent_id=self.get_agent_id(state)
            try:
                assert(agent_id in self.lanes_dic)
            except:
                breakpoint()

            if len(state.shape) < 2: state = state.unsqueeze(0)            
            q_network_local=self.agent_dic[agent_id]["NN"]
            q_network_local.eval() #puts network in evaluation mode
            with torch.no_grad():
                action_values = q_network_local(state)
            q_network_local.train() #puts network back in training mode
            
            # TODO
            action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
            self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
            actions.append(action)   
        return actions

    def learn(self,agent_id):
        """Runs a learning iteration for the Q network on each agent"""
        memory=self.agent_dic[agent_id]["memory"]
        states, actions, rewards, next_states, dones = self.sample_experiences(memory) #Sample experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions ]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(agent_id, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        # divide the next-states into batches with different agent_id
        # compute the q-target-next
        # return results in order
         # or
        #I can forget about batches and performance, just compute q-targets one by one
        # TODO: change for batches

        for state in next_states:
            agent_id=self.get_agent_id(state)
            Q_targets_next = self.agent_dic[agent_id](state)["NN"].detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        # TODO: why (1-dones)?
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        agent_id=self.get_agent_id(state)
        Q_expected = self.agent_dic[agent_id](state)["NN"].gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self,agent_id):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from(agent_id)

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self,memory):
        """Draws a random sample of experience from the memory buffer"""
        experiences = memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones

