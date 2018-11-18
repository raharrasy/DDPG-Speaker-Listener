import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import Actor, Critic
from torch.autograd import Variable
import numpy as np

import gym

class MADDPG(object):
	def __init__(self, inputSize , outputSpaceSize, opponentOutputSpaceSize, actorLearningRate=1e-4, criticLearningRate=1e-3,
		batchSize = 64, discountRate=0.99, numHiddenUnits = 25,
		polyakAveragingWeight = 1e-3, agentType = 'listener'):

		self.inputSize = inputSize
		self.outputSpaceSize = outputSpaceSize
		self.opponentOutputSpaceSize = opponentOutputSpaceSize

		self.actorLearningRate = actorLearningRate
		self.criticLearningRate = criticLearningRate

		self.batchSize = batchSize
		self.discountRate = discountRate
		self.numHiddenUnits = numHiddenUnits
		self.polyakAveragingWeight = polyakAveragingWeight
		self.agentType = agentType

		self.actor = Actor(self.inputSize, self.numHiddenUnits, self.outputSpaceSize, self.agentType)
		self.critic = Critic(self.inputSize, self.numHiddenUnits, self.outputSpaceSize, self.opponentOutputSpaceSize)
		self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actorLearningRate)
		self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.criticLearningRate)

		self.targetActor = Actor(self.inputSize, self.numHiddenUnits, self.outputSpaceSize, self.agentType)
		self.targetCritic = Critic(self.inputSize, self.numHiddenUnits, self.outputSpaceSize, self.opponentOutputSpaceSize)

		self.hard_update()

	def select_action(self, state, epsilon=0.05):
		with torch.no_grad():
			self.actor.eval()
			action = self.actor((Variable(state)),epsilon)
			self.actor.train()
		return action.data

	def soft_update(self):
		tau = self.polyakAveragingWeight
		for target_param, param in zip(self.targetActor.parameters(), self.actor.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

		for target_param, param in zip(self.targetCritic.parameters(), self.critic.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

	def hard_update(self):
		for target_param, param in zip(self.targetActor.parameters(), self.actor.parameters()):
			target_param.data.copy_(param.data)

		for target_param, param in zip(self.targetCritic.parameters(), self.critic.parameters()):
			target_param.data.copy_(param.data)


	def update_parameters(self, batch, agent_id, agentControllers):
		temp1, temp2, temp3, temp4= (), (), (), ()
		for k,l,m,n in zip(batch.state,batch.action, batch.reward, batch.mask):
			temp1 = temp1 + (k[agent_id],)
			temp2 = temp2 + (l[agent_id],)
			temp3 = temp3 + (m[agent_id],)
			temp4 = temp4 + (n[agent_id],)

		states = Variable(torch.cat(temp1))
		action = Variable(torch.cat(temp2))
		reward = Variable(torch.cat(temp3))
		mask = Variable(torch.cat(temp4))

		next_states_oppo = []
		for i in range(2):
			temp1 = ()
			for k in batch.next_state:
				temp1 = temp1 + (k[i],)
			next_states = Variable(torch.cat(temp1))
			next_states_oppo.append(next_states)



		#action = Variable(torch.cat(batch.action[:,agent_id]))
		#reward = Variable(torch.cat(batch.reward[:,agent_id]))
		#mask = Variable(torch.cat(batch.mask[:,agent_id]))

		with torch.no_grad():
			collectiveActions = None
			idx = 0
			for nextState, act in zip(next_states_oppo, agentControllers):
				nextAction = act.select_action(Variable(nextState), 0.0)
				if idx == 0:
					collectiveActions = nextAction
					idx += 1
				else:
					collectiveActions = torch.cat((collectiveActions,nextAction), dim=1)

		next_state_action_values = self.critic(Variable(next_states_oppo[agent_id]), collectiveActions)
		reward = reward.unsqueeze(1)
		mask = mask.unsqueeze(1)
		expected_state_value = reward + (self.discountRate * mask * next_state_action_values)

		self.critic_optim.zero_grad()

		i = 0
		allActions = None
		for a in range(len(batch.action)):
			combinedActions = None
			for i in range(2):
				if i == 0:
					combinedActions = batch.action[a][i]
				else :
					combinedActions = torch.cat((combinedActions,batch.action[a][i]), dim=1)

			if a == 0:
				allActions = combinedActions
			else:
				allActions = torch.cat((allActions,combinedActions), dim=0)


		state_value = self.critic((states), (allActions))

		value_loss = F.mse_loss(state_value, expected_state_value)
		value_loss.backward()
		self.critic_optim.step()

		self.critic_optim.zero_grad()
		self.actor_optim.zero_grad()

		replacementActions = self.actor((states),0)
		
		if agent_id == 0:
			allActions[:,0:3] = replacementActions
		else:
			allActions[:,3:] = replacementActions

		policy_loss = -self.critic((states),allActions)

		policy_loss = policy_loss.mean()
		policy_loss.backward()
		self.actor_optim.step()

			#self.soft_update()

		return value_loss.item(), policy_loss.item()


