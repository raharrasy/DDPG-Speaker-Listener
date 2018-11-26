import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import math


class Actor(nn.Module):
	def __init__(self, inputSize, hiddenLayerSize, actionSpace, agentType = 'listener'):

		super(Actor, self).__init__()
		self. inputSize = inputSize
		self.actionSpaceSize = actionSpace
		self.agentType = agentType

		self.linear1 = nn.Linear(inputSize, hiddenLayerSize)
		self.layerNorm = nn.LayerNorm(hiddenLayerSize)


		self.linear1.weight = torch.nn.Parameter(6e-3*2.0/math.sqrt(inputSize) * torch.rand(self.linear1.weight.shape[0], self.linear1.weight.shape[1]) - 1.0/math.sqrt(inputSize))
		self.linear1.bias = torch.nn.Parameter(6e-3*2.0/math.sqrt(inputSize) * torch.rand(self.linear1.bias.shape[0]) - 1.0/math.sqrt(inputSize))

		self.linear2 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
		self.layerNorm2 = nn.LayerNorm(hiddenLayerSize)

		self.linear2.weight = torch.nn.Parameter(6e-3 * 2.0/math.sqrt(inputSize) * torch.rand(self.linear2.weight.shape[0], 
			self.linear2.weight.shape[1]) - 1.0/math.sqrt(inputSize))
		self.linear2.bias = torch.nn.Parameter(6e-3 * 2.0/math.sqrt(inputSize) * torch.rand(self.linear2.bias.shape[0]) - 1.0/math.sqrt(inputSize))

		self.acts = nn.Linear(hiddenLayerSize, self.actionSpaceSize)
		self.acts.weight = torch.nn.Parameter(6e-4 * torch.rand(self.acts.weight.shape[0], 
			self.acts.weight.shape[1]) - 3e-4)
		self.acts.bias = torch.nn.Parameter(6e-4 * torch.rand(self.acts.bias.shape[0]) - 3e-4)

	def forward(self, inputs, hard=False, epsilon=0.0) :
		out = self.linear1(inputs)
		out = self.layerNorm(out)
		out = F.relu(out)
		out = self.linear2(out)
		out = self.layerNorm2(out)
		out = F.relu(out)
		out = self.acts(out)
		out = GumbelSoftmax(temperature=1.0, hard=hard, epsilon=epsilon).sample(out)
		return out


class Critic(nn.Module):
	def __init__(self, inputSize, hiddenLayerSize, agentActionSpace, oppoActionSpace):

		super(Critic, self).__init__()
		self. inputSize = inputSize

		self.agentActionSize = agentActionSpace
		self.oppoActionSize = oppoActionSpace
		self.linear1 = nn.Linear(self.inputSize+self.agentActionSize+self.oppoActionSize, hiddenLayerSize)
		self.layerNorm = nn.LayerNorm(hiddenLayerSize)

		self.linear1.weight = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear1.weight.shape[0], 
			self.linear1.weight.shape[1]) - 1.0/math.sqrt(inputSize))
		self.linear1.bias = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear1.bias.shape[0]) - 1.0/math.sqrt(inputSize))

		self.linear2 = nn.Linear(hiddenLayerSize, hiddenLayerSize)
		self.layerNorm2 = nn.LayerNorm(hiddenLayerSize)

		self.linear2.weight = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear2.weight.shape[0], 
			self.linear2.weight.shape[1]) - 1.0/math.sqrt(inputSize))
		self.linear2.bias = torch.nn.Parameter(2.0/math.sqrt(inputSize) * torch.rand(self.linear2.bias.shape[0]) - 1.0/math.sqrt(inputSize))

		self.acts = nn.Linear(hiddenLayerSize, 1)
		self.acts.weight = torch.nn.Parameter(6e-3 * torch.rand(self.acts.weight.shape[0], 
			self.acts.weight.shape[1]) - 3e-3)
		self.acts.bias = torch.nn.Parameter(6e-3 * torch.rand(self.acts.bias.shape[0]) - 3e-3)

	def forward(self, inputs,agentActions) :
		out = self.linear1(torch.cat((inputs,agentActions),1))
		out = self.layerNorm(out)
		out = F.relu(out)
		out = self.linear2(out)
		out = self.layerNorm2(out)
		out = F.relu(out)
		out = self.acts(out)

		return out


class GumbelSoftmax(object):
	def __init__(self, temperature=1.0, epsilon = 0.05, hard=False):
		self.temperature = temperature
		self.hard = hard
		self.epsilon = epsilon

	def sample(self,logits):

		gumbelLogits = logits + self.gumbelDistSample(logits)
		gumbelSoftmax = F.softmax(gumbelLogits/self.temperature, dim=1)

		if self.hard:
			gumbelHardSamples = self.onehot_from_logits(gumbelSoftmax)
			rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
				range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)

			# chooses between best and random actions using epsilon greedy
			samples = torch.stack([gumbelHardSamples[i] if r > self.epsilon else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])
			output = (samples-gumbelSoftmax).detach() + gumbelSoftmax
			return output
			
		return gumbelSoftmax


	def onehot_from_logits(self, logits):
		argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
		return argmax_acs
    
	def gumbelDistSample(self, logits, epsilon = 1e-20, dtype = torch.FloatTensor):
		sample = Variable(torch.rand(*logits.shape).type(logits.data.dtype), requires_grad = False)
		return -torch.log(-torch.log(sample + epsilon)+epsilon)








