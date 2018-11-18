
import gym
from DDPGController import MADDPG
from ExperienceReplay import ExperienceReplay, Transition
from OUNoise import OUNoise
import torch
import numpy as np



def make_env(scenario_name, benchmark=False):
	'''
	Creates a MultiAgentEnv object as env. This can be used similar to a gym
	environment by calling env.reset() and env.step().
	Use env.render() to view the environment on the screen.
	Input:
		scenario_name   :   name of the scenario from ./scenarios/ to be Returns
							(without the .py extension)
		benchmark       :   whether you want to produce benchmarking data
							(usually only done during evaluation)
	Some useful env properties (see environment.py):
		.observation_space  :   Returns the observation space for each agent
		.action_space       :   Returns the action space for each agent
		.n                  :   Returns the number of Agents
	'''
	from multiagent.environment import MultiAgentEnv
	import multiagent.scenarios as scenarios

	# load scenario from script
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	# create world
	world = scenario.make_world()
	# create multiagent environment
	if benchmark:        
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
	return env

env = make_env('simple_speaker_listener')
#env = gym.make('HalfCheetah-v2')
#env = gym.make('BipedalWalker-v2')

controller1 = MADDPG(3, 3, 5, agentType = 'speaker')
controller2 = MADDPG(11, 5, 3, agentType = 'listener')
expReplaySize = int(1e6)
experienceReplay = ExperienceReplay(expReplaySize)
controllers = [controller1,controller2]
#noise = OUNoise(4)
#Exp noise determine!

epsilon = 1.0
batchSize = 64
updateFrequencies = 32
actionCounter = 0
total = 0
for i_episode in range(1200):
	done = [False,False]
	observation = env.reset()
	total = 0
	counter = 0

	epsilon = 1.0-((i_episode+0.0)/700.0)*0.95
	while not done[0] and not done[1] and counter < 1000:
		counter += 1
		#env.render()
		action1 = controller1.select_action(torch.Tensor([observation[0]]), epsilon)
		action2 = controller2.select_action(torch.Tensor([observation[1]]), epsilon)
		newObservation, reward, done, info = env.step([action1[0].numpy(),action2[0].numpy()])
		actionCounter += 1
		total += reward[1]
		experienceReplay.addExperience([torch.Tensor([observation[0]]), torch.Tensor([observation[1]])], 
			[action1,action2], [torch.Tensor([int(not done)]),torch.Tensor([int(not done)])], 
			[torch.Tensor([newObservation[0]]),torch.Tensor([newObservation[1]])], [torch.Tensor([reward[0]]),
			torch.Tensor([reward[1]])])
		
		if experienceReplay.curSize >= batchSize :
			samples = experienceReplay.sample(batchSize)
			batch = Transition(*zip(*samples))
			valueLoss, policyLoss = controller1.update_parameters(batch,0,controllers)

			samples2 = experienceReplay.sample(batchSize)
			batch2 = Transition(*zip(*samples2))
			valueLoss2, policyLoss2 = controller2.update_parameters(batch2,1,controllers)

		if actionCounter % updateFrequencies == 0:
			controller1.hard_update()
			controller2.hard_update()

		observation = newObservation

	print('total rewards', total/counter)

