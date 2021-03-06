from torch.autograd import Variable
import gym
from DDPGController import MADDPG
from ExperienceReplay import ExperienceReplay, Transition
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

controller1 = MADDPG(3, 3, 5, 1.0,agentType = 'speaker')
controller2 = MADDPG(11, 5, 3, 1.0,agentType = 'listener')
expReplaySize = int(1e6)
experienceReplay = ExperienceReplay(expReplaySize)
controllers = [controller1,controller2]
#noise = OUNoise(4)
#Exp noise determine!

epsilon = 1.0
noise_factor = 5.0
batchSize = 1024
updateFrequencies = 32
saveFrequencies = 1000
actionCounter = 0
total = 0

for i_episode in range(10000):
	done = [False,False]
	observation = env.reset()
	total = 0
	counter = 0

	if i_episode % saveFrequencies ==0:
		env_name = 'speaker_listener'
		controller1.save_model(env_name,suffix='speaker_'+str(i_episode//1000))
		controller2.save_model(env_name,suffix='listener_'+str(i_episode//1000))

	epsilon = 1.0-min(1.0,(i_episode+0.0)/6000.0)*0.95
	#controller1.epsilon = epsilon
	#controller2.epsilon = epsilon

	while not done[0] and not done[1] and counter < 1000:
		counter += 1
		#env.render()
		action1 = controller1.select_action(torch.Tensor([observation[0]]),epsilon=0)
		action2 = controller2.select_action(torch.Tensor([observation[1]]),epsilon=0)
		action1 = (action1 == action1.max(1, keepdim=True)[0]).float()
		action2 = (action2 == action2.max(1, keepdim=True)[0]).float()
		rand_acs1 = Variable(torch.eye(action1.shape[1])[[np.random.choice(range(action1.shape[1]), size=action1.shape[0])]], requires_grad=False)
		rand_acs2 = Variable(torch.eye(action2.shape[1])[[np.random.choice(range(action2.shape[1]), size=action2.shape[0])]], requires_grad=False)
		samples1 = torch.stack([action1[i] if r > epsilon else rand_acs1[i] for i, r in enumerate(torch.rand(action1.shape[0]))])
		samples2 = torch.stack([action2[i] if r > epsilon else rand_acs2[i] for i, r in enumerate(torch.rand(action2.shape[0]))])
		newObservation, reward, done, info = env.step([samples1[0].numpy(),samples2[0].numpy()])
		
		actionCounter += 1
		total += reward[1]
		experienceReplay.addExperience([torch.Tensor([observation[0]]), torch.Tensor([observation[1]])], 
			[action1,action2], [torch.Tensor([int(not done)]),torch.Tensor([int(not done)])], 
			[torch.Tensor([newObservation[0]]),torch.Tensor([newObservation[1]])], [torch.Tensor([reward[0]]),
			torch.Tensor([reward[1]])])
		
		if experienceReplay.curSize >= batchSize :
			if actionCounter % 100 == 0:
				samples = experienceReplay.sample(batchSize)
				batch = Transition(*zip(*samples))
				valueLoss, policyLoss = controller1.update_parameters(batch,0,controllers)

				samples2 = experienceReplay.sample(batchSize)
				batch2 = Transition(*zip(*samples2))
				valueLoss2, policyLoss2 = controller2.update_parameters(batch2,1,controllers)

			controller1.soft_update()
			controller2.soft_update()

		observation = newObservation

	print('total rewards', total/counter)

