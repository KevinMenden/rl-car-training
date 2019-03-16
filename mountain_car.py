"""
Reinforcement Learning for the MountainCarContinuous-v0 environment
from OpenAI Gym
"""

import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from torch.autograd import Variable
from collections import namedtuple

class DQN(nn.Module):
    """
    Deep Q Network for approximation of the Q function
    """
    def __init__(self):
        super(DQN, self).__init__()
        self.n_input = 2
        self.n_actions = 1

        self.fc1 = nn.Linear(self.n_input, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def choose_action(self, state, epsilon=1):
        """
        Choose which action to take after given a state and epsilon
        :param state: current state
        :param epsilon:
        :return: action
        """
        if random.random() > epsilon:
            return torch.tensor([[random.random()]])
        else:
            return self.forward(state)


class ReplayMemory(object):
    """
    ReplayMemory object for storing states and actions
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        """
        Saves an experience, or just one timepoint
        :param experience:
        """
        # if memory full, start filling up from oldest memory
        if self.position == self.capacity - 1:
            self.position = 0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position += 1

    def sample(self, batch_size):
        """
        Take a random batch of samples from the memory
        :param batch_size:
        :return: returns a batch of memories
        """
        batch = random.sample(self.memory, batch_size)
        return batch

    def __len__(self):
        return len(self.memory)


def optimize_models(replay_memory, policy, target, batch_size):
    """
    Sample a batch and optimize
    :param experience:
    :param policy:
    :param target:
    :return:
    """
    if len(batch_size) > len(replay_memory):
        batch_size = len(replay_memory)

    # Sample from the memory
    memories = replay_memory.sample(batch_size)

    # Transpose the batch (https://stackoverflow.com/a/19343/3343043)
    batch = Experience(*zip(*memories))

    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    # Compute the Q values
    state_action_values = policy(state_batch)

    # Compute the V values
    next_state_values = target(next_state_batch)
    # Compute expected Q values
    expected_q_vals = next_state_values * gamma + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_q_vals.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # clamp the parameters
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Named tuple to store experiences in
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Create policy and target networks
policy = DQN()
target = DQN()
target.load_state_dict(policy.state_dict())

optimizer = optim.RMSprop(policy.parameters())

# Create environment
env = gym.make("MountainCarContinuous-v0")

# Parameters
batch_size = 64
epsilon = 0.2
n_episodes = 50
gamma = 0.999
memory_size = 10000

# Create Memory
replay_memory = ReplayMemory(memory_size)


for ep in range(n_episodes):
    # Reset environment, get initial state
    env.reset()
    state, _, _, _ = env.step(env.action_space.sample())
    state = Variable(torch.from_numpy(state).type(torch.FloatTensor)).unsqueeze(0)


    for t in count():
        # Select and perform an action
        action = policy.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = next_state[0] + 0.5
        reward = torch.tensor([reward])
        next_state = Variable(torch.from_numpy(next_state).type(torch.FloatTensor)).unsqueeze(0)

        # Render only half the time for less overhead
        if t % 2 == 0:
            env.render()

        if done:
            next_state = None

        # Create experience and push to memory
        experience = Experience(state, action, next_state, reward)
        replay_memory.push(experience)

        state = next_state

        # Optimize the policy and target networks
        optimize_models(replay_memory, policy, target)

        if done:
            episode_rewards.append(reward)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
