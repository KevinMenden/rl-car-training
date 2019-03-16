import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# Create environment
env = gym.make("MountainCarContinuous-v0")
print(env.action_space)
print(env.observation_space)

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    screen = env.render(mode='rgb_array').transpose(2, 0, 1)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = resize(screen).unsqueeze(0)
    return screen


class DQN(nn.Module):
    """
    The Q approximating network
    Architecture from DQN paper
    """
    def __init__(self):
        super(DQN, self).__init__()

        self.n_actions = 1
        self.gamma = 0.9
        self.eps_final = 0.0001
        self.eps_start = 0.1
        self.n_iter = 2000000
        self.replay_memory_size = 10000
        self.batch_size = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, self.n_actions)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

net = DQN()
screen = get_screen()
env.close()

res = net(screen)