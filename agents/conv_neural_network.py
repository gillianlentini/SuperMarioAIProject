'''
   This code was written by following the following PyTorch tutorial
   Tutorial Title: TRAIN A MARIO-PLAYING RL AGENT
   Project Title: MadMario
   Author: Yuansong Feng, Suraj Subramanian, Howard Wang, Steven Guo
   Date: June 2020
   Code version: 2.0
   Availability: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html#train-a-mario-playing-rl-agent
                 https://github.com/YuansongFeng/MadMario
'''

import copy

import torch.nn as nn


class MarioNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)
