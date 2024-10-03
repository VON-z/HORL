#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   REINFORCE.py
@Time    :   2024/10/03 17:57:59
@Author  :   Q
@Version :   1.0
@Contact :   1036991178@qq.com
@Desc    :   None
'''

# here put the standard library

# here put the third-party packages
import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# here put the local import source
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim,
                learning_rate, gamma, device) -> None:
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()

learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                learning_rate=learning_rate, gamma=gamma, device=device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'rewards': [],
                'next_states': [],
                'dones': [],
            }
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['rewards'].append(reward)
                transition_dict['next_states'].append(next_state)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)

            if (i_episode+1)%10 == 0:
                pbar.set_postfix({
                    'episode': f'{(num_episodes/10*i + i_episode+1)}',
                    'return': f'{np.mean(return_list[-10:]):.3f}'
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'REINFORCE on {env_name}')
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'REINFORCE on {env_name}')
plt.show()