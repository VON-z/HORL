#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DQN.py
@Time    :   2024/10/02 19:23:10
@Author  :   Q
@Version :   1.0
@Contact :   1036991178@qq.com
@Desc    :   DQN
'''

# here put the standard library
import collections
import random

# here put the third-party packages
import numpy as np
import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

# here put the local import source
import rl_utils

class ReplayBuffer:
    """经验回放池
    
    """
    def __init__(self, capacity) -> None:
        """初始化
        
        Args:
            capacity (int): buffer容量
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
    

class QNet(torch.nn.Module):
    """_summary_
    
    Args:
        torch (_type_): _description_
    """
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim,
                learning_rate, gamma, epsilon, target_update, device) -> None:
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, learning_rate=lr,
            gamma=gamma, epsilon=epsilon, target_update=target_update, device=device)

return_list = []

for i in range(10):
    with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns,
                        'dones': b_d,
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if(i_episode+1)%10 == 0:
                pbar.set_postfix({
                    'episode': f'{(num_episodes/10*i + i_episode+1)}',
                    'return': f'{np.mean(return_list[-10:]):.2f}'
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'DQN on {env_name}')
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'DQN on {env_name}')
plt.show()
