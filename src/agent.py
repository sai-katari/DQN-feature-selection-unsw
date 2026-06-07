# agent.py - Double DQN with replay buffer and target network
# Uses online network to select actions, target network to evaluate them.
# This avoids the Q-value overestimation problem in standard DQN.

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buf)


class DoubleDQNAgent:
    def __init__(self, state_size, action_size, device=None):
        self.state_size  = state_size
        self.action_size = action_size
        self.device = device or torch.device("cpu")

        self.online_net = QNetwork(state_size, action_size, C.HIDDEN_SIZES).to(self.device)
        self.target_net = QNetwork(state_size, action_size, C.HIDDEN_SIZES).to(self.device)
        self._sync_target()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=C.LEARNING_RATE)
        self.buffer    = ReplayBuffer(C.REPLAY_BUFFER)

        self.epsilon      = C.EPS_START
        self._eps_step    = (C.EPS_START - C.EPS_END) / max(C.EPS_DECAY_EP, 1)
        self.episode_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.online_net(state_t)
        return int(q_vals.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < C.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(C.BATCH_SIZE)

        states_t      = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        q_vals  = self.online_net(states_t)
        q_taken = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # double DQN: online net picks action, target net scores it
        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)
            best_actions  = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states_t)
            next_q_chosen = next_q_target.gather(1, best_actions).squeeze(1)
            target_q      = rewards_t + C.GAMMA * next_q_chosen * (1.0 - dones_t)

        loss = F.mse_loss(q_taken, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def end_episode(self, episode_idx):
        self.episode_count += 1
        if self.epsilon > C.EPS_END:
            self.epsilon = max(C.EPS_END, self.epsilon - self._eps_step)
        if self.episode_count % C.TARGET_UPDATE_EP == 0:
            self._sync_target()

    def save(self, path):
        torch.save({
            "online_state_dict":    self.online_net.state_dict(),
            "target_state_dict":    self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon":              self.epsilon,
            "episode_count":        self.episode_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon       = ckpt["epsilon"]
        self.episode_count = ckpt["episode_count"]

    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())