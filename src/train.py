from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
import random


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  

class ProjectAgent:

    def act(self, obs, randomize=False):
        dev = torch.device('cpu')
        with torch.no_grad():
            q_values = self.model(torch.Tensor(obs).unsqueeze(0).to(dev))
            return torch.argmax(q_values).item()

    def save(self, filepath):
        self.filepath = filepath + "/model.pt"
        torch.save(self.model.state_dict(), self.filepath)
        return None

    def load(self):
        dev = torch.device('cpu')
        self.filepath = os.getcwd() + "/model.pt"
        self.model = self.build_model(dev)
        self.model.load_state_dict(torch.load(self.filepath, map_location=dev))
        self.model.eval()
        return None

    def get_configuration(self):
        params = {
            'num_actions': env.action_space.n,
            'lr': 0.001,
            'discount_factor': 0.98,
            'replay_capacity': 100000,
            'min_epsilon': 0.02,
            'max_epsilon': 1.0,
            'epsilon_decay_steps': 15000,
            'epsilon_wait': 100,
            'batch_sz': 600,
            'gradient_updates': 2,
            'target_update_method': 'replace',
            'target_update_interval': 400,
            'tau': 0.005,
            'loss_fn': torch.nn.SmoothL1Loss()}
        return params

    def build_model(self, device):
        input_dim, output_dim = env.observation_space.shape[0], env.action_space.n
        hidden_units = 256

        net = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim)
        ).to(device)

        return net

    def greedy_action(self, policy_net, state):
        dev = "cuda" if next(policy_net.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            q_vals = policy_net(torch.Tensor(state).unsqueeze(0).to(dev))
            return torch.argmax(q_vals).item()

    def apply_gradient(self):
        if len(self.memory) > self.batch_sz:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_sz)
            max_next_q = self.target_model(next_states).max(1)[0].detach()
            updated_vals = torch.addcmul(rewards, 1 - dones, max_next_q, value=self.discount_factor)
            current_q = self.model(states).gather(1, actions.to(torch.long).unsqueeze(1))
            loss = self.loss_fn(current_q, updated_vals.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

        config = self.get_configuration()

        self.num_actions = config['num_actions']
        self.discount_factor = config['discount_factor']
        self.batch_sz = config['batch_sz']
        self.replay_capacity = config['replay_capacity']

        self.epsilon = config['max_epsilon']
        self.min_epsilon = config['min_epsilon']
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / config['epsilon_decay_steps']
        self.epsilon_wait = config['epsilon_wait']

        self.loss_fn = config['loss_fn']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.gradient_updates = config['gradient_updates']

        self.target_update_method = config['target_update_method']
        self.target_update_interval = config['target_update_interval']
        self.tau = config['tau']

        self.memory = ReplayBuffer(self.replay_capacity, self.device)

        episode_rewards = []
        episode_num = 0
        cumulative_reward = 0
        obs, _ = env.reset()
        step_count = 0
        max_episodes = 200

        best_score = 0
        evaluation_freq = 1

        while episode_num < max_episodes:
            if step_count > self.epsilon_wait:
                self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

            if np.random.rand() < self.epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, obs)

            next_obs, reward, done, truncated, _ = env.step(action)
            self.memory.append(obs, action, reward, next_obs, done)
            cumulative_reward += reward

            for _ in range(self.gradient_updates):
                self.apply_gradient()

            if self.target_update_method == 'replace':
                if step_count % self.target_update_interval == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            elif self.target_update_method == 'ema':
                target_dict = self.target_model.state_dict()
                model_dict = self.model.state_dict()
                for key in model_dict:
                    target_dict[key] = self.tau * model_dict[key] + (1 - self.tau) * target_dict[key]
                self.target_model.load_state_dict(target_dict)

            step_count += 1

            if done or truncated:
                episode_num += 1

                if episode_num % evaluation_freq == 0:
                    score = evaluate_HIV(self, nb_episode=1)
                    if score > best_score:
                        print(f'Score: {score} > Best Score: {best_score}')
                        best_score = score
                        self.best_model = deepcopy(self.model).to(self.device)

                print(f"Episode {episode_num}, epsilon {self.epsilon:.2f}, memory size {len(self.memory)}, reward {cumulative_reward:.1f}")
                obs, _ = env.reset()
                episode_rewards.append(cumulative_reward)
                cumulative_reward = 0
            else:
                obs = next_obs

        self.model.load_state_dict(self.best_model.state_dict())
        self.save(os.getcwd())

        return episode_rewards


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0
        self.device = device

    def append(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train()
