import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import gym
import pygame
from pygame.locals import *
from gym import spaces

class SmartCityEnv(gym.Env):
    """Smart City Navigation Environment"""
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        super(SmartCityEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=20, shape=(6,), dtype=np.int32)
        
        # Road network configuration
        self.track = np.zeros((21, 21), dtype=int)
        self._build_roads()
        
        self.tower_points = {
            (10, 10): "A", (18, 18): "B", (18, 2): "C",
            (2, 18): "D", (2, 2): "E"
        }
        
        # Visualization setup
        self.tile_size = 32
        self.screen = None
        self.clock = None
        self.reset()

    def _build_roads(self):
        """Initialize road network"""
        self.track[0:3, :] = 1    # North horizontal
        self.track[9:12, :] = 1   # Central horizontal
        self.track[18:21, :] = 1  # South horizontal
        self.track[:, 0:3] = 1    # West vertical
        self.track[:, 9:12] = 1   # Central vertical
        self.track[:, 18:21] = 1  # East vertical

    def get_allowed_actions(self, position):
        """Get valid actions for current position"""
        r, c = position
        h_roads = {0,1,2,9,10,11,18,19,20}
        v_roads = {0,1,2,9,10,11,18,19,20}
        
        if r in h_roads and c in v_roads:
            return [0, 1, 2, 3]
        elif r in h_roads:
            return [2, 3]
        elif c in v_roads:
            return [0, 1]
        return []

    def reset(self):
        """Reset environment state"""
        valid_points = list(self.tower_points.keys())
        self.agent_pos = random.choice(valid_points)
        self.target = random.choice([p for p in valid_points if p != self.agent_pos])
        self.last_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target))
        return self._get_obs()

    def step(self, action):
        """Execute environment step"""
        r, c = self.agent_pos
        allowed = self.get_allowed_actions(self.agent_pos)
        reward = -0.05  # Time penalty
        
        # Action validation
        if action not in allowed:
            return self._get_obs(), -1.0, False, {}
        
        # Calculate movement
        dr = [-1, 1, 0, 0][action]
        dc = [0, 0, -1, 1][action]
        new_pos = (r + dr, c + dc)
        
        # Update position and rewards
        if self._is_valid(new_pos):
            self.agent_pos = new_pos
            reward += 0.2  # Movement bonus
            
        # Calculate distance-based reward
        current_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target))
        reward += (self.last_distance - current_distance) * 0.5
        self.last_distance = current_distance
        
        # Check termination
        done = self.agent_pos == self.target
        if done:
            reward += 20.0
            
        return self._get_obs(), reward, done, {}

    def _is_valid(self, pos):
        """Validate position"""
        return (0 <= pos[0] < 21) and (0 <= pos[1] < 21) and self.track[pos] == 1

    def _get_obs(self):
        """Enhanced state representation"""
        dx = self.target[0] - self.agent_pos[0]
        dy = self.target[1] - self.agent_pos[1]
        return np.array([*self.agent_pos, *self.target, dx, dy], dtype=np.int32)

    # Visualization methods remain unchanged

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""
    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize layer parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """Reset noise parameters"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Noise scaling function"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        """Forward pass with noise"""
        if self.training:
            return F.linear(x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)

class DuelingDQN(nn.Module):
    """Dueling Network Architecture"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.feature = nn.Sequential(
            NoisyLinear(input_size, 64),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            NoisyLinear(64, 64),
            nn.ReLU(),
            NoisyLinear(64, output_size)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(64, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)
        )

    def forward(self, x):
        """Forward pass"""
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def reset_noise(self):
        """Reset noise in all layers"""
        for layer in self.children():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

class EnhancedAgent:
    """Enhanced RL Agent with improved exploration"""
    def __init__(self, env, gamma=0.99, lr=0.001):
        self.env = env
        self.state_size = 6  # Updated state size
        self.action_size = 4
        self.gamma = gamma
        
        # Network setup
        self.model = DuelingDQN(self.state_size, self.action_size)
        self.target_model = DuelingDQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_network()
        
        # Experience replay
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        
        # Exploration parameters
        self.epsilon_start = 1.0  # Initial exploration rate
        self.epsilon_end = 0.1    # Minimum exploration rate
        self.epsilon_decay = 0.995 # Slower decay rate
        self.epsilon = self.epsilon_start

        # Adaptive exploration
        self.best_reward = -np.inf
        self.epsilon_reset_threshold = 0.5  # Reset epsilon if reward drops below this threshold

    def update_target_network(self):
        """Sync target network"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Improved action selection with action masking"""
        allowed_actions = self.env.get_allowed_actions((state[0], state[1]))
        
        if np.random.rand() <= self.epsilon:
            return random.choice(allowed_actions) if allowed_actions else 0
            
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.model(state_t).numpy()
            
        masked_q = np.full_like(q_values, -np.inf)
        masked_q[allowed_actions] = q_values[allowed_actions]
        return np.argmax(masked_q)

    def replay(self):
        """Experience replay with noise management"""
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])
        
        # Dueling DQN target calculation
        with torch.no_grad():
            target_q = self.target_model(next_states)
            max_target_q = target_q.max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * max_target_q
            
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(current_q.squeeze(), targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update exploration parameters
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self, episodes=500):
        """Training loop with adaptive exploration"""
        rewards_history = []
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                total_reward += reward
                
            rewards_history.append(total_reward)
            
            # Adaptive exploration reset
            if total_reward > self.best_reward:
                self.best_reward = total_reward
            elif total_reward < self.best_reward * self.epsilon_reset_threshold:
                self.epsilon = self.epsilon_start  # Reset exploration rate
            
            print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f}")
            
            if (e+1) % 50 == 0:
                self.update_target_network()
        
        return rewards_history

    def save_model(self, filename="smart_city_dqn.pth"):
        """Save trained weights"""
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="smart_city_dqn.pth"):
        """Load pretrained weights"""
        self.model.load_state_dict(torch.load(filename))
        self.update_target_network()

# Training and evaluation
if __name__ == "__main__":
    env = SmartCityEnv()
    agent = EnhancedAgent(env)
    
    # Train the agent
    rewards = agent.train(episodes=50)
    agent.save_model()
    
    # Evaluate
    test_episodes = 10
    for _ in range(test_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            
        print(f"Test Reward: {total_reward:.2f}")
    
    env.close()