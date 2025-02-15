import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gym
import pygame
from pygame.locals import *
from gym import spaces

# تعريف البيئة مع التحسينات
class SmartCityEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        super(SmartCityEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=20, shape=(4,), dtype=np.int32)
        
        # بناء الشبكة الطرقية
        self.track = np.zeros((21, 21), dtype=int)
        self._build_roads()
        
        self.tower_points = {
            (10, 10): "A", (18, 18): "B", (18, 2): "C",
            (2, 18): "D", (2, 2): "E"
        }
        
        # إعدادات الرسومات
        self.tile_size = 32
        self.screen = None
        self.clock = None
        self.reset()

    def _build_roads(self):
        self.track[0:3, :] = 1    # North horizontal
        self.track[9:12, :] = 1   # Central horizontal
        self.track[18:21, :] = 1 # South horizontal
        self.track[:, 0:3] = 1    # West vertical
        self.track[:, 9:12] = 1   # Central vertical
        self.track[:, 18:21] = 1  # East vertical

    def get_allowed_actions(self, position):
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
        valid_points = list(self.tower_points.keys())
        self.agent_pos = random.choice(valid_points)
        self.target = random.choice([p for p in valid_points if p != self.agent_pos])
        return np.array([*self.agent_pos, *self.target], dtype=np.int32)

    def step(self, action):
        r, c = self.agent_pos
        allowed = self.get_allowed_actions(self.agent_pos)
        reward = -0.05  # عقوبة الوقت الأساسية
        
        if action not in allowed:
            reward -= 0.5  # عقوبة الحركة غير القانونية
            return self._get_obs(), reward, False, {}
        
        # حساب الحركة
        dr = [-1, 1, 0, 0][action]
        dc = [0, 0, -1, 1][action]
        new_pos = (r + dr, c + dc)
        
        if self._is_valid(new_pos):
            self.agent_pos = new_pos
            reward += 0.1  # مكافأة الحركة الصحيحة
            
        # حساب المكافآت
        if self.agent_pos == self.target:
            return self._get_obs(), 20.0, True, {}
        
        # مكافأة حسب القرب من الهدف
        dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target))
        reward += (1.0 / (dist + 1)) * 0.5
        
        return self._get_obs(), reward, False, {}

    def _is_valid(self, pos):
        return (0 <= pos[0] < 21) and (0 <= pos[1] < 21) and self.track[pos] == 1

    def _get_obs(self):
        return np.array([*self.agent_pos, *self.target], dtype=np.int32)

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((672, 672))
            self.clock = pygame.time.Clock()
        
        colors = {
            'road': (80, 80, 80),
            'divider': (200, 200, 0),
            'agent': (255, 0, 0),
            'target': (0, 200, 0),
            'points': (160, 32, 240)
        }
        
        self.screen.fill((0, 0, 0))
        for r in range(21):
            for c in range(21):
                if self.track[r, c] == 1:
                    rect = (c*32, r*32, 32, 32)
                    pygame.draw.rect(self.screen, colors['road'], rect)
                    
                    # رسم خطوط التوجيه
                    if r in {1,10,19} and c not in {0,1,2,9,10,11,18,19,20}:
                        pygame.draw.line(self.screen, colors['divider'],
                                       (c*32, r*32+16), (c*32+32, r*32+16), 2)
                    if c in {1,10,19} and r not in {0,1,2,9,10,11,18,19,20}:
                        pygame.draw.line(self.screen, colors['divider'],
                                       (c*32+16, r*32), (c*32+16, r*32+32), 2)
                
                # رسم النقاط
                if (r,c) in self.tower_points:
                    color = colors['target'] if (r,c) == self.target else colors['points']
                    pygame.draw.circle(self.screen, color, (c*32+16, r*32+16), 10)
        
        # رسم الوكيل
        pygame.draw.rect(self.screen, colors['agent'],
                       (self.agent_pos[1]*32, self.agent_pos[0]*32, 32, 32))
        
        pygame.display.flip()
        self.clock.tick(10)
        return None if mode == 'human' else pygame.surfarray.array3d(self.screen)

    def close(self):
        pygame.quit()

# خوارزمية DQN المحسنة
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
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
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

class AdvancedDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            NoisyLinear(input_size, 128),  # استخدام الطبقة الضوضائية
            nn.ReLU(),
            NoisyLinear(128, 256),
            nn.ReLU(),
            NoisyLinear(256, output_size)
        )
        
    def forward(self, x):
        return self.net(x)
    
    def reset_noise(self):
        for layer in self.net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

class RLAgent:
    def __init__(self, env, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, lr=0.001):
        self.env = env
        self.state_size = 4
        self.action_size = 4
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = 0.98  # معدل اضمحلال أبطأ
        self.batch_size = 128  # زيادة حجم الدفعة
        self.memory = deque(maxlen=200000)  # زيادة حجم الذاكرة
        
        self.model = AdvancedDQN(self.state_size, self.action_size)
        self.target_model = AdvancedDQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_network()
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            allowed = self.env.get_allowed_actions((state[0], state[1]))
            return random.choice(allowed) if allowed else 0
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state_t)
            self.model.train()
            allowed = self.env.get_allowed_actions((state[0], state[1]))
            if not allowed:
                return 0
            q_values = q_values.squeeze().numpy()
            return allowed[np.argmax(q_values[allowed]) if allowed else 0 ]
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # تقييد التدرجات
        self.optimizer.step()
        
        # إعادة توليد الضوضاء بعد التحديث
        self.model.reset_noise()
        self.target_model.reset_noise()
    
    def train(self, episodes=800, update_target_every=50):
        rewards_history = []
        epsilon_step = (self.epsilon_start - self.epsilon_end) / episodes  # اضمحلال خطي
        
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
                
            # تحديث إبسيلون (خطي)
            self.epsilon = max(self.epsilon_end, self.epsilon - epsilon_step)
            
            if e % update_target_every == 0:
                self.update_target_network()
                
            rewards_history.append(total_reward)
            print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f}")
        
        return rewards_history
    def save_model(self, filename="smart_city_dqn.pth"):
            torch.save(self.model.state_dict(), filename)
            print(f"Model saved to {filename}")

env = SmartCityEnv()
agent = RLAgent(env)

# بدء التدريب
rewards = agent.train(episodes=10000)
agent.save_model()
# اختبار النموذج
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
    print(f"Test Episode Reward: {total_reward:.2f}")

env.close()