# -*- coding: utf-8 -*-
#dynamic_rl_agents.py


#Динамические RL-агенты для задачи ATSP с меняющимися пробками.
#Каждый агент получает состояние динамической среды как:
#    current (int), visited_mask (np.array bool), time_vector (np.array float32).
#Внутри агента состояние преобразуется в словарь с ключами {'current', 'visited_mask', 'dist_vector'},
# где поле 'dist_vector' заполняется вектором времени в часах.
# Это позволяет использовать нейросети из статического rl_agents.py без изменений.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
import random
from collections import deque

# Статические архитектуры нейросетей
from rl_agents import (
    PolicyNetwork, EnhancedPolicyNetwork,
    DuelingQNetwork, NoisyDuelingQNetwork,
    PrioritizedReplayBuffer
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy Gradient (базовый)
class DynamicPolicyGradientAgent:
    """
    Динамический Policy Gradient (REINFORCE).
    Использует базовую PolicyNetwork из rl_agents.
    Входной вектор: one_hot(current) + visited_mask + time_vector.
    """
    def __init__(self, n_warehouses, learning_rate=0.001, gamma=0.99):
        self.policy_network = PolicyNetwork(n_warehouses).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.saved_log_probs = []   # логарифмы вероятностей выбранных действий
        self.rewards = []           # пошаговые награды (-travel_h)

    def select_action(self, current, visited_mask, time_vector, greedy=False):
        """
        Принимает состояние динамической среды, формирует тензоры,
        передаёт их в сеть и сэмплирует действие.
        """
        cur_t = torch.tensor([current], dtype=torch.long, device=DEVICE)
        vis_t = torch.tensor([visited_mask], dtype=torch.float32, device=DEVICE)
        time_t = torch.tensor([time_vector], dtype=torch.float32, device=DEVICE)
        # Используем ключ 'dist_vector' для совместимости со статическими сетями
        state = {'current': cur_t, 'visited_mask': vis_t, 'dist_vector': time_t}
        probs = self.policy_network(state)
        m = dist.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        """
        Вычисляет дисконтированные возвраты по наградам и выполняет шаг REINFORCE.
        Сбрасывает накопленные данные после обновления.
        """
        if not self.rewards:
            return
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []

# Enhanced Policy Gradient
class DynamicEnhancedPolicyGradientAgent:
    """
    Улучшенный Policy Gradient с энтропийной регуляризацией,
    cosine annealing и gradient clipping.
    Использует EnhancedPolicyNetwork из rl_agents.
    """
    def __init__(self, n_warehouses, learning_rate=0.001, gamma=0.99, entropy_coef=0.01):
        self.policy_network = EnhancedPolicyNetwork(n_warehouses).to(DEVICE)
        self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1500)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.saved_log_probs = []   # log_prob действий
        self.rewards = []           # пошаговые награды
        self.entropies = []         # энтропия распределений

    def select_action(self, current, visited_mask, time_vector, greedy=False):
        """Аналогично базовому агенту, но сохраняет также энтропию."""
        cur_t = torch.tensor([current], dtype=torch.long, device=DEVICE)
        vis_t = torch.tensor([visited_mask], dtype=torch.float32, device=DEVICE)
        time_t = torch.tensor([time_vector], dtype=torch.float32, device=DEVICE)
        state = {'current': cur_t, 'visited_mask': vis_t, 'dist_vector': time_t}
        probs = self.policy_network(state)
        m = dist.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        self.entropies.append(m.entropy())
        return action.item()

    def update(self):
        """
        Вычисляет возвраты, добавляет энтропийный бонус и выполняет шаг оптимизации.
        Применяет gradient clipping и шаг планировщика lr.
        """
        if not self.rewards:
            return
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        entropy = torch.stack(self.entropies).sum()
        loss = torch.stack(policy_loss).sum() - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []

# Improved DQN (dynamic)
class DynamicImprovedDQNAgent:
    """
    Улучшенный DQN с Duelling архитектурой, PER, мягким обновлением целевой сети.
    Использует DuelingQNetwork / NoisyDuelingQNetwork из rl_agents.
    Поддерживает опциональные Noisy слои вместо ε‑greedy.
    """
    def __init__(self, n_warehouses, learning_rate=0.0003, gamma=0.99, tau=0.005,
                 use_per=True, use_noisy=False):
        self.use_noisy = use_noisy
        if use_noisy:
            self.q_network = NoisyDuelingQNetwork(n_warehouses).to(DEVICE)
            self.target_network = NoisyDuelingQNetwork(n_warehouses).to(DEVICE)
        else:
            self.q_network = DuelingQNetwork(n_warehouses).to(DEVICE)
            self.target_network = DuelingQNetwork(n_warehouses).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1500)
        self.gamma = gamma
        self.tau = tau                     # коэффициент мягкого обновления
        self.use_per = use_per
        if use_per:
            self.memory = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
            self.beta = 0.4
            self.beta_increment = 0.001
        else:
            self.memory = deque(maxlen=10000)
        self.batch_size = 128
        if not use_noisy:
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
        else:
            self.epsilon = 0.0

    def select_action(self, current, visited_mask, time_vector, greedy=False):
        """
        Выбор действия:
          - если greedy=False, используется ε‑greedy (или argmax при Noisy),
          - если greedy=True, всегда argmax (для тестирования).
        """
        if not greedy and not self.use_noisy and random.random() < self.epsilon:
            valid = [i for i, v in enumerate(visited_mask) if v == 0]
            return random.choice(valid) if valid else 0
        cur_t = torch.tensor([current], dtype=torch.long, device=DEVICE)
        vis_t = torch.tensor([visited_mask], dtype=torch.float32, device=DEVICE)
        time_t = torch.tensor([time_vector], dtype=torch.float32, device=DEVICE)
        state = {'current': cur_t, 'visited_mask': vis_t, 'dist_vector': time_t}
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """
        Сохраняет переход в буфер.
        state и next_state – кортежи (current, visited_mask, time_vector).
        """
        experience = (state, action, reward, next_state, done)
        if self.use_per:
            priority = self.memory.max_priority()
            self.memory.add(experience, priority)
        else:
            self.memory.append(experience)

    def update(self):
        """Шаг обучения на мини-батче из буфера."""
        if len(self.memory) < self.batch_size:
            return
        if self.use_per:
            batch, indices, weights = self.memory.sample(self.batch_size, self.beta)
            if len(batch) < self.batch_size:
                return
            self.beta = min(1.0, self.beta + self.beta_increment)
            weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        else:
            batch = random.sample(list(self.memory), min(self.batch_size, len(self.memory)))
            weights = torch.ones(len(batch), device=DEVICE)

        states, actions, rewards, next_states, dones = zip(*batch)
        # Извлекаем компоненты состояний в тензоры
        state_cur  = torch.tensor([s[0] for s in states], dtype=torch.long, device=DEVICE)
        state_vis  = torch.tensor([s[1] for s in states], dtype=torch.float32, device=DEVICE)
        state_time = torch.tensor([s[2] for s in states], dtype=torch.float32, device=DEVICE)
        next_cur   = torch.tensor([s[0] for s in next_states], dtype=torch.long, device=DEVICE)
        next_vis   = torch.tensor([s[1] for s in next_states], dtype=torch.float32, device=DEVICE)
        next_time  = torch.tensor([s[2] for s in next_states], dtype=torch.float32, device=DEVICE)

        state_tensors = {'current': state_cur, 'visited_mask': state_vis, 'dist_vector': state_time}
        next_tensors  = {'current': next_cur,  'visited_mask': next_vis,  'dist_vector': next_time}
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones   = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            next_actions = self.q_network(next_tensors).argmax(1)
            next_q = self.target_network(next_tensors)[torch.arange(len(batch), device=DEVICE), next_actions]
            targets = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.q_network(state_tensors)[torch.arange(len(batch), device=DEVICE), actions]
        td_errors = targets - current_q
        loss = (weights * td_errors.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Мягкое обновление целевой сети
        for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        if self.use_per:
            self.memory.update_priorities(indices, td_errors.detach().abs().cpu().numpy())
        if self.use_noisy and hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Improved Double DQN (dynamic)
class DynamicImprovedDoubleDQNAgent(DynamicImprovedDQNAgent):
    """
    Улучшенный Double DQN с периодическим жёстким копированием весов.
    Наследует всю логику от DynamicImprovedDQNAgent.
    """
    def __init__(self, n_warehouses, **kwargs):
        super().__init__(n_warehouses, **kwargs)
        self.batch_size = 256           # увеличенный размер батча
        self.update_step = 0

    def update(self):
        """Выполняет обновление и раз в 1000 шагов делает жёсткую замену целевой сети."""
        super().update()
        self.update_step += 1
        if self.update_step % 1000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
