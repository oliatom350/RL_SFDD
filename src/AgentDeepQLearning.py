import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import Counter
from QNet import DQN_Network

class AgentDQLearning:
    def __init__(self, env, gamma=0.99, lr=1e-3, epsilon=1.0, min_epsilon=0.01, decay=0.995, seed=42):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Red neuronal para estimar la Q-función
        self.q_net = DQN_Network(self.action_dim, self.state_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        np.random.seed(seed)
        np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.env.reset(seed=seed)

    def get_action(self, state):
        """Política epsilon-greedy para seleccionar una acción."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()


    def update(self, state, action, reward, next_state, done):
        """Actualiza la red neuronal con un paso de aprendizaje."""
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)

        q_val = self.q_net(state_tensor)[action]

        with torch.no_grad():
            q_next = torch.max(self.q_net(next_state_tensor))
            target = reward if done else reward + self.gamma * q_next.item()

        target_tensor = torch.tensor(target, dtype=torch.float32)

        loss = self.criterion(q_val, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decaimiento de epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


    def train(self, env, n_episodes=500):
        """Entrena el agente en el entorno dado."""
        rewards = []
        lengths = []
        pole_angles = {}
        cart_positions = {}
        action_counts = Counter()

        for ep in range(n_episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            done = False
            total_reward = 0
            steps = 0

            episode_pole_angles = []
            episode_cart_positions = []

            while not done:
                action = self.get_action(state)
                action_counts[action] += 1
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = np.array(next_state, dtype=np.float32)

                # Guardamos valores por paso
                episode_cart_positions.append(state[0])   # posición del carrito
                episode_pole_angles.append(state[2])      # ángulo del poste

                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            lengths.append(steps)
            cart_positions[ep] = episode_cart_positions
            pole_angles[ep] = episode_pole_angles

        # Devuelve también métricas físicas
        return rewards, lengths, action_counts, pole_angles, cart_positions
