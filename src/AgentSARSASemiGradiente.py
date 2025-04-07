import numpy as np
from collections import Counter

class AgentSARSASemiGradiente:
    def __init__(self, env, alpha=0.01, gamma=0.99, epsilon=0.1, seed=42):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.weights = np.zeros((self.action_dim, self.state_dim))

        np.random.seed(seed)
        np.random.default_rng(seed)
        self.env.reset(seed=seed)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.weights @ state
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action):
        q_sa = self.weights[action] @ state
        q_next = self.weights[next_action] @ next_state
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_sa
        self.weights[action] += self.alpha * td_error * state

    def train(self, env, n_episodes=500):
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

            action = self.get_action(state)
            action_counts[action] += 1
            episode_pole_angles = []
            episode_cart_positions = []

            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = np.array(next_state, dtype=np.float32)

                # Guardamos valores por paso
                episode_cart_positions.append(state[0])   # posición del carrito
                episode_pole_angles.append(state[2])      # ángulo del poste

                next_action = self.get_action(next_state)
                action_counts[action] += 1
                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            lengths.append(steps)
            cart_positions[ep] = episode_cart_positions
            pole_angles[ep] = episode_pole_angles

        return rewards, lengths, action_counts, pole_angles, cart_positions
