import numpy as np
import gymnasium as gym
from tqdm import tqdm

class AgentQLearning:
    def __init__(self, env, num_episodes=100000, gamma=0.95, alpha=0.1,
                 initial_epsilon=1.0, min_epsilon=0.01, seed=42):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = initial_epsilon
        self.seed = seed

        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        self.Q = np.zeros((self.nS, self.nA))

        self.episode_lengths = []
        self.episode_rewards = []
        self.q_deltas = []

        self.epsilon_decay = (initial_epsilon - min_epsilon) / num_episodes

        np.random.seed(seed)
        self.env.reset(seed=seed)

    def get_epsilon_greedy_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state])

    def train(self):
        successful_episodes = 0

        for episode in tqdm(range(self.num_episodes), desc="Training"):
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            old_Q = self.Q.copy()

            while not done:
                action = self.get_epsilon_greedy_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Q-Learning update
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state][best_next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error

                state = next_state
                total_reward += reward
                steps += 1

            self.episode_lengths.append(steps)
            self.episode_rewards.append(total_reward)
            self.q_deltas.append(np.mean(np.abs(self.Q - old_Q)))

            if reward > 0:
                successful_episodes += 1

        print(f"\nSuccess rate: {successful_episodes}/{self.num_episodes} ({100*successful_episodes/self.num_episodes:.2f}%)")

    def get_optimal_policy(self):
        return np.argmax(self.Q, axis=1)

    def test(self, num_tests=100):
        successes = 0
        for _ in range(num_tests):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.Q[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
            if reward > 0:
                successes += 1
        print(f"Test success rate: {successes}/{num_tests} ({100*successes/num_tests:.2f}%)")
