import numpy as np
import gymnasium as gym
from tqdm import tqdm

class AgentMonteCarloOffPolicy:
    def __init__(self, env, num_episodes=100000, gamma=0.95, epsilon=0.2, seed=42):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.seed = seed
        
        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        
        self.Q = np.zeros((self.nS, self.nA))
        self.C = np.zeros((self.nS, self.nA))  # Acumulador de pesos
        
        np.random.seed(seed)
        self.env.reset(seed=seed)

    def behavior_policy(self, state):
        # Epsilon-greedy
        probs = np.ones(self.nA) * (self.epsilon / self.nA)
        best_action = np.argmax(self.Q[state])
        probs[best_action] += (1 - self.epsilon)
        return probs

    def target_policy(self, state):
        return np.argmax(self.Q[state])

    def train(self):
        for episode in tqdm(range(self.num_episodes), desc="Training (Off-Policy)"):
            state, _ = self.env.reset()
            episode_history = []
            done = False

            # Generate episode using behavior policy
            while not done:
                action_probs = self.behavior_policy(state)
                action = np.random.choice(self.nA, p=action_probs)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_history.append((state, action, reward, action_probs[action]))
                state = next_state
                done = terminated or truncated

            G = 0
            W = 1  # Importance weight

            # Go backwards through the episode
            for t in reversed(range(len(episode_history))):
                state, action, reward, prob_b = episode_history[t]
                G = self.gamma * G + reward

                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])

                # Stop if behavior diverges from target (i.e., action ≠ greedy action)
                if action != self.target_policy(state):
                    break

                W /= prob_b  # Update importance sampling weight

    def test(self, num_tests=100):
        successes = 0
        for _ in range(num_tests):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.target_policy(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
            if reward > 0:
                successes += 1
        print(f"Test success rate: {successes}/{num_tests} ({100*successes/num_tests:.2f}%)")
