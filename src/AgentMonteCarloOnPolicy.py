import numpy as np
import gymnasium as gym
from tqdm import tqdm

class AgentMonteCarloOnPolicy:
    def __init__(self, env, num_episodes=100000, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.01, seed=42):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.seed = seed
        
        # Action and state spaces
        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        
        # Q-table and returns
        self.Q = np.zeros((self.nS, self.nA))
        self.returns = {(s, a): [] for s in range(self.nS) for a in range(self.nA)}
        
        # For epsilon decay
        self.epsilon_decay = (initial_epsilon - min_epsilon) / num_episodes
        
        np.random.seed(seed)
        self.env.reset(seed=seed)
    
    def get_epsilon_greedy_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        return np.argmax(self.Q[state])  # Greedy action
    
    def train(self):
        successful_episodes = 0
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
            
            # Generate episode
            episode_history = []
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.get_epsilon_greedy_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_history.append((state, action, reward))
                state = next_state
            
            # Check if episode was successful
            if reward > 0:
                successful_episodes += 1
            
            # Calculate returns and update Q
            G = 0
            for t in reversed(range(len(episode_history))):
                state, action, reward = episode_history[t]
                G = self.gamma * G + reward
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
        
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