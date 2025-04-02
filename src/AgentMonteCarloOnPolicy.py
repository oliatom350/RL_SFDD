import numpy as np
import gymnasium as gym

class AgentMonteCarloOnPolicy:
    def __init__(self, env: gym.Env, num_episodes=5000, gamma=1.0, epsilon=0.1):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.nA = env.action_space.n  # Número de acciones
        self.nS = env.observation_space.n  # Número de estados
        self.Q = np.zeros((self.nS, self.nA))  # Inicializamos valores Q en 0
        self.returns_sum = np.zeros((self.nS, self.nA))  # Acumulador de retornos
        self.returns_count = np.zeros((self.nS, self.nA))  # Contador de visitas
    
    def get_action(self, state):
        """ Selecciona una acción siguiendo la política epsilon-greedy """
        return np.random.choice(self.nA, p=self.get_epsilon_greedy_policy(state))
    
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """ Almacena la muestra y actualiza valores Q al final del episodio """
        self.episode_data.append((obs, action, reward))
        if terminated or truncated:
            self.process_episode()
    
    def process_episode(self):
        """ Actualiza la Q-table después de un episodio completo """
        G = 0  # Retorno acumulado
        for state, action, reward in reversed(self.episode_data):
            G = self.gamma * G + reward
            self.returns_sum[state, action] += G
            self.returns_count[state, action] += 1
            self.Q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
        self.episode_data = []  # Reiniciar episodio
    
    def get_epsilon_greedy_policy(self, state):
        """ Genera una política epsilon-greedy """
        pi_A = np.ones(self.nA) * (self.epsilon / self.nA)
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A
    
    def train(self):
        """ Entrena al agente con Monte Carlo """
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            self.episode_data = []
            done = False
            
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.update(obs, action, next_obs, reward, terminated, truncated, info)
                obs = next_obs
                done = terminated or truncated
        
    def stats(self):
        """ Retorna estadísticas del entrenamiento """
        return self.Q