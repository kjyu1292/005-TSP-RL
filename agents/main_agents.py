import numpy as np
import random

from agents.base_agent import Agent

class QAgent(Agent):
    def __init__(self, states_size, actions_size, epsilon = 0.999, epsilon_min = 0.01, epsilon_decay = 0.999, gamma = 0.95, lr = 0.8):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = np.zeros([states_size, actions_size])
        self.reset_memory()

    def train(self, s, a, r, s_next):
        self.Q[s,a] = self.Q[s,a] + self.lr * (r + self.gamma*np.max(self.Q[s_next,a]) - self.Q[s,a])

    def act(self, s):
        q = np.copy(self.Q[s,:])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if random.uniform(0, 1) > self.epsilon:
            a = np.argmax(q)
        else:
            a = random.choice([x for x in range(self.actions_size) if x not in self.states_memory])
        return a
    
    def remember_state(self,s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []


class SARSAgent(Agent):
    def __init__(self, states_size, actions_size, epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.999, gamma = 0.95, lr = 0.8):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = np.zeros([states_size, actions_size])
        self.reset_memory()

    def train(self, s, a, r, s_next, a_next):
        self.Q[s,a] = self.Q[s,a] + self.lr * (r + self.gamma * self.Q[s_next, a_next] - self.Q[s,a])

    def act(self, s):
        q = np.copy(self.Q[s,:])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if random.uniform(0, 1) > self.epsilon:
            a = np.argmax(q)
        else:
            a = random.choice([x for x in range(self.actions_size) if x not in self.states_memory])
        return a
    
    def remember_state(self,s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []


class DoubleQAgent(Agent):
    def __init__(self, states_size, actions_size, epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.999, gamma = 0.95, lr = 0.8):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q_A = np.zeros([states_size, actions_size])
        self.Q_B = np.zeros([states_size, actions_size])
        self.reset_memory()

    def train(self, s, a, r, s_next, probability = random.uniform(0, 1)):
        if probability > 0.5:
            self.Q_A[s,a] = self.Q_A[s,a] + self.lr * (r + self.gamma*np.max(self.Q_B[s_next,a]) - self.Q_A[s,a])
        elif probability <= 0.5:
            self.Q_B[s,a] = self.Q_B[s,a] + self.lr * (r + self.gamma*np.max(self.Q_A[s_next,a]) - self.Q_B[s,a])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, s):

        q_a = self.Q_A[s,:]
        q_b = self.Q_B[s,:]
        q = np.average(
            np.concatenate(
                (q_a.reshape(-1, 1), q_b.reshape(-1, 1)), 
                axis = 1
            ), axis = 1
        )

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if random.uniform(0, 1) > self.epsilon:
            a = np.argmax(q)
        else:
            a = random.choice([x for x in range(self.actions_size) if x not in self.states_memory])
        return a

    def remember_state(self,s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []


class DelayedQAgent(Agent):
    def __init__(self, states_size, actions_size, epsilon = 0.1, gamma = 0.95, m = 2):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.m = m

        self.Q = np.zeros([states_size, actions_size]) + (1 / (1 - self.gamma))
        self.U = np.zeros([states_size, actions_size])
        self.l = np.zeros([states_size, actions_size])
        self.t = np.zeros([states_size, actions_size])
        self.LEARN = np.ones([states_size, actions_size])

        self.t_ = 0
        self.reset_memory()

    def act(self, s):

        q = np.copy(self.Q[s,:])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        a = np.argmax(q)
        return a

    def remember_state(self,s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []