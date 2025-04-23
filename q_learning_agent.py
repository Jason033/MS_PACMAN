import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, nA, gamma=0.99, alpha=0.1, epsilon=0.1):
        """
        nA: 動作數量
        gamma: 折扣因子
        alpha: 學習率
        epsilon: epsilon-greedy 的機率
        """
        self.nA = nA  
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(nA))
    
    def select_action(self, state):
        """根據 epsilon-greedy 策略選擇動作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        else:
            return int(np.argmax(self.Q[state]))
    
    def learn(self, state, action, reward, next_state, done):
        """依 Q-Learning 更新規則更新 Q 值"""
        best_next = 0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta

# 同 monte_carlo_agent 的 reset/step 輔助函式也適用
def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result

def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = terminated or truncated
        return observation, reward, done, info
    return result
