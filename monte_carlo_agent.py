import numpy as np
import random
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, nA, gamma=0.99, epsilon=0.1):
        """
        nA: 動作數量
        gamma: 折扣因子
        epsilon: epsilon-greedy 的機率
        """
        self.nA = nA  
        self.gamma = gamma
        self.epsilon = epsilon
        # Q[s] 為一個長度為 nA 的 numpy array
        self.Q = defaultdict(lambda: np.zeros(nA))
        # 用來累積每個狀態-動作組合的回報
        self.returns_sum = defaultdict(lambda: np.zeros(nA))
        self.returns_count = defaultdict(lambda: np.zeros(nA))
    
    def select_action(self, state):
        """根據 epsilon-greedy 策略從 Q 值中選擇動作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        else:
            return int(np.argmax(self.Q[state]))
    
    def generate_episode(self, env, discretize_fn, max_steps=1000, render=False):
        """
        與環境互動產生一個完整的 episode
        discretize_fn 用來將原始觀察轉換為離散狀態
        render: 若 True，則每步呼叫 env.render() 顯示畫面
        """
        episode = []
        state = discretize_fn(reset_env(env))
        done = False
        steps = 0
        while not done and steps < max_steps:
            if render:
                env.render()
            action = self.select_action(state)
            next_obs, reward, done, _ = step_env(env, action)
            next_state = discretize_fn(next_obs)
            episode.append((state, action, reward))
            state = next_state
            steps += 1
        return episode
    
    def update(self, episode):
        """
        使用第一訪問（first-visit）方法依 episode 更新 Q 值
        """
        visited = set()
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                self.returns_sum[state][action] += G
                self.returns_count[state][action] += 1.0
                self.Q[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]
    
    def get_policy(self):
        """回傳目前學到的策略（對每個狀態選擇使 Q 最大的動作）"""
        policy = {}
        for state, actions in self.Q.items():
            policy[state] = int(np.argmax(actions))
        return policy

# 為了方便新版 Gym API 封裝 reset 與 step
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
