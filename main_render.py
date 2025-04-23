import gym
import argparse
import numpy as np
from monte_carlo_agent import MonteCarloAgent, reset_env, step_env
from q_learning_agent import QLearningAgent

# 若使用 RAM 版，採用原本簡單離散化（只取前 5 維）
def discretize_state_ram(state):
    return tuple((np.array(state)[:5] // 32).astype(int))

# 若使用圖像版，定義一個簡單的離散化：將 RGB 影像轉為灰階、縮小尺寸後量化
def discretize_state_image(state):
    from PIL import Image
    # 假設 state 是 numpy array (height, width, 3)
    img = Image.fromarray(state)
    # 轉為灰階
    img = img.convert('L')
    # 縮小尺寸 (例如 11 x 8)
    img = img.resize((11, 8))
    state_arr = (np.array(img) // 32).astype(int)
    return tuple(state_arr.flatten())

def run_monte_carlo(env, discretize_fn, num_episodes=500, render=False):
    nA = env.action_space.n
    agent = MonteCarloAgent(nA)
    for episode in range(num_episodes):
        ep = agent.generate_episode(env, discretize_fn, render=render)
        agent.update(ep)
        if (episode+1) % 50 == 0:
            print(f"Monte Carlo 訓練: 完成 {episode+1}/{num_episodes} 集")
    return agent

def run_q_learning(env, discretize_fn, num_episodes=500, max_steps=1000, render=False):
    nA = env.action_space.n
    agent = QLearningAgent(nA)
    # 取得初始 state
    state = discretize_fn(reset_env(env))
    for episode in range(num_episodes):
        state = discretize_fn(reset_env(env))
        done = False
        steps = 0
        while not done and steps < max_steps:
            if render:
                env.render()
            action = agent.select_action(state)
            next_obs, reward, done, _ = step_env(env, action)
            next_state = discretize_fn(next_obs)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        if (episode+1) % 50 == 0:
            print(f"Q-Learning 訓練: 完成 {episode+1}/{num_episodes} 集")
    return agent

def evaluate_policy(env, agent, discretize_fn, episodes=10, max_steps=1000, algorithm="MC", render=False):
    total_rewards = []
    for i in range(episodes):
        state = discretize_fn(reset_env(env))
        done = False
        ep_reward = 0
        steps = 0
        while not done and steps < max_steps:
            if render:
                env.render()
            # 採用貪婪策略
            action = int(np.argmax(agent.Q[state]))
            next_obs, reward, done, _ = step_env(env, action)
            state = discretize_fn(next_obs)
            ep_reward += reward
            steps += 1
        total_rewards.append(ep_reward)
        print(f"評估第 {i+1} 集：reward = {ep_reward}")
    avg_reward = np.mean(total_rewards)
    print(f"{algorithm} 演算法在 {episodes} 集中的平均 reward = {avg_reward}")
    return total_rewards

def main():
    parser = argparse.ArgumentParser(description="MsPacman 強化學習專案：顯示遊戲過程 & 演算法切換")
    parser.add_argument("--algo", choices=["mc", "q"], default="q", help="選擇演算法：mc (Monte Carlo) 或 q (Q-Learning)")
    parser.add_argument("--train_episodes", type=int, default=500, help="訓練集數")
    parser.add_argument("--eval_episodes", type=int, default=10, help="評估集數")
    parser.add_argument("--env", choices=["ram", "image"], default="ram", help="環境版本：ram (僅用離散化不顯示) 或 image (用影像版，可顯示遊戲過程)")
    parser.add_argument("--render", action="store_true", help="是否顯示遊戲過程")
    args = parser.parse_args()

    # 根據選擇的環境版本與是否顯示決定使用哪個 gym 環境
    if args.env == "ram":
        # RAM 版不支援真實遊戲視覺效果，但可用於較快的表格方法學習
        env = gym.make("MsPacman-ram-v0", render_mode="human" if args.render else None)
        discretize_fn = discretize_state_ram
    else:
        # 使用圖像版，並設置 render_mode 為 "human" 以呈現視覺畫面
        env = gym.make("MsPacman-v0", render_mode="human" if args.render else None)
        discretize_fn = discretize_state_image

    print("使用環境:", env.spec.id)

    if args.algo == "mc":
        print("使用 Monte Carlo Control 進行訓練...")
        agent = run_monte_carlo(env, discretize_fn, num_episodes=args.train_episodes, render=args.render)
        evaluate_policy(env, agent, discretize_fn, episodes=args.eval_episodes, algorithm="Monte Carlo", render=args.render)
    else:
        print("使用 Q-Learning 進行訓練...")
        agent = run_q_learning(env, discretize_fn, num_episodes=args.train_episodes, render=args.render)
        evaluate_policy(env, agent, discretize_fn, episodes=args.eval_episodes, algorithm="Q-Learning", render=args.render)

    env.close()

if __name__ == "__main__":
    main()
