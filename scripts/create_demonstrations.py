import argparse
import os
import time
import pickle
import numpy as np
import torch
from src.env import CustomMountainCar

def value_iteration(
    transition_matrix, reward, gamma, softmax=True, 
    alpha=1., max_iter=100, tol=1e-5
    ):
    """
    Args:
        transition_matrix (torch.tensor): transition matrix [act_dim, state_dim, state_dim]
        reward (torch.tensor): reward vector [state_dim]
        gamma (float): discount factor
        softmax (bool): whether to use soft value iteration. Default=True
        alpha (float): softmax temperature
        max_iter (int): max iteration. Default=100
        tol (float): error tolerance. Default=1e-5

    Returns:
        q (torch.tensor): q function [state_dim, act_dim]
        info (dict): {"tol", "iter"}
    """
    start = time.time()
    state_dim = transition_matrix.shape[1]
    act_dim = transition_matrix.shape[0]
    
    q = [torch.zeros(state_dim, act_dim)]
    for t in range(max_iter):
        if softmax:
            v = torch.logsumexp(alpha * q[t], dim=-1) / alpha
        else:
            v = q[t].max(-1)[0]
        q_t = torch.sum(transition_matrix * (reward + gamma * v).view(1, 1, -1), dim=-1).T
        q.append(q_t)

        q_error = torch.abs(q_t - q[t]).mean()
        if q_error < tol:
            break
    
    tnow = time.time() - start
    return q[-1], {"tol": q_error.item(), "iter": t, "time": tnow}

def episode(env, policy, max_steps=500):
    data = {"obs": [], "act": [], "reward": [], "next_obs": [], "done": []}
    obs = env.reset()
    for t in range(max_steps):
        s = env.obs2state(obs)[0]
        a = torch.multinomial(policy[s], 1).numpy()[0]

        next_obs, reward, done, into = env.step(a)

        data["obs"].append(obs)
        data["act"].append(a)
        data["reward"].append(reward)
        data["next_obs"].append(next_obs)
        data["done"].append(1 if done else 0)
        
        if done:
            break
        obs = next_obs
    return data

def main(arglist):
    seed = 0
    x_bins = 20
    v_bins = 20
    env = CustomMountainCar(x_bins=x_bins, v_bins=v_bins, seed=seed)
    env.make_initial_distribution()
    env.make_transition_matrix()

    transition_matrix = torch.from_numpy(env.transition_matrix)
    reward = torch.from_numpy(env.reward)

    gamma = 0.99 # discount factor
    alpha = 10 # softmax temperature
    max_iter = 2000

    q_soft, info = value_iteration(
        transition_matrix, reward, gamma, softmax=True, alpha=alpha, max_iter=max_iter
    )
    print(f"soft value iteration info: {info}")
    
    # expert policy
    beta = 200
    policy = torch.softmax(beta * q_soft, dim=-1)
    
    dataset = []
    for i in range(arglist.num_eps):
        data = episode(env, policy, max_steps=arglist.max_steps)
        dataset.append(data)
    
    # print demonstration stats
    eps_len = [len(d["reward"]) for d in dataset]
    print(f"demonstration performance: min={np.min(eps_len)}, max={np.max(eps_len)}, mean={np.mean(eps_len):.2f}")
    
    save_path = "../data"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, "data.p"), "wb") as f:
        pickle.dump(dataset, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_eps", type=int, default=50, help="number of demonstration episodes, default=50")
    parser.add_argument("--max_steps", type=int, default=500, help="max number of steps in demonstration episodes, default=500")
    arglist = parser.parse_args()
    return arglist

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)