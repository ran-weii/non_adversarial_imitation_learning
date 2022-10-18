import argparse
import os
import time
import json
import numpy as np
import torch
from src.env import CustomMountainCar
from src.ail import DAC

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../results")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--num_eps", type=int, default=3)
    parser.add_argument("--render", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def episode(env, agent, max_steps=500, render=True):
    agent.eval()

    data = {"obs": [], "act": [], "reward": [], "next_obs": [], "done": []}
    obs = env.reset()
    
    for t in range(max_steps):
        a = agent.choose_action(obs)

        next_obs, reward, done, into = env.step(a)

        data["obs"].append(obs)
        data["act"].append(a)
        data["reward"].append(reward)
        data["next_obs"].append(next_obs)
        data["done"].append(1 if done else 0)
        
        if done:
            break
        obs = next_obs
        
        if render:
            env.render()
    return data

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)

    exp_path = os.path.join(arglist.exp_path, arglist.exp_name)

    # load args
    with open(os.path.join(exp_path, "args.json"), "r") as f:
        config = json.load(f)
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "models", "model_30.pt"), map_location=torch.device("cpu"))

    obs_dim = 2
    act_dim = 3
    agent = DAC(
        obs_dim, act_dim, config["hidden_dim"], config["num_hidden"], config["activation"], 
        algo=config["algo"], norm_obs=config["norm_obs"], 
    )
    agent.load_state_dict(state_dict, strict=True)
    print(agent)

    x_bins = 20
    v_bins = 20
    env = CustomMountainCar(x_bins=x_bins, v_bins=v_bins, seed=arglist.seed)
    
    scores = []
    for e in range(arglist.num_eps):
        data = episode(env, agent, render=arglist.render)
        scores.append(len(data['reward']))
        print(f"score: {scores[-1]}")

    print(f"scores mean: {np.mean(scores), np.std(scores)}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)