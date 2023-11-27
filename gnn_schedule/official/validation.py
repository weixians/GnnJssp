import torch
import numpy as np
from gnn_schedule.env.jssp_env import JsspEnv
from gnn_schedule.official.agent_utils import greedy_select_action
from gnn_schedule.official.Params import configs
from gnn_schedule.official.util import to_tensor


def validate(vali_set, model):
    env = JsspEnv(configs)
    device = torch.device(configs.device)
    make_spans = []
    # rollout using model
    for data in vali_set:
        obs, _ = env.reset(data=data, n_j=data.shape[-2], n_m=data.shape[-1])
        rewards = 0
        while True:
            obs = to_tensor(*obs, device)
            with torch.no_grad():
                pi, _ = model(obs)
            # action = sample_select_action(pi, candidate)
            action = greedy_select_action(pi, obs[2].squeeze())
            obs, reward, done, _, _ = env.step(action.item())
            rewards += reward
            if done:
                break
        make_spans.append(env.t)
        # print(rewards - env.posRewards)
    return np.array(make_spans)
